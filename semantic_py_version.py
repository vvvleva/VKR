# C:\Users\Violetta\Desktop\Role\probnyu\RDF_struct\RDFprob.py
import json
import numpy as np
import torch
import re
import random
import warnings
import time
import os
import sys
from typing import List, Dict, Tuple, Any, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import networkx as nx
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

# Скачиваем ресурсы для nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

warnings.filterwarnings("ignore")


class KnowledgeGraphManager:
    """Управление графом знаний для ролей и сервисов"""
    
    def __init__(self):
        self.graph = Graph()
        
        # Пространства имен
        self.ROLE = Namespace("http://example.org/role/")
        self.SERVICE = Namespace("http://example.org/service/")
        self.FUNCTION = Namespace("http://example.org/function/")
        self.SKILL = Namespace("http://example.org/skill/")
        self.ONTOLOGY = Namespace("http://example.org/ontology/")
        
        # Биндим префиксы
        self.graph.bind("role", self.ROLE)
        self.graph.bind("service", self.SERVICE)
        self.graph.bind("function", self.FUNCTION)
        self.graph.bind("skill", self.SKILL)
        self.graph.bind("ontology", self.ONTOLOGY)
        
        # Инициализация онтологии
        self._create_ontology()
        
        # Индексы для быстрого поиска
        self.semantic_index = {}
        self.role_to_service = {}
        self.role_functions = {}
        
    def _create_ontology(self):
        """Создание базовой онтологии"""
        # Классы
        self.graph.add((self.ONTOLOGY.Role, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.Service, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.Function, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.Skill, RDF.type, OWL.Class))
        
        # Свойства
        self.graph.add((self.ONTOLOGY.hasFunction, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.ONTOLOGY.hasFunction, RDFS.domain, self.ONTOLOGY.Role))
        self.graph.add((self.ONTOLOGY.hasFunction, RDFS.range, self.ONTOLOGY.Function))
        
        self.graph.add((self.ONTOLOGY.requiresSkill, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.ONTOLOGY.requiresSkill, RDFS.domain, self.ONTOLOGY.Function))
        self.graph.add((self.ONTOLOGY.requiresSkill, RDFS.range, self.ONTOLOGY.Skill))
        
        self.graph.add((self.ONTOLOGY.partOfService, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.ONTOLOGY.partOfService, RDFS.domain, self.ONTOLOGY.Role))
        self.graph.add((self.ONTOLOGY.partOfService, RDFS.range, self.ONTOLOGY.Service))
        
        self.graph.add((self.ONTOLOGY.similarTo, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.ONTOLOGY.similarTo, RDFS.domain, self.ONTOLOGY.Role))
        self.graph.add((self.ONTOLOGY.similarTo, RDFS.range, self.ONTOLOGY.Role))
        
        # Типы ролей
        self.graph.add((self.ONTOLOGY.AdminRole, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.AdminRole, RDFS.subClassOf, self.ONTOLOGY.Role))
        
        self.graph.add((self.ONTOLOGY.UserRole, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.UserRole, RDFS.subClassOf, self.ONTOLOGY.Role))
        
        self.graph.add((self.ONTOLOGY.ExpertRole, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.ExpertRole, RDFS.subClassOf, self.ONTOLOGY.Role))
        
        self.graph.add((self.ONTOLOGY.ReviewRole, RDF.type, OWL.Class))
        self.graph.add((self.ONTOLOGY.ReviewRole, RDFS.subClassOf, self.ONTOLOGY.Role))
    
    def _extract_skills(self, text: str) -> List[str]:
        """Извлечение навыков из текста"""
        skills = []
        text_lower = text.lower()
        
        # Паттерны для извлечения навыков
        skill_patterns = [
            (r'умение\s+(\w+\s*\w*)', 'умение'),
            (r'знание\s+(\w+\s*\w*)', 'знание'),
            (r'навык\w*\s+(\w+\s*\w*)', 'навык'),
            (r'способность\s+(\w+\s*\w*)', 'способность'),
            (r'опыт\s+(\w+\s*\w*)', 'опыт'),
            (r'владеть\s+(\w+\s*\w*)', 'владение'),
            (r'работать\s+с\s+(\w+\s*\w*)', 'работа с'),
        ]
        
        for pattern, skill_type in skill_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                skills.append(f"{skill_type}: {match}")
        
        # Дополнительные навыки по контексту
        if any(term in text_lower for term in ['отчет', 'документ', 'акт', 'справка']):
            skills.append('документооборот')
        if any(term in text_lower for term in ['финанс', 'бюджет', 'средств', 'экономическ']):
            skills.append('финансовая грамотность')
        if any(term in text_lower for term in ['контрол', 'проверк', 'аудит', 'реценз']):
            skills.append('контроль качества')
        if any(term in text_lower for term in ['исследован', 'анализ', 'разработк', 'научн']):
            skills.append('аналитические навыки')
        if any(term in text_lower for term in ['управлен', 'администрирован', 'координ']):
            skills.append('управленческие навыки')
            
        return list(set(skills))[:8]  # Ограничиваем 8 навыками
    
    def _classify_role(self, role_name: str, functions: List[str]) -> str:
        """Классификация типа роли"""
        role_lower = role_name.lower()
        functions_text = ' '.join(functions).lower()
        
        if any(word in role_lower for word in ['админ', 'администратор', 'руковод', 'управлен']):
            return 'AdminRole'
        elif any(word in role_lower for word in ['эксперт', 'рецензент', 'проверяющ', 'аналитик']):
            return 'ExpertRole'
        elif any(word in functions_text for word in ['созда', 'добав', 'регистр', 'заполн']):
            return 'UserRole'
        elif any(word in functions_text for word in ['просмотр', 'чтение', 'ознакомлен', 'поиск']):
            return 'UserRole'
        elif any(word in functions_text for word in ['оценк', 'реценз', 'проверк']):
            return 'ReviewRole'
        else:
            return 'Role'
        
    def build_from_json(self, json_data: List[Dict]):
        """Построение графа знаний из JSON данных"""
        print("Построение графа знаний...")
        
        for service_idx, service in enumerate(json_data):
            service_uri = self.SERVICE[f"service_{service_idx}"]
            service_name = service['service_name']
            
            # Добавляем сервис
            self.graph.add((service_uri, RDF.type, self.ONTOLOGY.Service))
            self.graph.add((service_uri, RDFS.label, Literal(service_name, lang='ru')))
            
            # Добавляем роли
            for role_idx, role in enumerate(service.get('roles', [])):
                role_uri = self.ROLE[f"{service_idx}_{role_idx}"]
                role_name = role['role_name']
                
                # Добавляем роль
                self.graph.add((role_uri, RDF.type, self.ONTOLOGY.Role))
                self.graph.add((role_uri, RDFS.label, Literal(role_name, lang='ru')))
                
                # Связь с сервисом
                self.graph.add((role_uri, self.ONTOLOGY.partOfService, service_uri))
                
                # Классификация типа роли
                role_type = self._classify_role(role_name, role['functions'])
                self.graph.add((role_uri, RDF.type, getattr(self.ONTOLOGY, role_type)))
                
                # Сохраняем информацию для быстрого доступа
                role_id = f"{service_idx}_{role_idx}"
                self.role_to_service[role_id] = {
                    'service_name': service_name,
                    'role_name': role_name,
                    'service_idx': service_idx,
                    'role_idx': role_idx
                }
                self.role_functions[role_id] = role['functions']
                
                # Добавляем функции и навыки
                for func_idx, function_text in enumerate(role['functions']):
                    func_uri = self.FUNCTION[f"{role_id}_{func_idx}"]
                    
                    # Добавляем функцию
                    self.graph.add((func_uri, RDF.type, self.ONTOLOGY.Function))
                    self.graph.add((func_uri, RDFS.label, Literal(function_text, lang='ru')))
                    
                    # Связь роли с функцией
                    self.graph.add((role_uri, self.ONTOLOGY.hasFunction, func_uri))
                    
                    # Извлекаем и добавляем навыки
                    skills = self._extract_skills(function_text)
                    for skill_idx, skill_name in enumerate(skills):
                        skill_uri = self.SKILL[f"{role_id}_{func_idx}_{skill_idx}"]
                        self.graph.add((skill_uri, RDF.type, self.ONTOLOGY.Skill))
                        self.graph.add((skill_uri, RDFS.label, Literal(skill_name, lang='ru')))
                        self.graph.add((func_uri, self.ONTOLOGY.requiresSkill, skill_uri))
        
        # Построение семантического индекса
        self._build_semantic_index()
        print(f"Граф знаний построен: {len(list(self.graph.subjects()))} вершин")
    
    def _build_semantic_index(self):
        """Построение семантического индекса"""
        print("Построение семантического индекса...")
        
        for role_uri in self.graph.subjects(RDF.type, self.ONTOLOGY.Role):
            role_id = str(role_uri).split('/')[-1]
            
            # Получаем метку роли
            for label in self.graph.objects(role_uri, RDFS.label):
                role_name = str(label).lower()
                
                # Индексируем по словам
                words = re.findall(r'\b\w+\b', role_name)
                for word in words:
                    if len(word) > 2:
                        if word not in self.semantic_index:
                            self.semantic_index[word] = set()
                        self.semantic_index[word].add(role_id)
            
            # Индексируем по функциям
            if role_id in self.role_functions:
                for function in self.role_functions[role_id]:
                    func_words = re.findall(r'\b\w+\b', function.lower())
                    for word in func_words:
                        if len(word) > 3:
                            if word not in self.semantic_index:
                                self.semantic_index[word] = set()
                            self.semantic_index[word].add(role_id)
    
    def semantic_search(self, query: str) -> List[str]:
        """Семантический поиск по графу"""
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        matching_roles = set()
        
        # Поиск по индексу
        for word in query_words:
            if word in self.semantic_index:
                matching_roles.update(self.semantic_index[word])
        
        # Расширение через связи в графе
        expanded_roles = set()
        for role_id in matching_roles:
            expanded_roles.add(role_id)
            
            # Ищем связанные роли через общие навыки
            role_uri = self.ROLE[role_id]
            
            # Получаем навыки текущей роли
            current_skills = set()
            for func in self.graph.objects(role_uri, self.ONTOLOGY.hasFunction):
                for skill in self.graph.objects(func, self.ONTOLOGY.requiresSkill):
                    for skill_label in self.graph.objects(skill, RDFS.label):
                        current_skills.add(str(skill_label))
            
            # Ищем другие роли с такими же навыки
            for other_role_uri in self.graph.subjects(RDF.type, self.ONTOLOGY.Role):
                other_role_id = str(other_role_uri).split('/')[-1]
                if other_role_id == role_id:
                    continue
                
                other_skills = set()
                for func in self.graph.objects(other_role_uri, self.ONTOLOGY.hasFunction):
                    for skill in self.graph.objects(func, self.ONTOLOGY.requiresSkill):
                        for skill_label in self.graph.objects(skill, RDFS.label):
                            other_skills.add(str(skill_label))
                
                # Если есть общие навыки
                if current_skills.intersection(other_skills):
                    expanded_roles.add(other_role_id)
        
        return list(expanded_roles)
    
    def get_role_details(self, role_id: str) -> Dict:
        """Получение детальной информации о роли"""
        if role_id not in self.role_to_service:
            return {}
        
        role_uri = self.ROLE[role_id]
        service_info = self.role_to_service[role_id]
        
        details = {
            'role_id': role_id,
            'role_name': service_info['role_name'],
            'service_name': service_info['service_name'],
            'service_idx': service_info['service_idx'],
            'role_idx': service_info['role_idx'],
            'functions': self.role_functions.get(role_id, []),
            'skills': set(),
            'role_type': 'Role'
        }
        
        # Получаем навыки
        for func in self.graph.objects(role_uri, self.ONTOLOGY.hasFunction):
            for skill in self.graph.objects(func, self.ONTOLOGY.requiresSkill):
                for skill_label in self.graph.objects(skill, RDFS.label):
                    details['skills'].add(str(skill_label))
        
        details['skills'] = list(details['skills'])
        
        # Определяем тип роли
        if (role_uri, RDF.type, self.ONTOLOGY.AdminRole) in self.graph:
            details['role_type'] = 'AdminRole'
        elif (role_uri, RDF.type, self.ONTOLOGY.ExpertRole) in self.graph:
            details['role_type'] = 'ExpertRole'
        elif (role_uri, RDF.type, self.ONTOLOGY.UserRole) in self.graph:
            details['role_type'] = 'UserRole'
        elif (role_uri, RDF.type, self.ONTOLOGY.ReviewRole) in self.graph:
            details['role_type'] = 'ReviewRole'
        
        return details
    
    def find_similar_roles(self, role_id: str, max_results: int = 5) -> List[Dict]:
        """Поиск похожих ролей на основе графа"""
        current_details = self.get_role_details(role_id)
        if not current_details:
            return []
        
        current_skills = set(current_details['skills'])
        similar_roles = []
        
        for other_role_id in self.role_to_service:
            if other_role_id == role_id:
                continue
            
            other_details = self.get_role_details(other_role_id)
            other_skills = set(other_details['skills'])
            
            # Вычисляем схожесть по навыкам
            common_skills = current_skills.intersection(other_skills)
            if common_skills:
                similarity = len(common_skills) / max(len(current_skills), 1)
                
                if similarity > 0.2:  # Порог схожести
                    role_info = other_details.copy()
                    role_info['similarity_score'] = similarity
                    role_info['common_skills'] = list(common_skills)
                    similar_roles.append(role_info)
        
        return sorted(similar_roles, key=lambda x: x['similarity_score'], reverse=True)[:max_results]


class UnifiedRoleSearcher:
    """Унифицированная система поиска ролей"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Инициализация улучшенного поисковика"""
        print("Инициализация унифицированной системы поиска ролей...")
        
        # Инициализация моделей
        try:
            self.model = SentenceTransformer(model_name)
            print(f" Основная модель {model_name} загружена")
        except Exception as e:
            print(f" Ошибка загрузки модели: {e}")
            self.model = None
        
        # Граф знаний
        self.knowledge_graph = KnowledgeGraphManager()
        
        # Данные и индексы
        self.services_data = []
        self.role_index = []
        self.role_embeddings = None
        self.bm25_index = None
        self.tokenized_role_corpus = []
        self.role_embeddings_cache = {}  # Кэш эмбеддингов
        
        # Параметры
        self.min_threshold = 0.3
        self.semantic_weight = 0.6
        self.embedding_weight = 0.4
        self.bm25_weight = 0.3
        
        # Словари
        self.load_synonyms()
        
        # Устройство для вычислений
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def load_synonyms(self):
        """Загрузка словарей синонимов и аббревиатур"""
        self.synonyms = {
            'создать': ['создать', 'добавить', 'сделать', 'сформировать', 'разработать', 'оформить'],
            'посмотреть': ['посмотреть', 'просмотреть', 'увидеть', 'ознакомиться', 'проверить', 'изучить'],
            'найти': ['найти', 'поиск', 'обнаружить', 'подобрать', 'выбрать', 'отобрать'],
            'управление': ['управление', 'администрирование', 'руководство', 'контроль', 'координация'],
            'отчет': ['отчет', 'отчетность', 'документ', 'справка', 'акт', 'ведение'],
            'проект': ['проект', 'программа', 'план', 'задание', 'работа', 'мероприятие'],
            'исследование': ['исследование', 'изучение', 'анализ', 'разработка', 'эксперимент'],
            'финансы': ['финансы', 'бюджет', 'деньги', 'экономика', 'средства', 'финансирование'],
            'роль': ['роль', 'должность', 'позиция', 'функция', 'обязанность', 'задача'],
            'сервис': ['сервис', 'услуга', 'система', 'платформа', 'модуль', 'инструмент'],
            'эксперт': ['эксперт', 'рецензент', 'оценщик', 'аналитик', 'специалист'],
            'администратор': ['администратор', 'админ', 'управляющий', 'менеджер', 'координатор'],
        }
        
        self.abbreviations = {
            'НИОКТР': 'научно-исследовательские опытно-конструкторские технологические работы',
            'РИД': 'результаты интеллектуальной деятельности',
            'ПТНИ': 'проекты тематик научных исследований',
            'ГЖС': 'государственный жилищный сертификат',
            'ЦКП': 'центр коллективного пользования',
            'УНУ': 'уникальная научная установка',
            'ФЭО': 'финансово-экономическое обоснование',
            'ОБАС': 'объем бюджетных ассигнований',
            'НОО': 'научная образовательная организация',
            'ГРБС': 'главный распорядитель бюджетных средств',
            'НТС': 'научно-технический совет',
            'РАН': 'российская академия наук',
            'КПЭ': 'ключевые показатели эффективности',
            'НИР': 'научно-исследовательские работы',
            'ОКР': 'опытно-конструкторские работы',
        }
    
    def enhance_query_preprocessing(self, query: str) -> str:
        """Улучшенная предобработка запроса"""
        if not query:
            return ""
        
        # Базовые преобразования
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        
        # Расширение аббревиатур
        for abbr, full in self.abbreviations.items():
            pattern = re.compile(rf'\b{re.escape(abbr.lower())}\b', re.IGNORECASE)
            query = pattern.sub(full, query)
        
        # Добавление синонимов
        expanded_terms = []
        for word in query.split():
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word][:2])
            else:
                expanded_terms.append(word)
        
        query = ' '.join(expanded_terms)
        
        # Удаление стоп-слов
        stop_words = {'и', 'в', 'с', 'по', 'для', 'на', 'из', 'от', 'к', 'до', 
                     'о', 'у', 'не', 'но', 'за', 'же', 'бы', 'во', 'со', 'об',
                     'а', 'или', 'же', 'ли', 'то', 'это', 'как', 'так', 'что'}
        
        query_words = [w for w in query.split() if w not in stop_words]
        
        return ' '.join(query_words)
    
    def prepare_role_texts(self, service: Dict, role: Dict) -> Dict:
        """Подготовка текстов для индексирования роли"""
        # Создаем несколько вариантов описания
        role_texts = []
        
        # 1. Полное описание
        full_description = f"""
        Роль: {role['role_name']}
        Сервис: {service['service_name']}
        Описание сервиса: {service.get('service_description', '')}
        Функции: {'; '.join(role['functions'][:10])}
        """
        role_texts.append(full_description)
        
        # 2. Описание функций
        functions_text = f"Функции роли {role['role_name']}: {' '.join(role['functions'])}"
        role_texts.append(functions_text)
        
        # 3. Краткое описание для BM25
        short_text = f"{role['role_name']} {' '.join(role['functions'][:3])}"
        role_texts.append(short_text)
        
        # 4. Расширенное описание для семантического поиска
        semantic_text = f"""
        {role['role_name']} в сервисе {service['service_name']}.
        Основные задачи: {' '.join(role['functions'][:5])}
        """
        role_texts.append(semantic_text)
        
        return {
            'service': service,
            'role': role,
            'service_name': service['service_name'],
            'role_name': role['role_name'],
            'functions': role['functions'],
            'texts': role_texts,
            'primary_text': role_texts[0],
            'bm25_text': role_texts[2],
            'semantic_text': role_texts[3],
        }
    
    def load_services_from_json(self, json_data: List[Dict]):
        """Загрузка и индексирование данных о сервисах"""
        try:
            print(f"Загрузка {len(json_data)} сервисов...")
            self.services_data = json_data
            
            # Создаем индекс ролей
            print("Создание индекса ролей...")
            self.role_index = []
            
            for service_idx, service in enumerate(json_data):
                for role_idx, role in enumerate(service.get('roles', [])):
                    indexed_role = self.prepare_role_texts(service, role)
                    indexed_role['service_idx'] = service_idx
                    indexed_role['role_idx'] = role_idx
                    self.role_index.append(indexed_role)
            
            print(f" Индексировано {len(self.role_index)} ролей")
            
            # Создаем BM25 индекс
            print("Создание BM25 индекса...")
            bm25_texts = [role['bm25_text'] for role in self.role_index]
            self.tokenized_role_corpus = [word_tokenize(text.lower()) for text in bm25_texts]
            self.bm25_index = BM25Okapi(self.tokenized_role_corpus)
            print(" BM25 индекс создан")
            
            # Создаем эмбеддинги для ролей
            if self.model:
                print("Создание эмбеддингов...")
                primary_texts = [role['primary_text'] for role in self.role_index]
                self.role_embeddings = self.model.encode(
                    primary_texts,
                    convert_to_tensor=False,
                    show_progress_bar=True,
                    batch_size=16,
                    normalize_embeddings=True
                )
                print(" Эмбеддинги созданы")
                
                # Кэшируем эмбеддинги
                for idx, embedding in enumerate(self.role_embeddings):
                    role_id = f"{self.role_index[idx]['service_idx']}_{self.role_index[idx]['role_idx']}"
                    self.role_embeddings_cache[role_id] = embedding
            
            # Строим граф знаний
            print("Построение графа знаний...")
            self.knowledge_graph.build_from_json(json_data)
            
            print("Индексирование завершено успешно!")
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            raise
    
    def load_services_from_file(self, file_path: str):
        """Загрузка данных из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            self.load_services_from_json(json_data)
        except Exception as e:
            print(f"Ошибка загрузки файла {file_path}: {e}")
            raise
    
    def search_with_bm25(self, query: str) -> List[Tuple[str, float]]:
        """Поиск с использованием BM25"""
        if not self.bm25_index:
            return []
        
        processed_query = self.enhance_query_preprocessing(query)
        tokenized_query = word_tokenize(processed_query.lower())
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Нормализация
        if len(scores) > 0:
            max_score = max(scores)
            if max_score > 0:
                scores = scores / max_score
        
        results = []
        for idx, score in enumerate(scores):
            if score > self.min_threshold:
                role_id = f"{self.role_index[idx]['service_idx']}_{self.role_index[idx]['role_idx']}"
                results.append((role_id, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:15]
    
    def search_with_embeddings(self, query: str) -> List[Tuple[str, float]]:
        """Поиск с использованием эмбеддингов"""
        if not self.model or self.role_embeddings is None:
            return []
        
        processed_query = self.enhance_query_preprocessing(query)
        query_embedding = self.model.encode(
            processed_query,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.role_embeddings
        )[0]
        
        results = []
        for idx, score in enumerate(similarities):
            if score > self.min_threshold:
                role_id = f"{self.role_index[idx]['service_idx']}_{self.role_index[idx]['role_idx']}"
                results.append((role_id, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:15]
    
    def search_with_semantic_graph(self, query: str) -> List[Tuple[str, float]]:
        """Поиск с использованием графа знаний"""
        processed_query = self.enhance_query_preprocessing(query)
        
        # Получаем роли из графа
        role_ids = self.knowledge_graph.semantic_search(processed_query)
        
        # Оцениваем релевантность
        query_words = set(processed_query.split())
        results = []
        
        for role_id in role_ids:
            role_details = self.knowledge_graph.get_role_details(role_id)
            if not role_details:
                continue
            
            # Вычисляем релевантность по совпадению слов
            role_text = f"{role_details['role_name']} {' '.join(role_details['functions'])}".lower()
            role_words = set(re.findall(r'\b\w+\b', role_text))
            
            common_words = query_words.intersection(role_words)
            if common_words:
                score = len(common_words) / max(len(query_words), 1)
                results.append((role_id, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:15]
    
    def ensemble_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Ансамблевый поиск с использованием всех методов"""
        if not self.role_index:
            return []
        
        # Получаем результаты от всех методов
        bm25_results = self.search_with_bm25(query)
        embedding_results = self.search_with_embeddings(query)
        semantic_results = self.search_with_semantic_graph(query)
        
        # Объединяем и взвешиваем результаты
        combined_scores = defaultdict(float)
        
        # BM25 (вес 0.3)
        for role_id, score in bm25_results:
            combined_scores[role_id] += score * self.bm25_weight
        
        # Эмбеддинги (вес 0.4)
        for role_id, score in embedding_results:
            combined_scores[role_id] += score * self.embedding_weight
        
        # Семантический поиск (вес 0.6)
        for role_id, score in semantic_results:
            combined_scores[role_id] += score * self.semantic_weight
        
        # Нормализуем итоговые оценки
        if combined_scores:
            max_score = max(combined_scores.values())
            for role_id in combined_scores:
                combined_scores[role_id] = combined_scores[role_id] / max_score
        
        # Сортируем и выбираем топ-K
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Формируем финальные результаты
        results = []
        for role_id, score in sorted_results[:top_k * 2]:  # Берем в 2 раза больше для фильтрации
            if score > self.min_threshold:
                # Получаем информацию о роли
                role_details = self.knowledge_graph.get_role_details(role_id)
                if not role_details:
                    continue
                
                # Находим оригинальные данные
                service_idx = role_details['service_idx']
                role_idx = role_details['role_idx']
                
                if service_idx < len(self.services_data) and role_idx < len(self.services_data[service_idx].get('roles', [])):
                    service = self.services_data[service_idx]
                    role = service['roles'][role_idx]
                    
                    result = {
                        'service': service,
                        'role': role,
                        'service_name': service['service_name'],
                        'role_name': role['role_name'],
                        'functions': role['functions'],
                        'search_score': float(score),
                        'confidence': self._get_confidence_level(score),
                        'role_type': role_details['role_type'],
                        'skills': role_details['skills'],
                        'explanation': self._generate_explanation(query, role_details)
                    }
                    
                    # Добавляем методы, которые нашли эту роль
                    methods = []
                    if role_id in dict(bm25_results):
                        methods.append('BM25')
                    if role_id in dict(embedding_results):
                        methods.append('Embeddings')
                    if role_id in dict(semantic_results):
                        methods.append('Semantic')
                    result['found_by'] = methods
                    
                    results.append(result)
        
        # Дополняем если результатов мало
        if len(results) < top_k and len(self.role_index) > 0:
            self._add_supplementary_results(results, top_k, query)
        
        return results[:top_k]
    
    def _get_confidence_level(self, score: float) -> str:
        """Определение уровня уверенности"""
        if score > 0.8:
            return 'Высокая'
        elif score > 0.6:
            return 'Средняя'
        elif score > 0.4:
            return 'Низкая'
        else:
            return 'Минимальная'
    
    def _generate_explanation(self, query: str, role_details: Dict) -> str:
        """Генерация объяснения почему роль подходит"""
        query_words = set(self.enhance_query_preprocessing(query).split())
        role_text = f"{role_details['role_name']} {' '.join(role_details['functions'])}".lower()
        role_words = set(re.findall(r'\b\w+\b', role_text))
        
        common_words = query_words.intersection(role_words)
        
        if common_words:
            explanation = f"Совпадение по терминам: {', '.join(list(common_words)[:5])}"
        else:
            explanation = "Семантическая схожесть на основе контекста"
        
        # Добавляем информацию о типе роли
        type_descriptions = {
            'AdminRole': 'административная роль',
            'UserRole': 'пользовательская роль',
            'ExpertRole': 'экспертная роль',
            'ReviewRole': 'роль проверки/рецензирования',
            'Role': 'базовая роль'
        }
        explanation += f". Это {type_descriptions.get(role_details['role_type'], 'роль')}"
        
        if role_details['skills']:
            explanation += f". Требуемые навыки: {', '.join(role_details['skills'][:3])}"
        
        return explanation
    
    def _add_supplementary_results(self, results: List[Dict], target_count: int, query: str):
        """Добавление дополнительных результатов при необходимости"""
        added_indices = set()
        for result in results:
            if 'service_idx' in result.get('role', {}):
                role_id = f"{result['role'].get('service_idx', 0)}_{result['role'].get('role_idx', 0)}"
                added_indices.add(role_id)
        
        available_roles = [idx for idx in range(len(self.role_index)) 
                          if f"{self.role_index[idx]['service_idx']}_{self.role_index[idx]['role_idx']}" not in added_indices]
        
        while len(results) < target_count and available_roles:
            idx = random.choice(available_roles)
            role_info = self.role_index[idx]
            
            role_id = f"{role_info['service_idx']}_{role_info['role_idx']}"
            role_details = self.knowledge_graph.get_role_details(role_id)
            
            result = {
                'service': role_info['service'],
                'role': role_info['role'],
                'service_name': role_info['service_name'],
                'role_name': role_info['role_name'],
                'functions': role_info['functions'],
                'search_score': 0.1,
                'confidence': 'МИНИМАЛЬНАЯ',
                'role_type': role_details.get('role_type', 'Role') if role_details else 'Role',
                'skills': role_details.get('skills', []) if role_details else [],
                'explanation': 'Дополнительная рекомендация на основе доступных данных',
                'found_by': ['Supplementary'],
                'note': 'Может быть полезно'
            }
            
            results.append(result)
            available_roles.remove(idx)
    
    def find_similar_to_role(self, role_name: str, max_results: int = 3) -> List[Dict]:
        """Поиск ролей, похожих на указанную"""
        # Находим ID роли
        target_role_id = None
        for role_id, details in self.knowledge_graph.role_to_service.items():
            if details['role_name'].lower() == role_name.lower():
                target_role_id = role_id
                break
        
        if not target_role_id:
            # Пробуем найти частичное совпадение
            for role_id, details in self.knowledge_graph.role_to_service.items():
                if role_name.lower() in details['role_name'].lower():
                    target_role_id = role_id
                    break
        
        if not target_role_id:
            return []
        
        # Ищем похожие роли в графе знаний
        similar_roles = self.knowledge_graph.find_similar_roles(target_role_id, max_results)
        
        # Форматируем результаты
        formatted_results = []
        for role_info in similar_roles:
            service_idx = role_info['service_idx']
            role_idx = role_info['role_idx']
            
            if service_idx < len(self.services_data) and role_idx < len(self.services_data[service_idx].get('roles', [])):
                service = self.services_data[service_idx]
                role = service['roles'][role_idx]
                
                formatted_result = {
                    'service_name': service['service_name'],
                    'role_name': role['role_name'],
                    'similarity_score': role_info['similarity_score'],
                    'common_skills': role_info.get('common_skills', []),
                    'functions': role['functions'][:3],
                    'role_type': role_info.get('role_type', 'Role')
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def interactive_search(self):
        """Интерактивный режим поиска"""
        if not self.role_index:
            print("Сначала загрузите данные сервисов!")
            return
    
        print(f"\n{'='*80}")
        print("УНИФИЦИРОВАННЫЙ ПОИСК РОЛЕЙ В СЕРВИСАХ ЕГИСУ НИОКТР")
        print("Используются: BM25 + Sentence-BERT + Граф знаний")
        print("Для выхода введите: exit, выход")
        print(f"{'='*80}\n")
        
        while True:
            print()
            query = input("Опишите, какую роль вы ищете: ").strip()
            
            if not query:
                print("Запрос не может быть пустым.")
                continue
            
            if query.lower() in ['exit', 'выход', 'quit']:
                print("Завершение работы...")
                break
            
            print(f"\nПоиск ролей для запроса: '{query}'")
            print("=" * 80)
            
            start_time = time.time()
            results = self.ensemble_search(query, top_k=5)
            search_time = time.time() - start_time
            
            if not results:
                print("Не найдено подходящих ролей. Попробуйте переформулировать запрос.")
                
                # Спрашиваем, хочет ли пользователь продолжить
                print("\nХотите попробовать другой запрос? (да/нет)")
                response = input("> ").lower().strip()
                if response in ['да', 'yes', 'y', 'д']:
                    # Очищаем вывод для нового поиска
                    print("\n" + "="*80)
                    print("Новый поиск")
                    print("="*80)
                    continue
                else:
                    print("\nЗавершение поиска...")
                    break
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Сервис: {result['service_name']}")
                print(f"   Роль: {result['role_name']} ({result['role_type']})")
                print(f"   Уверенность: {result['confidence']} ({result['search_score']:.3f})")
                print(f"   Методы поиска: {', '.join(result['found_by'])}")
                print(f"   Объяснение: {result['explanation']}")
                
                if result.get('skills'):
                    print(f"   Ключевые навыки: {', '.join(result['skills'][:3])}")
                
                print(f"   Основные функции:")
                for func in result['functions'][:3]:
                    func_display = func[:80] + "..." if len(func) > 80 else func
                    print(f"      {func_display}")
                
                if len(result['functions']) > 3:
                    print(f"     ... и еще {len(result['functions']) - 3} функций")
            
            print(f"\n{'='*80}")
            print(f"Найдено {len(results)} подходящих ролей")
            print(f"Время поиска: {search_time:.2f} секунд")
            
            # Предлагаем поиск еще одной роли
            print("\nХотите найти еще одну роль? (да/нет)")
            response = input("> ").lower().strip()
            
            if response in ['да', 'yes', 'y', 'д']:
                # Очищаем вывод для нового поиска
                print("\n" + "="*80)
                print("Новый поиск")
                print("="*80)
            else:
                print("\nЗавершение поиска...")
                break


# Для использования в accuracy.ipynb
class AdvancedRoleServiceSearcher(UnifiedRoleSearcher):
    """Алиас для совместимости с тестированием"""
    pass


# Вспомогательные функции
def load_json_data(file_path: str) -> List[Dict]:
    """Загрузка данных из JSON файла"""
    try:
        print(f"Загрузка данных из файла: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        print(f"Загружено {len(json_data)} сервисов")
        return json_data
        
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        raise

def main():
    """Основная функция запуска системы"""
    print("="*80)
    print("СИСТЕМА ПОИСКА РОЛЕЙ В ЕГИСУ НИОКТР")
    print("="*80)
    
    # Инициализация поисковика
    searcher = UnifiedRoleSearcher()
    
    # Определение путей к файлам
    base_dir = "C:\\CITIS\\Work\\RDF_struct"
    services_path = os.path.join(base_dir, "services.json")
    
    # Загрузка данных
    try:
        if os.path.exists(services_path):
            print(f"Загрузка данных из: {services_path}")
            searcher.load_services_from_file(services_path)
        elif os.path.exists("services.json"):
            print("Загрузка данных из текущей директории...")
            searcher.load_services_from_file("services.json")
        else:
            print("Файл services.json не найден.")
            return
    except Exception as e:
        print(f"Не удалось загрузить данные: {e}")
        return
    
    # Запуск интерактивного поиска
    print("\nЗапуск интерактивного режима поиска...")
    searcher.interactive_search()


if __name__ == "__main__":
    main()