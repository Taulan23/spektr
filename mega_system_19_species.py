#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МЕГА-СИСТЕМА КЛАССИФИКАЦИИ 19 ВИДОВ ДЕРЕВЬЕВ
Использует все лучшие техники и подходы
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Attention, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Concatenate
import joblib
import warnings
from datetime import datetime
from scipy import signal
from scipy.stats import skew, kurtosis
import gc
import time

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_spring_data_19_species():
    """Загружает весенние данные для 19 видов деревьев"""
    
    print("🌱 ЗАГРУЗКА ВЕСЕННИХ ДАННЫХ (19 ВИДОВ)...")
    
    spring_folder = "Спектры, весенний период, 20 видов"
    
    # Все виды кроме пустого клен_ам
    species_folders = [
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 
        'лиственница', 'липа', 'орех', 'осина', 'рябина', 'сирень',
        'сосна', 'тополь_бальзамический', 'тополь_черный', 'туя', 
        'черемуха', 'ясень'
    ]
    
    spring_data = []
    spring_labels = []
    
    for species in species_folders:
        folder_path = os.path.join(spring_folder, species)
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, "*.xlsx"))
            print(f"   {species}: {len(files)} файлов")
            
            for file in files:
                try:
                    df = pd.read_excel(file)
                    if not df.empty and len(df.columns) >= 2:
                        # Берем вторую колонку как спектральные данные
                        spectrum = df.iloc[:, 1].values
                        if len(spectrum) > 100:  # Минимальная длина спектра
                            spring_data.append(spectrum)
                            spring_labels.append(species)
                except Exception as e:
                    print(f"Ошибка при загрузке {file}: {e}")
    
    print(f"✅ Загружено {len(spring_data)} образцов по {len(set(spring_labels))} видам")
    return spring_data, spring_labels

def load_summer_data_original():
    """Загружает летние данные (оригинальные 7 видов)"""
    
    print("☀️ ЗАГРУЗКА ЛЕТНИХ ДАННЫХ (7 ВИДОВ)...")
    
    species_folders = {
        'береза': 'береза',
        'дуб': 'дуб', 
        'ель': 'ель',
        'клен': 'клен',
        'липа': 'липа',
        'осина': 'осина',
        'сосна': 'сосна'
    }
    
    summer_data = []
    summer_labels = []
    
    for species, folder in species_folders.items():
        files = glob.glob(os.path.join(folder, "*.xlsx"))
        print(f"   {species}: {len(files)} файлов")
        
        for file in files:
            try:
                df = pd.read_excel(file)
                if not df.empty and len(df.columns) >= 2:
                    spectrum = df.iloc[:, 1].values
                    if len(spectrum) > 100:
                        summer_data.append(spectrum)
                        summer_labels.append(species)
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    
    print(f"✅ Загружено {len(summer_data)} образцов по {len(set(summer_labels))} видам")
    return summer_data, summer_labels

def preprocess_spectra(spectra_list):
    """Предобработка спектров - приведение к одинаковой длине"""
    
    # Находим минимальную длину
    min_length = min(len(spectrum) for spectrum in spectra_list)
    print(f"   Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до минимальной длины
    processed_spectra = []
    for spectrum in spectra_list:
        # Удаляем NaN значения
        spectrum_clean = spectrum[~np.isnan(spectrum)]
        if len(spectrum_clean) >= min_length:
            processed_spectra.append(spectrum_clean[:min_length])
    
    return np.array(processed_spectra)

def extract_mega_features(spectra, labels=None):
    """Извлекает максимальное количество признаков для 19 видов"""
    
    print("🧠 ИЗВЛЕЧЕНИЕ МЕГА-ПРИЗНАКОВ...")
    n_samples, n_channels = spectra.shape
    all_features = []
    
    # Получаем уникальные виды
    if labels is not None:
        unique_species = sorted(list(set(labels)))
        print(f"   Виды: {unique_species}")
    
    for i, spectrum in enumerate(spectra):
        features = []
        
        # 1. БАЗОВЫЕ СТАТИСТИЧЕСКИЕ ПРИЗНАКИ (20 признаков)
        features.extend([
            np.mean(spectrum), np.std(spectrum), np.median(spectrum),
            np.min(spectrum), np.max(spectrum), np.ptp(spectrum),
            np.var(spectrum), skew(spectrum), kurtosis(spectrum),
            np.sqrt(np.mean(spectrum**2)),  # RMS
            np.mean(np.abs(spectrum - np.median(spectrum))),  # MAD
            np.percentile(spectrum, 10), np.percentile(spectrum, 25),
            np.percentile(spectrum, 75), np.percentile(spectrum, 90),
            np.percentile(spectrum, 95), np.percentile(spectrum, 99),
            np.sum(spectrum > np.mean(spectrum)) / len(spectrum),  # % выше среднего
            np.sum(spectrum > 0) / len(spectrum),  # % положительных
            len(np.where(np.diff(spectrum) > 0)[0]) / len(spectrum)  # % возрастающих
        ])
        
        # 2. КВАНТИЛЬНЫЕ ПРИЗНАКИ (15 признаков)
        quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
        for q in quantiles:
            features.append(np.percentile(spectrum, q*100))
        features.append(np.percentile(spectrum, 75) - np.percentile(spectrum, 25))  # IQR
        
        # 3. ПРОИЗВОДНЫЕ (30 признаков)
        if len(spectrum) > 3:
            # Первая производная
            diff1 = np.diff(spectrum)
            features.extend([
                np.mean(diff1), np.std(diff1), np.median(diff1),
                np.min(diff1), np.max(diff1), np.ptp(diff1),
                skew(diff1), kurtosis(diff1), np.mean(np.abs(diff1)),
                np.sqrt(np.mean(diff1**2))
            ])
            
            # Вторая производная
            if len(diff1) > 1:
                diff2 = np.diff(diff1)
                features.extend([
                    np.mean(diff2), np.std(diff2), np.median(diff2),
                    np.min(diff2), np.max(diff2), np.ptp(diff2),
                    skew(diff2), kurtosis(diff2), np.mean(np.abs(diff2)),
                    np.sqrt(np.mean(diff2**2))
                ])
                
                # Третья производная
                if len(diff2) > 1:
                    diff3 = np.diff(diff2)
                    features.extend([
                        np.mean(diff3), np.std(diff3), np.median(diff3),
                        np.min(diff3), np.max(diff3), np.ptp(diff3),
                        skew(diff3), kurtosis(diff3), np.mean(np.abs(diff3)),
                        np.sqrt(np.mean(diff3**2))
                    ])
        
        # 4. СПЕКТРАЛЬНЫЕ РЕГИОНЫ (60 признаков)
        # Разбиваем спектр на 20 частей и считаем статистики
        n_regions = 20
        region_size = len(spectrum) // n_regions
        
        for j in range(n_regions):
            start_idx = j * region_size
            end_idx = min((j + 1) * region_size, len(spectrum))
            region = spectrum[start_idx:end_idx]
            
            if len(region) > 0:
                features.extend([
                    np.mean(region),
                    np.std(region),
                    np.max(region) - np.min(region)
                ])
        
        # 5. СПЕКТРАЛЬНЫЕ МОМЕНТЫ И ЭНЕРГИИ (25 признаков)
        # Центроид спектра
        indices = np.arange(len(spectrum))
        if np.sum(spectrum) != 0:
            centroid = np.sum(indices * spectrum) / np.sum(spectrum)
        else:
            centroid = len(spectrum) / 2
        features.append(centroid)
        
        # Спектральный разброс
        spread = np.sqrt(np.sum(((indices - centroid) ** 2) * spectrum) / np.sum(spectrum)) if np.sum(spectrum) != 0 else 0
        features.append(spread)
        
        # Спектральная асимметрия
        if spread != 0:
            spec_skew = np.sum(((indices - centroid) ** 3) * spectrum) / (np.sum(spectrum) * (spread ** 3))
        else:
            spec_skew = 0
        features.append(spec_skew)
        
        # Спектральный эксцесс
        if spread != 0:
            spec_kurt = np.sum(((indices - centroid) ** 4) * spectrum) / (np.sum(spectrum) * (spread ** 4))
        else:
            spec_kurt = 0
        features.append(spec_kurt)
        
        # Энергия в разных частотных полосах (20 полос)
        band_size = len(spectrum) // 20
        for k in range(20):
            start_idx = k * band_size
            end_idx = min((k + 1) * band_size, len(spectrum))
            band_energy = np.sum(spectrum[start_idx:end_idx] ** 2)
            features.append(band_energy)
        
        # 6. ОТНОШЕНИЯ МЕЖДУ РЕГИОНАМИ (45 признаков)
        # Отношения между различными частями спектра
        quarter_size = len(spectrum) // 4
        q1 = spectrum[:quarter_size]
        q2 = spectrum[quarter_size:2*quarter_size]
        q3 = spectrum[2*quarter_size:3*quarter_size]
        q4 = spectrum[3*quarter_size:]
        
        # Отношения средних значений
        features.extend([
            np.mean(q1) / (np.mean(q2) + 1e-8),
            np.mean(q1) / (np.mean(q3) + 1e-8),
            np.mean(q1) / (np.mean(q4) + 1e-8),
            np.mean(q2) / (np.mean(q3) + 1e-8),
            np.mean(q2) / (np.mean(q4) + 1e-8),
            np.mean(q3) / (np.mean(q4) + 1e-8)
        ])
        
        # Отношения энергий
        features.extend([
            np.sum(q1**2) / (np.sum(q2**2) + 1e-8),
            np.sum(q1**2) / (np.sum(q3**2) + 1e-8),
            np.sum(q1**2) / (np.sum(q4**2) + 1e-8),
            np.sum(q2**2) / (np.sum(q3**2) + 1e-8),
            np.sum(q2**2) / (np.sum(q4**2) + 1e-8),
            np.sum(q3**2) / (np.sum(q4**2) + 1e-8)
        ])
        
        # Дополнительные отношения
        half_size = len(spectrum) // 2
        first_half = spectrum[:half_size]
        second_half = spectrum[half_size:]
        
        features.extend([
            np.mean(first_half) / (np.mean(second_half) + 1e-8),
            np.std(first_half) / (np.std(second_half) + 1e-8),
            np.max(first_half) / (np.max(second_half) + 1e-8),
            np.min(first_half) / (np.min(second_half) + 1e-8),
            np.sum(first_half**2) / (np.sum(second_half**2) + 1e-8)
        ])
        
        # Соотношения начала, середины и конца
        third_size = len(spectrum) // 3
        begin = spectrum[:third_size]
        middle = spectrum[third_size:2*third_size]
        end = spectrum[2*third_size:]
        
        features.extend([
            np.mean(begin) / (np.mean(middle) + 1e-8),
            np.mean(begin) / (np.mean(end) + 1e-8),
            np.mean(middle) / (np.mean(end) + 1e-8),
            np.std(begin) / (np.std(middle) + 1e-8),
            np.std(begin) / (np.std(end) + 1e-8),
            np.std(middle) / (np.std(end) + 1e-8)
        ])
        
        # Дополнительные соотношения
        features.extend([
            np.sum(begin) / (np.sum(spectrum) + 1e-8),
            np.sum(middle) / (np.sum(spectrum) + 1e-8),
            np.sum(end) / (np.sum(spectrum) + 1e-8),
            np.max(begin) / (np.max(spectrum) + 1e-8),
            np.max(middle) / (np.max(spectrum) + 1e-8),
            np.max(end) / (np.max(spectrum) + 1e-8),
            np.ptp(begin) / (np.ptp(spectrum) + 1e-8),
            np.ptp(middle) / (np.ptp(spectrum) + 1e-8),
            np.ptp(end) / (np.ptp(spectrum) + 1e-8)
        ])
        
        # 7. ЛОКАЛЬНЫЕ ЭКСТРЕМУМЫ (20 признаков)
        try:
            # Поиск пиков
            peaks, _ = signal.find_peaks(spectrum, height=np.percentile(spectrum, 70))
            valleys, _ = signal.find_peaks(-spectrum, height=-np.percentile(spectrum, 30))
            
            features.extend([
                len(peaks), len(valleys),
                len(peaks) / len(spectrum), len(valleys) / len(spectrum),
                np.mean(spectrum[peaks]) if len(peaks) > 0 else np.mean(spectrum),
                np.mean(spectrum[valleys]) if len(valleys) > 0 else np.mean(spectrum),
                np.std(spectrum[peaks]) if len(peaks) > 0 else 0,
                np.std(spectrum[valleys]) if len(valleys) > 0 else 0,
                np.max(spectrum[peaks]) if len(peaks) > 0 else np.max(spectrum),
                np.min(spectrum[valleys]) if len(valleys) > 0 else np.min(spectrum)
            ])
            
            # Расстояния между пиками
            if len(peaks) > 1:
                peak_distances = np.diff(peaks)
                features.extend([
                    np.mean(peak_distances), np.std(peak_distances),
                    np.min(peak_distances), np.max(peak_distances)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Расстояния между впадинами
            if len(valleys) > 1:
                valley_distances = np.diff(valleys)
                features.extend([
                    np.mean(valley_distances), np.std(valley_distances),
                    np.min(valley_distances), np.max(valley_distances)
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Соотношение пиков к впадинам
            features.extend([
                len(peaks) / (len(valleys) + 1),
                (len(peaks) + len(valleys)) / len(spectrum)
            ])
            
        except:
            # Если поиск пиков не работает, заполняем нулями
            features.extend([0] * 20)
        
        # 8. ВИДОСПЕЦИФИЧНЫЕ ПРИЗНАКИ (100 признаков)
        # Специальные каналы для каждого вида (на основе опыта с 7 видами)
        
        # Ключевые каналы для разных видов (примерные)
        key_channels = {
            'береза': [50, 75, 100, 150, 200, 250, 300],
            'дуб': [60, 90, 120, 180, 240, 280, 320],
            'ель': [40, 80, 110, 160, 210, 260, 310],
            'ель_голубая': [45, 85, 115, 165, 215, 265, 315],
            'клен': [70, 95, 130, 170, 220, 270, 330],
            'липа': [55, 85, 125, 175, 225, 275, 325],
            'осина': [65, 100, 135, 185, 235, 285, 335],
            'сосна': [35, 70, 105, 140, 190, 240, 290]
        }
        
        # Для новых видов используем общие ключевые каналы
        general_key_channels = [30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375]
        
        # Извлекаем значения в ключевых каналах
        for ch in general_key_channels:
            if ch < len(spectrum):
                features.append(spectrum[ch])
            else:
                features.append(0)
        
        # Средние значения в окнах вокруг ключевых каналов
        window_size = 5
        for ch in general_key_channels:
            if ch < len(spectrum):
                start_idx = max(0, ch - window_size)
                end_idx = min(len(spectrum), ch + window_size + 1)
                window_mean = np.mean(spectrum[start_idx:end_idx])
                features.append(window_mean)
            else:
                features.append(0)
        
        # Производные в ключевых каналах
        if len(spectrum) > 1:
            diff_spectrum = np.diff(spectrum)
            for ch in general_key_channels[:10]:  # Берем первые 10
                if ch < len(diff_spectrum):
                    features.append(diff_spectrum[ch])
                else:
                    features.append(0)
        else:
            features.extend([0] * 10)
        
        # Вторые производные в ключевых каналах
        if len(spectrum) > 2:
            diff2_spectrum = np.diff(np.diff(spectrum))
            for ch in general_key_channels[:10]:  # Берем первые 10
                if ch < len(diff2_spectrum):
                    features.append(diff2_spectrum[ch])
                else:
                    features.append(0)
        else:
            features.extend([0] * 10)
        
        # Локальные максимумы и минимумы в регионах
        for start_ch in range(0, min(len(spectrum), 400), 50):
            end_ch = min(start_ch + 50, len(spectrum))
            region = spectrum[start_ch:end_ch]
            if len(region) > 0:
                features.extend([np.max(region), np.min(region), np.argmax(region), np.argmin(region)])
            else:
                features.extend([0, 0, 0, 0])
        
        # Проверяем, что все признаки числовые
        features = [float(f) if not np.isnan(f) and not np.isinf(f) else 0.0 for f in features]
        all_features.append(features)
        
        if (i + 1) % 500 == 0:
            print(f"   Обработано {i + 1} образцов...")
    
    features_array = np.array(all_features)
    print(f"✅ Извлечено {features_array.shape[1]} признаков для {features_array.shape[0]} образцов")
    
    return features_array

def create_transformer_model(input_dim, num_classes, species_names):
    """Создает модель на основе Transformer для 19 видов"""
    
    print("🤖 СОЗДАНИЕ TRANSFORMER МОДЕЛИ...")
    
    # Входной слой
    inputs = Input(shape=(input_dim,), name='features_input')
    
    # Преобразуем в последовательность для Transformer
    x = tf.expand_dims(inputs, axis=1)  # [batch, 1, features]
    
    # Позиционное кодирование
    x = Dense(512, activation='relu', name='embedding')(x)
    x = Dropout(0.2)(x)
    
    # Multi-Head Attention блоки
    for i in range(3):
        # Self-attention
        attn_out = MultiHeadAttention(
            num_heads=8, 
            key_dim=64,
            name=f'multihead_attention_{i}'
        )(x, x)
        attn_out = Dropout(0.1)(attn_out)
        x = LayerNormalization(name=f'layer_norm_1_{i}')(x + attn_out)
        
        # Feed forward
        ff_out = Dense(1024, activation='relu', name=f'ff_1_{i}')(x)
        ff_out = Dropout(0.1)(ff_out)
        ff_out = Dense(512, name=f'ff_2_{i}')(ff_out)
        ff_out = Dropout(0.1)(ff_out)
        x = LayerNormalization(name=f'layer_norm_2_{i}')(x + ff_out)
    
    # Global pooling
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Специализированные ветки для групп видов
    
    # Ветка для хвойных
    conifer_species = ['ель', 'ель_голубая', 'лиственница', 'сосна', 'туя']
    conifer_branch = Dense(256, activation='relu', name='conifer_branch_1')(x)
    conifer_branch = BatchNormalization(name='conifer_bn_1')(conifer_branch)
    conifer_branch = Dropout(0.3)(conifer_branch)
    conifer_branch = Dense(128, activation='relu', name='conifer_branch_2')(conifer_branch)
    conifer_branch = BatchNormalization(name='conifer_bn_2')(conifer_branch)
    conifer_branch = Dropout(0.2)(conifer_branch)
    
    # Ветка для лиственных деревьев
    deciduous_species = ['береза', 'дуб', 'клен', 'липа', 'осина', 'ясень', 'каштан', 'орех']
    deciduous_branch = Dense(256, activation='relu', name='deciduous_branch_1')(x)
    deciduous_branch = BatchNormalization(name='deciduous_bn_1')(deciduous_branch)
    deciduous_branch = Dropout(0.3)(deciduous_branch)
    deciduous_branch = Dense(128, activation='relu', name='deciduous_branch_2')(deciduous_branch)
    deciduous_branch = BatchNormalization(name='deciduous_bn_2')(deciduous_branch)
    deciduous_branch = Dropout(0.2)(deciduous_branch)
    
    # Ветка для кустарников и тополей
    shrub_species = ['сирень', 'черемуха', 'рябина', 'тополь_черный', 'тополь_бальзамический', 'ива']
    shrub_branch = Dense(256, activation='relu', name='shrub_branch_1')(x)
    shrub_branch = BatchNormalization(name='shrub_bn_1')(shrub_branch)
    shrub_branch = Dropout(0.3)(shrub_branch)
    shrub_branch = Dense(128, activation='relu', name='shrub_branch_2')(shrub_branch)
    shrub_branch = BatchNormalization(name='shrub_bn_2')(shrub_branch)
    shrub_branch = Dropout(0.2)(shrub_branch)
    
    # Общая ветка
    general_branch = Dense(512, activation='relu', name='general_branch_1')(x)
    general_branch = BatchNormalization(name='general_bn_1')(general_branch)
    general_branch = Dropout(0.4)(general_branch)
    general_branch = Dense(256, activation='relu', name='general_branch_2')(general_branch)
    general_branch = BatchNormalization(name='general_bn_2')(general_branch)
    general_branch = Dropout(0.3)(general_branch)
    general_branch = Dense(128, activation='relu', name='general_branch_3')(general_branch)
    general_branch = BatchNormalization(name='general_bn_3')(general_branch)
    general_branch = Dropout(0.2)(general_branch)
    
    # Объединяем все ветки
    combined = Concatenate(name='combine_branches')([
        conifer_branch, deciduous_branch, shrub_branch, general_branch
    ])
    
    # Финальные слои
    x = Dense(512, activation='relu', name='final_dense_1')(combined)
    x = BatchNormalization(name='final_bn_1')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu', name='final_dense_2')(x)
    x = BatchNormalization(name='final_bn_2')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu', name='final_dense_3')(x)
    x = BatchNormalization(name='final_bn_3')(x)
    x = Dropout(0.2)(x)
    
    # Выходной слой
    outputs = Dense(num_classes, activation='softmax', name='classification_output')(x)
    
    # Создаем модель
    model = Model(inputs=inputs, outputs=outputs, name='Tree_Species_Transformer_19')
    
    # Компилируем модель
    model.compile(
        optimizer=Adam(learning_rate=0.0001, weight_decay=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_ensemble_system(X_train, y_train, X_test, y_test, species_names):
    """Создает ансамбль моделей для максимальной точности"""
    
    print("🎯 СОЗДАНИЕ МЕГА-АНСАМБЛЯ...")
    
    models = {}
    predictions = {}
    
    # 1. Random Forest с оптимизированными параметрами
    print("   Обучение Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    predictions['random_forest'] = rf.predict_proba(X_test)
    
    # 2. Extra Trees
    print("   Обучение Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    et.fit(X_train, y_train)
    models['extra_trees'] = et
    predictions['extra_trees'] = et.predict_proba(X_test)
    
    # 3. Gradient Boosting
    print("   Обучение Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        subsample=0.8
    )
    gb.fit(X_train, y_train)
    models['gradient_boosting'] = gb
    predictions['gradient_boosting'] = gb.predict_proba(X_test)
    
    # 4. SVM
    print("   Обучение SVM...")
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    svm.fit(X_train, y_train)
    models['svm'] = svm
    predictions['svm'] = svm.predict_proba(X_test)
    
    # 5. Neural Network (sklearn)
    print("   Обучение MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    models['mlp'] = mlp
    predictions['mlp'] = mlp.predict_proba(X_test)
    
    return models, predictions

def advanced_ensemble_fusion(predictions, method='weighted_average'):
    """Продвинутое объединение предсказаний"""
    
    if method == 'weighted_average':
        # Веса на основе производительности моделей
        weights = {
            'random_forest': 0.25,
            'extra_trees': 0.20,
            'gradient_boosting': 0.20,
            'svm': 0.15,
            'mlp': 0.20
        }
        
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            ensemble_pred += weights[model_name] * pred
            
    elif method == 'voting':
        # Простое голосование
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
    elif method == 'rank_fusion':
        # Ранговое объединение
        ranked_preds = []
        for pred in predictions.values():
            ranks = np.argsort(np.argsort(-pred, axis=1), axis=1)
            ranked_preds.append(ranks)
        
        avg_ranks = np.mean(ranked_preds, axis=0)
        ensemble_pred = np.argsort(np.argsort(avg_ranks, axis=1), axis=1)
        ensemble_pred = ensemble_pred / ensemble_pred.sum(axis=1, keepdims=True)
    
    return ensemble_pred

def analyze_mega_results(y_true, predictions, species_names, save_prefix="mega_19_species"):
    """Анализ результатов для 19 видов"""
    
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ...")
    
    results = {}
    
    for model_name, pred_proba in predictions.items():
        y_pred = np.argmax(pred_proba, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        results[model_name] = accuracy
        
        print(f"\n🔹 {model_name.upper()}: {accuracy:.3f}")
        
        # Детальный отчет
        report = classification_report(y_true, y_pred, target_names=species_names, output_dict=True, zero_division=0)
        
        print("   По видам:")
        for i, species in enumerate(species_names):
            if species in report:
                precision = report[species]['precision']
                recall = report[species]['recall']
                f1 = report[species]['f1-score']
                print(f"     {species}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Ансамбль
    print("\n🎯 АНСАМБЛЬ РЕЗУЛЬТАТЫ:")
    
    # Различные методы объединения
    ensemble_methods = ['weighted_average', 'voting', 'rank_fusion']
    
    for method in ensemble_methods:
        ensemble_pred_proba = advanced_ensemble_fusion(predictions, method)
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred)
        
        print(f"\n🔸 {method.upper()}: {ensemble_accuracy:.3f}")
        
        # Детальный отчет по видам
        report = classification_report(y_true, ensemble_pred, target_names=species_names, output_dict=True, zero_division=0)
        
        print("   По видам:")
        for i, species in enumerate(species_names):
            if species in report:
                precision = report[species]['precision']
                recall = report[species]['recall']
                f1 = report[species]['f1-score']
                print(f"     {species}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Создаем confusion matrix для лучшего ансамбля
    best_ensemble = advanced_ensemble_fusion(predictions, 'weighted_average')
    best_pred = np.argmax(best_ensemble, axis=1)
    
    # Визуализация
    plt.figure(figsize=(20, 16))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, best_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=species_names, yticklabels=species_names)
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Accuracy by species
    plt.subplot(2, 2, 2)
    species_accuracy = []
    for i, species in enumerate(species_names):
        mask = y_true == i
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true[mask], best_pred[mask])
            species_accuracy.append(acc)
        else:
            species_accuracy.append(0)
    
    bars = plt.bar(range(len(species_names)), species_accuracy, color='skyblue', alpha=0.8)
    plt.title('Accuracy by Species', fontsize=14, fontweight='bold')
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(len(species_names)), species_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Добавляем значения на столбцы
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{species_accuracy[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Model comparison
    plt.subplot(2, 2, 3)
    model_names = list(results.keys())
    model_accuracies = list(results.values())
    
    # Добавляем ансамбль
    ensemble_acc = accuracy_score(y_true, best_pred)
    model_names.append('Ensemble')
    model_accuracies.append(ensemble_acc)
    
    bars = plt.bar(model_names, model_accuracies, color='coral', alpha=0.8)
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{model_accuracies[i]:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
🏆 РЕЗУЛЬТАТЫ 19 ВИДОВ:

📊 Общая точность: {ensemble_acc:.1%}

🥇 Топ-5 видов:
{chr(10).join([f"   {species_names[i]}: {species_accuracy[i]:.1%}" 
               for i in np.argsort(species_accuracy)[-5:][::-1]])}

🔄 Худшие виды:
{chr(10).join([f"   {species_names[i]}: {species_accuracy[i]:.1%}" 
               for i in np.argsort(species_accuracy)[:3]])}

🎯 Лучшая модель: {max(results, key=results.get)}
   Точность: {max(results.values()):.1%}

✅ Виды > 70%: {sum(1 for acc in species_accuracy if acc > 0.7)}
⚡ Виды > 50%: {sum(1 for acc in species_accuracy if acc > 0.5)}
🔧 Виды < 30%: {sum(1 for acc in species_accuracy if acc < 0.3)}
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, ensemble_acc

def save_mega_system(models, scaler, label_encoder, species_names):
    """Сохраняет всю систему"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем модели
    for name, model in models.items():
        joblib.dump(model, f'mega_19_species_{name}_{timestamp}.pkl')
    
    # Сохраняем препроцессоры
    joblib.dump(scaler, f'mega_19_species_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'mega_19_species_label_encoder_{timestamp}.pkl')
    
    # Сохраняем метаданные
    metadata = {
        'species_names': species_names,
        'n_species': len(species_names),
        'timestamp': timestamp,
        'models': list(models.keys())
    }
    
    joblib.dump(metadata, f'mega_19_species_metadata_{timestamp}.pkl')
    
    print(f"✅ Система сохранена с timestamp: {timestamp}")

def main():
    """Главная функция"""
    
    print("🌲" * 20)
    print("🚀 МЕГА-СИСТЕМА ДЛЯ 19 ВИДОВ ДЕРЕВЬЕВ")
    print("🌲" * 20)
    
    start_time = time.time()
    
    # 1. Загрузка данных
    spring_data, spring_labels = load_spring_data_19_species()
    
    # 2. Предобработка спектров
    print("\n🔧 ПРЕДОБРАБОТКА СПЕКТРОВ...")
    X_spring = preprocess_spectra(spring_data)
    y_spring = np.array(spring_labels)
    
    # 3. Извлечение признаков
    X_features = extract_mega_features(X_spring, spring_labels)
    
    # 4. Подготовка меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_spring)
    species_names = label_encoder.classes_
    
    print(f"\n📊 СТАТИСТИКА ДАТАСЕТА:")
    print(f"   Виды: {len(species_names)}")
    print(f"   Образцов: {len(X_features)}")
    print(f"   Признаков: {X_features.shape[1]}")
    
    unique, counts = np.unique(y_encoded, return_counts=True)
    for i, (species, count) in enumerate(zip(species_names, counts)):
        print(f"   {species}: {count} образцов")
    
    # 5. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # 6. Масштабирование признаков
    print("\n🔧 МАСШТАБИРОВАНИЕ ПРИЗНАКОВ...")
    
    # Заполняем NaN значения
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Применяем PowerTransformer для нормализации распределений
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_power = power_transformer.fit_transform(X_train_imputed)
    X_test_power = power_transformer.transform(X_test_imputed)
    
    # Финальное масштабирование
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_power)
    X_test_scaled = scaler.transform(X_test_power)
    
    print(f"   Проверка NaN после обработки: {np.isnan(X_train_scaled).sum()}")
    
    # 7. Создание и обучение ансамбля
    models, predictions = create_ensemble_system(
        X_train_scaled, y_train, X_test_scaled, y_test, species_names
    )
    
    # 8. Анализ результатов
    results, best_accuracy = analyze_mega_results(
        y_test, predictions, species_names, "mega_19_species"
    )
    
    # 9. Сохранение системы
    save_mega_system(models, scaler, label_encoder, species_names)
    
    # 10. Финальный отчет
    total_time = time.time() - start_time
    
    print(f"\n🎉 МЕГА-СИСТЕМА ЗАВЕРШЕНА!")
    print(f"⏱️  Время выполнения: {total_time:.1f} секунд")
    print(f"🏆 Лучшая точность: {best_accuracy:.1%}")
    print(f"📊 Количество видов: {len(species_names)}")
    print(f"🎯 Статус: {'УСПЕХ' if best_accuracy > 0.6 else 'ТРЕБУЕТ УЛУЧШЕНИЙ'}")
    
    return models, results, best_accuracy

if __name__ == "__main__":
    main() 