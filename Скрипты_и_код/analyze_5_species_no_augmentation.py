#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ 5 ВИДОВ ДЕРЕВЬЕВ БЕЗ АУГМЕНТАЦИИ ШУМА
Для сравнения с результатами научника
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import glob

def load_spectral_data_5_species():
    """Загрузка данных для 5 видов деревьев"""
    
    # Выбираем 5 видов: береза, дуб, ель, клен, сосна
    species_dirs = ['береза', 'дуб', 'ель', 'клен', 'сосна']
    
    data = []
    labels = []
    
    for species in species_dirs:
        print(f"Загружаем данные для {species}...")
        
        # Путь к данным
        if species == 'клен':
            # Для клена используем данные из весеннего периода
            pattern = f"Спектры, весенний период, 20 видов/клен/*.xlsx"
        else:
            pattern = f"{species}/*.xlsx"
        
        files = glob.glob(pattern)
        
        for file in files[:30]:  # Берем по 30 спектров на вид
            try:
                df = pd.read_excel(file)
                
                # Извлекаем спектральные данные (обычно в столбцах с числовыми значениями)
                spectral_data = []
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        spectral_data.extend(df[col].dropna().values)
                
                if len(spectral_data) > 0:
                    # Нормализуем длину спектра (берем первые 1000 точек)
                    spectral_data = spectral_data[:1000]
                    if len(spectral_data) < 1000:
                        spectral_data.extend([0] * (1000 - len(spectral_data)))
                    
                    data.append(spectral_data)
                    labels.append(species)
                    
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
                continue
    
    return np.array(data), np.array(labels)

def add_noise_to_data(X, noise_level):
    """Добавление шума к данным"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def analyze_5_species_no_augmentation():
    """Основной анализ 5 видов без аугментации"""
    
    print("="*80)
    print("🌳 АНАЛИЗ 5 ВИДОВ ДЕРЕВЬЕВ БЕЗ АУГМЕНТАЦИИ ШУМА")
    print("="*80)
    
    # Загружаем данные
    X, y = load_spectral_data_5_species()
    
    print(f"📊 ЗАГРУЖЕННЫЕ ДАННЫЕ:")
    print(f"   • Общее количество образцов: {len(X)}")
    print(f"   • Размерность признаков: {X.shape[1]}")
    print(f"   • Виды: {np.unique(y)}")
    
    # Кодируем метки
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Разбиваем на обучающую и тестовую выборки (50/50 как у научника)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.5, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📋 РАЗБИЕНИЕ ДАННЫХ:")
    print(f"   • Обучающая выборка: {len(X_train)} образцов")
    print(f"   • Тестовая выборка: {len(X_test)} образцов")
    print(f"   • Соотношение: 50/50 (как у научника)")
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем модель Extra Trees (без аугментации)
    print(f"\n🤖 ОБУЧЕНИЕ МОДЕЛИ:")
    print(f"   • Алгоритм: Extra Trees")
    print(f"   • n_estimators: 1712")
    print(f"   • Аугментация шума: НЕТ")
    
    model = ExtraTreesClassifier(
        n_estimators=1712,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Тестируем на разных уровнях шума
    noise_levels = [0.0, 0.01, 0.10]  # 0%, 1%, 10%
    results = {}
    
    print(f"\n🧪 ТЕСТИРОВАНИЕ НА РАЗНЫХ УРОВНЯХ ШУМА:")
    
    for noise_level in noise_levels:
        print(f"\n📊 Тестирование с {noise_level*100:.0f}% шума:")
        
        # Добавляем шум к тестовым данным
        if noise_level > 0:
            X_test_noisy = add_noise_to_data(X_test_scaled, noise_level)
        else:
            X_test_noisy = X_test_scaled
        
        # Предсказания
        y_pred = model.predict(X_test_noisy)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        results[noise_level] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'predictions': y_pred
        }
        
        print(f"   • Точность: {accuracy:.1%}")
        
        # Детальная статистика по классам
        class_names = le.classes_
        for i, class_name in enumerate(class_names):
            class_accuracy = cm_normalized[i, i]
            print(f"   • {class_name}: {class_accuracy:.1%}")
    
    # Создаем визуализации
    create_visualizations(results, le.classes_)
    
    # Сохраняем результаты
    save_results(results, le.classes_)
    
    print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    print(f"📁 Результаты сохранены в файлы:")
    print(f"   • confusion_matrices_5_species.png")
    print(f"   • results_5_species_no_augmentation.txt")

def create_visualizations(results, class_names):
    """Создание визуализаций"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (noise_level, result) in enumerate(results.items()):
        cm_normalized = result['confusion_matrix_normalized']
        
        # Создаем DataFrame
        df_cm = pd.DataFrame(cm_normalized, 
                           index=class_names, 
                           columns=class_names)
        
        # Тепловая карта
        sns.heatmap(df_cm, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   ax=axes[idx],
                   cbar_kws={'label': 'Вероятность'},
                   square=True,
                   linewidths=0.5,
                   linecolor='white',
                   annot_kws={'size': 10})
        
        accuracy = result['accuracy']
        axes[idx].set_title(f'{noise_level*100:.0f}% шума\nТочность: {accuracy:.1%}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Предсказанный класс')
        axes[idx].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_5_species.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results, class_names):
    """Сохранение результатов в текстовый файл"""
    
    with open('results_5_species_no_augmentation.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("РЕЗУЛЬТАТЫ АНАЛИЗА 5 ВИДОВ ДЕРЕВЬЕВ БЕЗ АУГМЕНТАЦИИ ШУМА\n")
        f.write("="*80 + "\n\n")
        
        f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write("- Алгоритм: Extra Trees\n")
        f.write("- n_estimators: 1712\n")
        f.write("- max_depth: None\n")
        f.write("- Разбиение данных: 50/50 (как у научника)\n")
        f.write("- Аугментация шума: НЕТ\n\n")
        
        for noise_level, result in results.items():
            f.write(f"РЕЗУЛЬТАТЫ ДЛЯ {noise_level*100:.0f}% ШУМА:\n")
            f.write("-" * 50 + "\n")
            
            accuracy = result['accuracy']
            f.write(f"Общая точность: {accuracy:.1%}\n\n")
            
            cm = result['confusion_matrix']
            cm_normalized = result['confusion_matrix_normalized']
            
            f.write("СЫРАЯ МАТРИЦА ОШИБОК:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}: {cm[i].tolist()}\n")
            f.write("\n")
            
            f.write("НОРМАЛИЗОВАННАЯ МАТРИЦА ОШИБОК:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}: {[f'{val:.3f}' for val in cm_normalized[i]]}\n")
            f.write("\n")
            
            f.write("ТОЧНОСТЬ ПО КЛАССАМ:\n")
            for i, class_name in enumerate(class_names):
                class_accuracy = cm_normalized[i, i]
                f.write(f"{class_name}: {class_accuracy:.1%}\n")
            f.write("\n" + "="*80 + "\n\n")

if __name__ == "__main__":
    analyze_5_species_no_augmentation() 