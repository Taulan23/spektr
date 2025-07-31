#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕННЫЙ анализ вероятностей для осины и сирени при 10% шуме
Extra Trees классификатор, 20 видов деревьев
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем seed для воспроизводимости
np.random.seed(42)

def load_20_species_data():
    """Загружает данные для 20 весенних видов"""
    
    spring_folder = "Исходные_данные/Спектры, весенний период, 20 видов"
    
    print("🌱 ЗАГРУЗКА ДАННЫХ 20 ВЕСЕННИХ ВИДОВ...")
    
    # Получаем все папки в правильном порядке
    all_folders = [f for f in os.listdir(spring_folder) 
                   if os.path.isdir(os.path.join(spring_folder, f)) and not f.startswith('.')]
    all_folders = sorted(all_folders)
    
    # Добавляем клен_ам из основной папки если его нет
    if "клен_ам" not in all_folders:
        if os.path.exists("клен_ам"):
            all_folders.append("клен_ам")
    
    print(f"   📁 Найдено папок: {len(all_folders)}")
    
    all_data = []
    all_labels = []
    species_counts = {}
    
    for species in all_folders:
        if species == "клен_ам":
            # Специальная обработка для клен_ам
            species_folder = None
            
            # Проверяем в основной папке
            main_folder_path = os.path.join("клен_ам", "клен_ам")
            if os.path.exists(main_folder_path):
                species_folder = main_folder_path
            elif os.path.exists("клен_ам"):
                species_folder = "клен_ам"
            
            # Проверяем в папке весенних данных
            if species_folder is None:
                spring_path = os.path.join(spring_folder, species)
                if os.path.exists(spring_path):
                    subfolder_path = os.path.join(spring_path, species)
                    if os.path.exists(subfolder_path):
                        species_folder = subfolder_path
                    else:
                        species_folder = spring_path
        else:
            species_folder = os.path.join(spring_folder, species)
        
        if species_folder is None or not os.path.exists(species_folder):
            print(f"   ⚠️  {species}: папка не найдена")
            continue
            
        files = glob.glob(os.path.join(species_folder, "*.xlsx"))
        
        print(f"   🌳 {species}: {len(files)} файлов")
        species_counts[species] = len(files)
        
        species_data = []
        for file in files:
            try:
                df = pd.read_excel(file, header=None)
                spectrum = df.iloc[:, 1].values  # Вторая колонка - спектр
                spectrum = spectrum[~pd.isna(spectrum)]  # Убираем NaN
                species_data.append(spectrum)
            except Exception as e:
                print(f"     ❌ Ошибка в файле {file}: {e}")
                continue
        
        if species_data:
            all_data.extend(species_data)
            all_labels.extend([species] * len(species_data))
    
    print(f"\n📊 ИТОГО ЗАГРУЖЕНО:")
    for species, count in species_counts.items():
        print(f"   🌳 {species}: {count} спектров")
    
    print(f"\n✅ Общий итог: {len(all_data)} спектров, {len(set(all_labels))} видов")
    
    return all_data, all_labels, species_counts

def extract_features(spectra_list):
    """Извлекает признаки из спектров"""
    
    print("🔧 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ...")
    
    # Находим минимальную длину
    min_length = min(len(spectrum) for spectrum in spectra_list)
    print(f"   📏 Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до минимальной длины
    processed_spectra = []
    for spectrum in spectra_list:
        truncated = spectrum[:min_length]
        processed_spectra.append(truncated)
    
    # Преобразуем в numpy array
    X = np.array(processed_spectra)
    print(f"   📊 Финальная форма данных: {X.shape}")
    
    return X

def add_noise(X, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return X
    noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
    return X + noise

def create_corrected_excel_analysis(model, X_test, y_test, tree_types, noise_level=10):
    """Создает исправленный Excel анализ для осины и сирени"""
    
    print(f"\n📊 СОЗДАНИЕ ИСПРАВЛЕННОГО EXCEL АНАЛИЗА ДЛЯ ОСИНЫ И СИРЕНИ...")
    print(f"Уровень шума: {noise_level}%")
    
    # Находим индексы осины и сирени в правильном порядке
    osina_idx = np.where(tree_types == 'осина')[0][0]
    sirene_idx = np.where(tree_types == 'сирень')[0][0]
    
    osina_indices = np.where(y_test == osina_idx)[0]
    sirene_indices = np.where(y_test == sirene_idx)[0]
    
    print(f"   🌳 Осина (индекс {osina_idx}): {len(osina_indices)} образцов")
    print(f"   🌸 Сирень (индекс {sirene_idx}): {len(sirene_indices)} образцов")
    
    # Добавляем шум к тестовым данным
    X_test_noisy = add_noise(X_test, noise_level / 100.0)
    
    # Получаем предсказания
    y_pred_proba = model.predict_proba(X_test_noisy)
    
    # Создаем DataFrame для анализа
    analysis_data = []
    
    # Анализ осины
    for i, idx in enumerate(osina_indices):
        row_data = {
            'Номер_образца': f'Осина_{i+1:02d}',
            'Истинный_класс': 'осина',
            'Уровень_шума': f'{noise_level}%'
        }
        
        # Добавляем вероятности для всех видов
        for j, species in enumerate(tree_types):
            row_data[f'Вероятность_{species}'] = y_pred_proba[idx, j]
        
        # Находим максимальную вероятность
        max_prob_idx = np.argmax(y_pred_proba[idx])
        max_prob = y_pred_proba[idx, max_prob_idx]
        predicted_species = tree_types[max_prob_idx]
        
        row_data['Максимальная_вероятность'] = max_prob
        row_data['Предсказанный_класс'] = predicted_species
        row_data['Правильно_классифицирован'] = predicted_species == 'осина'
        
        # Устанавливаем максимальную вероятность = 1, остальные = 0
        for j, species in enumerate(tree_types):
            if j == max_prob_idx:
                row_data[f'Макс_вероятность_{species}'] = 1.0
            else:
                row_data[f'Макс_вероятность_{species}'] = 0.0
        
        analysis_data.append(row_data)
    
    # Анализ сирени
    for i, idx in enumerate(sirene_indices):
        row_data = {
            'Номер_образца': f'Сирень_{i+1:02d}',
            'Истинный_класс': 'сирень',
            'Уровень_шума': f'{noise_level}%'
        }
        
        # Добавляем вероятности для всех видов
        for j, species in enumerate(tree_types):
            row_data[f'Вероятность_{species}'] = y_pred_proba[idx, j]
        
        # Находим максимальную вероятность
        max_prob_idx = np.argmax(y_pred_proba[idx])
        max_prob = y_pred_proba[idx, max_prob_idx]
        predicted_species = tree_types[max_prob_idx]
        
        row_data['Максимальная_вероятность'] = max_prob
        row_data['Предсказанный_класс'] = predicted_species
        row_data['Правильно_классифицирован'] = predicted_species == 'сирень'
        
        # Устанавливаем максимальную вероятность = 1, остальные = 0
        for j, species in enumerate(tree_types):
            if j == max_prob_idx:
                row_data[f'Макс_вероятность_{species}'] = 1.0
            else:
                row_data[f'Макс_вероятность_{species}'] = 0.0
        
        analysis_data.append(row_data)
    
    # Создаем DataFrame
    df_analysis = pd.DataFrame(analysis_data)
    
    # Вычисляем средние вероятности
    osina_data = df_analysis[df_analysis['Истинный_класс'] == 'осина']
    sirene_data = df_analysis[df_analysis['Истинный_класс'] == 'сирень']
    
    # Средние вероятности для осины
    osina_avg = {
        'Номер_образца': 'СРЕДНЕЕ_ОСИНА',
        'Истинный_класс': 'осина',
        'Уровень_шума': f'{noise_level}%'
    }
    
    # Средние вероятности для сирени
    sirene_avg = {
        'Номер_образца': 'СРЕДНЕЕ_СИРЕНЬ',
        'Истинный_класс': 'сирень',
        'Уровень_шума': f'{noise_level}%'
    }
    
    # Вычисляем средние значения для всех вероятностей
    for species in tree_types:
        osina_avg[f'Вероятность_{species}'] = osina_data[f'Вероятность_{species}'].mean()
        sirene_avg[f'Вероятность_{species}'] = sirene_data[f'Вероятность_{species}'].mean()
        osina_avg[f'Макс_вероятность_{species}'] = osina_data[f'Макс_вероятность_{species}'].mean()
        sirene_avg[f'Макс_вероятность_{species}'] = sirene_data[f'Макс_вероятность_{species}'].mean()
    
    osina_avg['Максимальная_вероятность'] = osina_data['Максимальная_вероятность'].mean()
    sirene_avg['Максимальная_вероятность'] = sirene_data['Максимальная_вероятность'].mean()
    osina_avg['Предсказанный_класс'] = 'осина' if osina_data['Правильно_классифицирован'].mean() > 0.5 else 'другой'
    sirene_avg['Предсказанный_класс'] = 'сирень' if sirene_data['Правильно_классифицирован'].mean() > 0.5 else 'другой'
    osina_avg['Правильно_классифицирован'] = osina_data['Правильно_классифицирован'].mean()
    sirene_avg['Правильно_классифицирован'] = sirene_data['Правильно_классифицирован'].mean()
    
    # Добавляем средние строки
    df_analysis = pd.concat([
        df_analysis,
        pd.DataFrame([osina_avg]),
        pd.DataFrame([sirene_avg])
    ], ignore_index=True)
    
    # Сохраняем в Excel с несколькими листами
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'corrected_detailed_analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Основной лист с детальным анализом
        df_analysis.to_excel(writer, sheet_name='Детальный_анализ', index=False)
        
        # Лист только с вероятностями
        prob_cols = ['Номер_образца', 'Истинный_класс', 'Уровень_шума'] + [f'Вероятность_{species}' for species in tree_types]
        df_analysis[prob_cols].to_excel(writer, sheet_name='Вероятности', index=False)
        
        # Лист с максимальными вероятностями (1/0)
        max_prob_cols = ['Номер_образца', 'Истинный_класс', 'Уровень_шума'] + [f'Макс_вероятность_{species}' for species in tree_types]
        df_analysis[max_prob_cols].to_excel(writer, sheet_name='Максимальные_вероятности', index=False)
        
        # Лист со статистикой
        stats_data = {
            'Метрика': [
                'Количество образцов осины',
                'Количество образцов сирени',
                'Правильно классифицировано осины',
                'Правильно классифицировано сирени',
                'Точность осины',
                'Точность сирени',
                'Средняя максимальная вероятность осины',
                'Средняя максимальная вероятность сирени',
                'Уровень шума',
                'Индекс осины в confusion matrix',
                'Индекс сирени в confusion matrix'
            ],
            'Значение': [
                len(osina_data),
                len(sirene_data),
                osina_data['Правильно_классифицирован'].sum(),
                sirene_data['Правильно_классифицирован'].sum(),
                f"{osina_data['Правильно_классифицирован'].mean():.4f}",
                f"{sirene_data['Правильно_классифицирован'].mean():.4f}",
                f"{osina_data['Максимальная_вероятность'].mean():.4f}",
                f"{sirene_data['Максимальная_вероятность'].mean():.4f}",
                f"{noise_level}%",
                osina_idx,
                sirene_idx
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Статистика', index=False)
    
    print(f"✅ Excel файл сохранен: {filename}")
    
    # Выводим статистику
    print(f"\n📊 СТАТИСТИКА АНАЛИЗА:")
    print(f"   🌳 Осина (индекс {osina_idx}): {len(osina_data)} образцов, точность: {osina_data['Правильно_классифицирован'].mean():.4f}")
    print(f"   🌸 Сирень (индекс {sirene_idx}): {len(sirene_data)} образцов, точность: {sirene_data['Правильно_классифицирован'].mean():.4f}")
    print(f"   📈 Средняя максимальная вероятность осины: {osina_data['Максимальная_вероятность'].mean():.4f}")
    print(f"   📈 Средняя максимальная вероятность сирени: {sirene_data['Максимальная_вероятность'].mean():.4f}")
    
    return filename

def main():
    """Основная функция"""
    
    print("🚀 ИСПРАВЛЕННЫЙ АНАЛИЗ ОСИНЫ И СИРЕНИ ПРИ 10% ШУМЕ")
    print("="*70)
    
    # Загрузка данных
    all_data, all_labels, species_counts = load_20_species_data()
    
    if not all_data:
        print("❌ Нет данных для обучения!")
        return
    
    # Извлечение признаков
    X = extract_features(all_data)
    
    # Кодирование меток
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    tree_types = le.classes_
    
    print(f"\n📊 ФИНАЛЬНЫЕ ДАННЫЕ:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y_encoded.shape}")
    print(f"   Классы: {tree_types}")
    
    # Разделение данных 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"   Тестовая выборка: {X_test.shape[0]} образцов")
    
    # Создание и обучение модели
    print(f"\n🎯 ОБУЧЕНИЕ EXTRA TREES МОДЕЛИ...")
    
    model = ExtraTreesClassifier(
        n_estimators=1712,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Оценка на чистых данных
    print(f"\n📊 ОЦЕНКА НА ЧИСТЫХ ДАННЫХ...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на чистых данных: {accuracy:.7f}")
    
    # Создание исправленного Excel анализа
    excel_file = create_corrected_excel_analysis(model, X_test, y_test, tree_types, noise_level=10)
    
    print(f"\n🎉 ИСПРАВЛЕННЫЙ АНАЛИЗ ГОТОВ!")
    print(f"📁 Excel файл: {excel_file}")

if __name__ == "__main__":
    main() 