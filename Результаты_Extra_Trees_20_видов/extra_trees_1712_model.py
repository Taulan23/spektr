
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extra Trees для классификации 20 весенних видов деревьев
1712 деревьев, максимальная глубина None
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_20_species_data():
    """Загружает данные для всех 20 видов"""
    
    spring_folder = "../Спектры, весенний период, 20 видов"
    
    print("🌱 ЗАГРУЗКА ДАННЫХ 20 ВИДОВ...")
    
    # Получаем все папки
    all_folders = [f for f in os.listdir(spring_folder) 
                   if os.path.isdir(os.path.join(spring_folder, f)) and not f.startswith('.')]
    
    # Добавляем клен_ам из основной папки
    if "клен_ам" not in all_folders:
        if os.path.exists("клен_ам"):
            all_folders.append("клен_ам")
    
    print(f"   📁 Найдено папок: {len(all_folders)}")
    
    all_data = []
    all_labels = []
    species_counts = {}
    
    for species in sorted(all_folders):
        if species == "клен_ам":
            # Специальная обработка для клен_ам (может быть в двух местах)
            species_folder = None
            
            # Проверяем в основной папке
            main_folder_path = os.path.join("../клен_ам", "клен_ам")
            if os.path.exists(main_folder_path):
                species_folder = main_folder_path
            elif os.path.exists("../клен_ам"):
                species_folder = "../клен_ам"
            
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

def preprocess_spectra(spectra_list):
    """Предобработка спектров"""
    
    print("🔧 ПРЕДОБРАБОТКА СПЕКТРОВ...")
    
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

def extract_enhanced_features(X):
    """Извлекает расширенные признаки из спектров"""
    
    print("⚙️ ИЗВЛЕЧЕНИЕ РАСШИРЕННЫХ ПРИЗНАКОВ...")
    
    features_list = []
    
    for spectrum in X:
        features = []
        
        # Базовые статистики
        features.extend([
            np.mean(spectrum),
            np.std(spectrum),
            np.median(spectrum),
            np.min(spectrum),
            np.max(spectrum),
            np.ptp(spectrum),  # peak-to-peak
            np.var(spectrum)
        ])
        
        # Квантили
        quantiles = np.percentile(spectrum, [10, 25, 75, 90])
        features.extend(quantiles)
        
        # Моменты распределения
        features.extend([
            np.sum(spectrum),
            np.sum(spectrum**2),
            np.sum(spectrum**3),
            np.sum(spectrum**4)
        ])
        
        # Спектральные признаки
        if len(spectrum) > 1:
            diff1 = np.diff(spectrum)
            diff2 = np.diff(diff1) if len(diff1) > 1 else [0]
            
            features.extend([
                np.mean(diff1),
                np.std(diff1),
                np.mean(diff2) if len(diff2) > 0 else 0,
                np.std(diff2) if len(diff2) > 0 else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Энергетические признаки
        if len(spectrum) > 10:
            # Разделяем спектр на части
            n_parts = 5
            part_size = len(spectrum) // n_parts
            
            for i in range(n_parts):
                start_idx = i * part_size
                end_idx = start_idx + part_size if i < n_parts - 1 else len(spectrum)
                part = spectrum[start_idx:end_idx]
                
                features.extend([
                    np.mean(part),
                    np.std(part),
                    np.max(part),
                    np.min(part)
                ])
        else:
            # Заполняем нулями если спектр слишком короткий
            features.extend([0] * (5 * 4))
        
        # Дополнительные признаки
        features.extend([
            np.sum(spectrum > np.mean(spectrum)),  # количество точек выше среднего
            np.sum(spectrum < np.mean(spectrum)),  # количество точек ниже среднего
            len(spectrum),  # длина спектра
            np.argmax(spectrum),  # индекс максимума
            np.argmin(spectrum),  # индекс минимума
        ])
        
        features_list.append(features)
    
    return np.array(features_list)

def add_noise(X, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return X
    noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
    return X + noise

def train_extra_trees_model(X_train, y_train):
    """Обучает модель Extra Trees с 1712 деревьями"""
    
    print("🌳 ОБУЧЕНИЕ EXTRA TREES МОДЕЛИ...")
    
    model = ExtraTreesClassifier(
        n_estimators=1712,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"   ✅ Модель обучена с {model.n_estimators} деревьями")
    
    return model

def evaluate_with_noise(model, X_test, y_test, species_names, noise_levels=[1, 5, 10]):
    """Оценивает модель с различными уровнями шума"""
    
    print(f"\n🔊 ОЦЕНКА С ШУМОМ...")
    
    results = {}
    confusion_matrices = {}
    
    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"Тестирование с уровнем шума: {noise_level}%")
        print(f"{'='*60}")
        
        # Добавляем шум к тестовым данным
        X_test_noisy = add_noise(X_test, noise_level / 100.0)
        
        # Предсказание
        y_pred = model.predict(X_test_noisy)
        y_pred_proba = model.predict_proba(X_test_noisy)
        
        # Точность
        accuracy = accuracy_score(y_test, y_pred)
        results[noise_level] = accuracy
        
        print(f"Точность при {noise_level}% шуме: {accuracy:.7f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[noise_level] = cm
        
        print(f"Отчет о классификации:")
        print(classification_report(y_test, y_pred, target_names=species_names, digits=7))
        
        # Создаем тепловую карту матрицы ошибок
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=species_names, yticklabels=species_names)
        plt.title(f'Матрица ошибок - {noise_level}% шума')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{noise_level}percent.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Нормализованная матрица (вероятности по столбцам = 1)
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        cm_normalized = np.nan_to_num(cm_normalized)  # Заменяем NaN на 0
        
        print(f"Нормализованная матрица (сумма по столбцам = 1):")
        print(cm_normalized)
        
        # Сохраняем нормализованную матрицу
        np.save(f'confusion_matrix_{noise_level}percent_normalized.npy', cm_normalized)
        
        # Создаем тепловую карту нормализованной матрицы
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.7f', cmap='Blues', 
                   xticklabels=species_names, yticklabels=species_names)
        plt.title(f'Нормализованная матрица ошибок - {noise_level}% шума')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{noise_level}percent_normalized.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results, confusion_matrices

def create_detailed_excel_analysis(model, X_test, y_test, species_names, timestamp):
    """Создает детальный Excel анализ для осины и сирени"""
    
    print("\n📊 СОЗДАНИЕ ДЕТАЛЬНОГО EXCEL АНАЛИЗА...")
    
    # Находим индексы осины и сирени
    осина_idx = np.where(species_names == 'осина')[0][0]
    сирень_idx = np.where(species_names == 'сирень')[0][0]
    
    target_species = ['осина', 'сирень']
    target_indices = [осина_idx, сирень_idx]
    
    # Создаем Excel файл
    with pd.ExcelWriter(f'detailed_analysis_{timestamp}.xlsx', engine='openpyxl') as writer:
        
        for species_name, target_idx in zip(target_species, target_indices):
            print(f"   📊 Анализ для {species_name}...")
            
            # Находим образцы целевого вида
            target_samples_mask = (y_test == target_idx)
            target_samples = X_test[target_samples_mask]
            
            if len(target_samples) == 0:
                print(f"     ⚠️ Нет образцов для {species_name}")
                continue
            
            print(f"     📈 Найдено {len(target_samples)} образцов {species_name}")
            
            # Применяем 10% шум
            noise = np.random.normal(0, 0.1, target_samples.shape).astype(np.float32)
            target_samples_noisy = target_samples + noise
            
            # Получаем вероятности для всех образцов
            probabilities = model.predict_proba(target_samples_noisy)
            
            # Создаем DataFrame с вероятностями
            prob_df = pd.DataFrame(probabilities, columns=species_names)
            prob_df.insert(0, 'Образец', range(1, len(prob_df) + 1))
            
            # Создаем матрицу с максимальными вероятностями (1 для максимума, 0 для остальных)
            max_prob_matrix = np.zeros_like(probabilities)
            max_indices = np.argmax(probabilities, axis=1)
            max_prob_matrix[np.arange(len(probabilities)), max_indices] = 1
            
            # Создаем DataFrame для максимальных вероятностей
            max_prob_df = pd.DataFrame(max_prob_matrix, columns=species_names)
            max_prob_df.insert(0, 'Образец', range(1, len(max_prob_df) + 1))
            
            # Вычисляем средние вероятности
            mean_probs = prob_df.iloc[:, 1:].mean()
            mean_max_probs = max_prob_df.iloc[:, 1:].mean()
            
            # Создаем DataFrame для средних значений
            mean_df = pd.DataFrame({
                'Вид дерева': species_names,
                'Средняя вероятность': mean_probs.values,
                'Максимальная вероятность (1/0)': mean_max_probs.values
            })
            
            # Сохраняем в Excel
            prob_df.to_excel(writer, sheet_name=f'{species_name}_детальные_вероятности', index=False)
            max_prob_df.to_excel(writer, sheet_name=f'{species_name}_максимальные_вероятности', index=False)
            mean_df.to_excel(writer, sheet_name=f'{species_name}_средние_значения', index=False)
            
            print(f"     ✅ Данные для {species_name} сохранены в Excel")
    
    print(f"💾 Excel файл создан: detailed_analysis_{timestamp}.xlsx")

def main():
    """Основная функция"""
    
    print("🌳" * 60)
    print("🌳 EXTRA TREES ДЛЯ 20 ВЕСЕННИХ ВИДОВ")
    print("🌳 1712 ДЕРЕВЬЕВ, МАКСИМАЛЬНАЯ ГЛУБИНА NONE")
    print("🌳" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Загрузка данных
    spectra_list, labels, species_counts = load_20_species_data()
    
    if len(spectra_list) == 0:
        print("❌ Ошибка: данные не загружены!")
        return
    
    # 2. Предобработка спектров
    X_spectra = preprocess_spectra(spectra_list)
    
    # 3. Извлечение признаков
    X_features = extract_enhanced_features(X_spectra)
    
    # 4. Подготовка меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    species_names = label_encoder.classes_
    
    print(f"\n📊 ФИНАЛЬНЫЕ ДАННЫЕ:")
    print(f"   🔢 Форма признаков: {X_features.shape}")
    print(f"   🏷️  Количество классов: {len(species_names)}")
    print(f"   📋 Виды: {list(species_names)}")
    
    # 5. Разделение на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✂️ РАЗДЕЛЕНИЕ НА TRAIN/TEST:")
    print(f"   📊 Train: {X_train.shape[0]} образцов")
    print(f"   📊 Test: {X_test.shape[0]} образцов")
    
    # 6. Нормализация признаков
    print("\n⚖️ НОРМАЛИЗАЦИЯ ПРИЗНАКОВ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Обучение модели
    model = train_extra_trees_model(X_train_scaled, y_train)
    
    # 8. Оценка на чистых данных
    print("\n📊 ОЦЕНКА НА ЧИСТЫХ ДАННЫХ...")
    y_pred_clean = model.predict(X_test_scaled)
    clean_accuracy = accuracy_score(y_test, y_pred_clean)
    print(f"Точность на чистых данных: {clean_accuracy:.7f}")
    
    # 9. Оценка с шумом
    results, confusion_matrices = evaluate_with_noise(
        model, X_test_scaled, y_test, species_names, noise_levels=[1, 5, 10]
    )
    
    # 10. Создание детального Excel анализа
    create_detailed_excel_analysis(model, X_test_scaled, y_test, species_names, timestamp)
    
    # 11. Сохранение модели
    model_filename = f'extra_trees_1712_model_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    print(f"\n💾 Модель сохранена: {model_filename}")
    
    # 12. Итоговые результаты
    print(f"\n🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ EXTRA TREES (1712 дерева):")
    print(f"   📊 Чистые данные: {clean_accuracy:.7f}")
    for noise_level, accuracy in results.items():
        print(f"   📊 {noise_level}% шума: {accuracy:.7f}")
    
    print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    print(f"📁 Созданные файлы:")
    print(f"   🌳 Модель: {model_filename}")
    print(f"   📊 Excel анализ: detailed_analysis_{timestamp}.xlsx")
    print(f"   📊 Матрицы ошибок:")
    for noise_level in [1, 5, 10]:
        print(f"     📊 {noise_level}% шума: confusion_matrix_{noise_level}percent.png")
        print(f"     📊 {noise_level}% шума (норм.): confusion_matrix_{noise_level}percent_normalized.png")

if __name__ == "__main__":
    main() 