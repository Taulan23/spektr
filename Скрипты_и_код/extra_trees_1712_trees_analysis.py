#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTRA TREES АНАЛИЗ С 1712 ДЕРЕВЬЯМИ И MAX_DEPTH=NONE
По запросу исследователя: расчеты с n_estimators=1712 и max_depth=None
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import pickle
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_20_species_data():
    """Загружает данные для всех 20 видов"""
    
    spring_folder = "Спектры, весенний период, 20 видов"
    
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
            np.mean((spectrum - np.mean(spectrum))**3),  # skewness
            np.mean((spectrum - np.mean(spectrum))**4)   # kurtosis
        ])
        
        # Производная
        derivative = np.diff(spectrum)
        features.extend([
            np.mean(derivative),
            np.std(derivative),
            np.max(np.abs(derivative))
        ])
        
        # Энергетические характеристики
        n_bands = 10
        band_size = len(spectrum) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(spectrum))
            if start_idx < len(spectrum):
                band_energy = np.sum(spectrum[start_idx:end_idx] ** 2)
                features.append(band_energy)
            else:
                features.append(0)
        
        # Отношения между частями спектра
        mid = len(spectrum) // 2
        first_half = np.mean(spectrum[:mid])
        second_half = np.mean(spectrum[mid:])
        ratio = first_half / second_half if second_half > 0 else 0
        features.append(ratio)
        
        # Дополнительные статистики
        features.extend([
            np.percentile(spectrum, 5),
            np.percentile(spectrum, 95),
            np.percentile(spectrum, 50) - np.percentile(spectrum, 25),  # IQR
        ])
        
        features_list.append(features)
    
    return np.array(features_list)

def add_noise(X, noise_level):
    """Добавляет аддитивный шум к данным"""
    if noise_level == 0:
        return X
    
    # АДДИТИВНЫЙ ШУМ: каждый спектральный отсчет получает СВОЙ случайный шум
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def train_extra_trees_1712_model(X_train, y_train):
    """Обучает модель Extra Trees с 1712 деревьями и max_depth=None"""
    
    print("🌳 ОБУЧЕНИЕ EXTRA TREES С 1712 ДЕРЕВЬЯМИ...")
    print("   📋 Параметры: n_estimators=1712, max_depth=None")
    
    # Модель Extra Trees с запрошенными параметрами
    model = ExtraTreesClassifier(
        n_estimators=1712,  # Запрошенное количество деревьев
        max_depth=None,     # Запрошенная максимальная глубина
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"   🔧 Параметры модели:")
    print(f"      - n_estimators: {model.n_estimators}")
    print(f"      - max_depth: {model.max_depth}")
    print(f"      - min_samples_split: {model.min_samples_split}")
    print(f"      - min_samples_leaf: {model.min_samples_leaf}")
    print(f"      - max_features: {model.max_features}")
    
    # Обучение
    print("   🚀 Начало обучения...")
    model.fit(X_train, y_train)
    
    print("   ✅ Обучение завершено!")
    
    return model

def evaluate_model_with_noise(model, X_test, y_test, species_names, noise_levels=[0, 1, 5, 10]):
    """Оценивает модель на разных уровнях шума"""
    
    print("🔍 АНАЛИЗ УСТОЙЧИВОСТИ К ШУМУ...")
    
    results = {}
    confusion_matrices = {}
    
    for noise_level in noise_levels:
        print(f"\n   📊 Тестирование с {noise_level}% шума...")
        
        # Добавляем шум
        X_test_noisy = add_noise(X_test, noise_level / 100.0)
        
        # Предсказания
        y_pred = model.predict(X_test_noisy)
        
        # Точность
        accuracy = accuracy_score(y_test, y_pred)
        results[noise_level] = accuracy
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=range(len(species_names)))
        confusion_matrices[noise_level] = cm
        
        print(f"     🎯 Точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Детальный отчет для чистых данных
        if noise_level == 0:
            print("\n📋 ДЕТАЛЬНЫЙ CLASSIFICATION REPORT (0% шума):")
            report = classification_report(y_test, y_pred, target_names=species_names, zero_division=0)
            print(report)
    
    return results, confusion_matrices

def create_noise_analysis_visualizations(results, confusion_matrices, species_names, timestamp):
    """Создает визуализации анализа шума"""
    
    print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ...")
    
    # График 1: Точность vs уровень шума
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Общая точность
    plt.subplot(2, 3, 1)
    noise_levels = list(results.keys())
    accuracies = list(results.values())
    
    plt.plot(noise_levels, accuracies, 'ro-', linewidth=3, markersize=10, markerfacecolor='red')
    plt.title('Extra Trees (1712 деревьев): Точность vs Уровень шума\n20 видов деревьев', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень шума (%)', fontsize=12)
    plt.ylabel('Точность', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Добавляем аннотации
    for noise, acc in zip(noise_levels, accuracies):
        plt.annotate(f'{acc:.3f}', (noise, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    # Subplot 2: Confusion matrix для 0% шума
    plt.subplot(2, 3, 2)
    cm_clean = confusion_matrices[0]
    cm_normalized = cm_clean.astype('float') / cm_clean.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, 
                xticklabels=species_names, 
                yticklabels=species_names,
                annot=True, 
                fmt='.2f',
                cmap='Reds',
                square=True,
                cbar_kws={'shrink': 0.8})
    
    plt.title('Нормализованная Confusion Matrix\n0% шума (1712 деревьев)', fontsize=12, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=10)
    plt.ylabel('Истинный класс', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Subplot 3: Confusion matrix для 10% шума
    plt.subplot(2, 3, 3)
    cm_noisy = confusion_matrices[10]
    cm_normalized_noisy = cm_noisy.astype('float') / cm_noisy.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized_noisy, 
                xticklabels=species_names, 
                yticklabels=species_names,
                annot=True, 
                fmt='.2f',
                cmap='Reds',
                square=True,
                cbar_kws={'shrink': 0.8})
    
    plt.title('Нормализованная Confusion Matrix\n10% шума (1712 деревьев)', fontsize=12, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=10)
    plt.ylabel('Истинный класс', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Subplot 4: Сравнение точности по видам (0% vs 20% шума)
    plt.subplot(2, 3, 4)
    
    # Вычисляем точность по видам
    cm_clean = confusion_matrices[0]
    cm_noisy = confusion_matrices[10]
    
    accuracy_clean = np.diag(cm_clean) / np.sum(cm_clean, axis=1)
    accuracy_noisy = np.diag(cm_noisy) / np.sum(cm_noisy, axis=1)
    
    x = np.arange(len(species_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracy_clean, width, label='0% шума', alpha=0.8, color='green')
    plt.bar(x + width/2, accuracy_noisy, width, label='10% шума', alpha=0.8, color='red')
    
    plt.xlabel('Виды деревьев', fontsize=10)
    plt.ylabel('Точность', fontsize=10)
    plt.title('Точность по видам: 0% vs 10% шума\n(1712 деревьев)', fontsize=12, fontweight='bold')
    plt.xticks(x, species_names, rotation=45, ha='right', fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: Потеря точности при шуме
    plt.subplot(2, 3, 5)
    accuracy_loss = accuracy_clean - accuracy_noisy
    
    plt.bar(x, accuracy_loss, color='orange', alpha=0.8)
    plt.xlabel('Виды деревьев', fontsize=10)
    plt.ylabel('Потеря точности', fontsize=10)
    plt.title('Потеря точности при 10% шуме\n(1712 деревьев)', fontsize=12, fontweight='bold')
    plt.xticks(x, species_names, rotation=45, ha='right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Общая статистика
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Создаем текстовую статистику
    stats_text = f"""
    📊 СТАТИСТИКА МОДЕЛИ (1712 деревьев)
    
    🎯 Общая точность:
       • 0% шума: {results[0]:.3f} ({results[0]*100:.1f}%)
       • 1% шума: {results[1]:.3f} ({results[1]*100:.1f}%)
       • 5% шума: {results[5]:.3f} ({results[5]*100:.1f}%)
               • 10% шума: {results[10]:.3f} ({results[10]*100:.1f}%)
    
    📉 Потеря точности:
       • 0% → 10%: {results[0] - results[10]:.3f} ({(results[0] - results[10])*100:.1f}%)
    
    🌳 Параметры модели:
       • n_estimators: 1712
       • max_depth: None
       • min_samples_split: 5
       • min_samples_leaf: 2
       • max_features: sqrt
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = f'extra_trees_1712_noise_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 График сохранен: {filename}")
    plt.show()

def create_individual_confusion_matrices(confusion_matrices, species_names, timestamp):
    """Создает отдельные confusion matrices для каждого уровня шума"""
    
    print("📋 СОЗДАНИЕ ОТДЕЛЬНЫХ CONFUSION MATRICES...")
    
    for noise_level in [0, 1, 5, 10]:
        plt.figure(figsize=(12, 10))
        
        cm = confusion_matrices[noise_level]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                    xticklabels=species_names, 
                    yticklabels=species_names,
                    annot=True, 
                    fmt='.2f',
                    cmap='Reds',
                    square=True,
                    cbar_kws={'shrink': 0.8})
        
        plt.title(f'Extra Trees (1712 деревьев): Нормализованная Confusion Matrix\n{noise_level}% шума', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Предсказанный класс', fontsize=12)
        plt.ylabel('Истинный класс', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        filename = f'extra_trees_1712_confusion_matrix_{noise_level}percent_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 Confusion matrix {noise_level}% сохранена: {filename}")
        plt.close()

def save_results(model, scaler, label_encoder, results, confusion_matrices, species_names, timestamp):
    """Сохраняет результаты анализа"""
    
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    
    # Сохраняем модель
    model_filename = f'extra_trees_1712_model_{timestamp}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"   🧠 Модель сохранена: {model_filename}")
    
    # Сохраняем предобработчики
    scaler_filename = f'extra_trees_1712_scaler_{timestamp}.pkl'
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   🔧 Scaler сохранен: {scaler_filename}")
    
    label_encoder_filename = f'extra_trees_1712_label_encoder_{timestamp}.pkl'
    with open(label_encoder_filename, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   🏷️ Label encoder сохранен: {label_encoder_filename}")
    
    # Сохраняем результаты в текстовый файл
    results_filename = f'extra_trees_1712_results_{timestamp}.txt'
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("РЕЗУЛЬТАТЫ EXTRA TREES С 1712 ДЕРЕВЬЯМИ И MAX_DEPTH=NONE\n")
        f.write("="*80 + "\n\n")
        
        f.write("📋 ПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write(f"   • n_estimators: {model.n_estimators}\n")
        f.write(f"   • max_depth: {model.max_depth}\n")
        f.write(f"   • min_samples_split: {model.min_samples_split}\n")
        f.write(f"   • min_samples_leaf: {model.min_samples_leaf}\n")
        f.write(f"   • max_features: {model.max_features}\n\n")
        
        f.write("📊 РЕЗУЛЬТАТЫ ПО УРОВНЯМ ШУМА:\n")
        for noise_level in sorted(results.keys()):
            accuracy = results[noise_level]
            f.write(f"   • {noise_level}% шума: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        f.write(f"\n📈 ОБЩАЯ СТАТИСТИКА:\n")
        f.write(f"   • Максимальная точность: {max(results.values()):.4f}\n")
        f.write(f"   • Минимальная точность: {min(results.values()):.4f}\n")
        f.write(f"   • Потеря точности (0% → 10%): {results[0] - results[10]:.4f}\n")
        f.write(f"   • Относительная потеря: {((results[0] - results[10]) / results[0] * 100):.2f}%\n")
        
        f.write(f"\n🌳 КОЛИЧЕСТВО ВИДОВ: {len(species_names)}\n")
        f.write(f"📊 РАЗМЕР ДАННЫХ: {model.n_features_in_} признаков\n")
        
    print(f"   📄 Результаты сохранены: {results_filename}")

def main():
    """Главная функция"""
    
    print("="*80)
    print("🌳 EXTRA TREES АНАЛИЗ С 1712 ДЕРЕВЬЯМИ И MAX_DEPTH=NONE")
    print("="*80)
    print("📋 По запросу исследователя")
    print("="*80)
    
    # Временная метка
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Загрузка данных
    all_data, all_labels, species_counts = load_20_species_data()
    
    if len(all_data) == 0:
        print("❌ Не удалось загрузить данные!")
        return
    
    # Предобработка спектров
    X = preprocess_spectra(all_data)
    
    # Извлечение признаков
    X_features = extract_enhanced_features(X)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(all_labels)
    species_names = label_encoder.classes_
    
    print(f"\n📊 ФИНАЛЬНАЯ ФОРМА ДАННЫХ:")
    print(f"   • X: {X_features.shape}")
    print(f"   • y: {y.shape}")
    print(f"   • Классы: {len(species_names)}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📏 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   • Обучающая выборка: {X_train.shape}")
    print(f"   • Тестовая выборка: {X_test.shape}")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели с 1712 деревьями
    model = train_extra_trees_1712_model(X_train_scaled, y_train)
    
    # Оценка модели с шумом
    results, confusion_matrices = evaluate_model_with_noise(
        model, X_test_scaled, y_test, species_names
    )
    
    # Создание визуализаций
    create_noise_analysis_visualizations(results, confusion_matrices, species_names, timestamp)
    create_individual_confusion_matrices(confusion_matrices, species_names, timestamp)
    
    # Сохранение результатов
    save_results(model, scaler, label_encoder, results, confusion_matrices, species_names, timestamp)
    
    print("\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("🎯 Результаты с 1712 деревьями и max_depth=None готовы")
    print("📊 Все файлы сохранены с временной меткой:", timestamp)
    print("="*80)

if __name__ == "__main__":
    main() 