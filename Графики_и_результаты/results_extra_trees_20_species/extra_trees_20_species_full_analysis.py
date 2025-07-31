#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTRA TREES ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ С ПОЛНЫМ АНАЛИЗОМ ШУМА
Обучение и тестирование на весенних данных с добавлением шума
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
    """Загружает данные для всех 20 видов (включая клен_ам)"""
    
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
            # Специальная обработка для клен_ам (может быть в двух местах)
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
    
    features_array = np.array(features_list)
    print(f"   📊 Извлечено признаков: {features_array.shape[1]}")
    
    return features_array

def add_noise(X, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return X
    
    noise = np.random.normal(0, noise_level * np.std(X), X.shape)
    return X + noise

def train_extra_trees_model(X_train, y_train):
    """Обучает модель Extra Trees"""
    
    print("🌳 ОБУЧЕНИЕ EXTRA TREES...")
    
    # Модель Extra Trees с оптимизированными параметрами
    model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Обучение
    model.fit(X_train, y_train)
    
    return model

def evaluate_model_with_noise(model, X_test, y_test, species_names, noise_levels=[0, 1, 5, 10, 20]):
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
    
    plt.plot(noise_levels, accuracies, 'bo-', linewidth=3, markersize=10, markerfacecolor='blue')
    plt.title('Extra Trees: Точность vs Уровень шума\n20 видов деревьев', fontsize=14, fontweight='bold')
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
                cmap='Blues',
                square=True,
                cbar_kws={'shrink': 0.8})
    
    plt.title('Нормализованная Confusion Matrix\n0% шума', fontsize=12, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=10)
    plt.ylabel('Истинный класс', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Subplot 3: Confusion matrix для 20% шума
    plt.subplot(2, 3, 3)
    cm_noisy = confusion_matrices[20]
    cm_noisy_normalized = cm_noisy.astype('float') / cm_noisy.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_noisy_normalized, 
                xticklabels=species_names, 
                yticklabels=species_names,
                annot=True, 
                fmt='.2f',
                cmap='Reds',
                square=True,
                cbar_kws={'shrink': 0.8})
    
    plt.title('Нормализованная Confusion Matrix\n20% шума', fontsize=12, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=10)
    plt.ylabel('Истинный класс', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Subplot 4: Деградация точности по видам
    plt.subplot(2, 3, 4)
    
    # Вычисляем точность по видам для разных уровней шума
    species_degradation = {}
    for species_idx, species in enumerate(species_names):
        degradation = []
        for noise_level in noise_levels:
            cm = confusion_matrices[noise_level]
            if cm.sum() > 0:
                # Точность для конкретного вида = правильные предсказания / общее количество образцов этого вида
                correct = cm[species_idx, species_idx]
                total = cm[species_idx, :].sum()
                accuracy = correct / total if total > 0 else 0
                degradation.append(accuracy)
            else:
                degradation.append(0)
        species_degradation[species] = degradation
    
    # Показываем только топ-10 видов по стабильности
    stability_scores = []
    for species, degradation in species_degradation.items():
        stability = np.std(degradation)
        stability_scores.append((species, stability, np.mean(degradation)))
    
    stability_scores.sort(key=lambda x: x[1])  # Сортируем по стабильности
    top_stable = stability_scores[:10]
    
    for species, _, _ in top_stable:
        degradation = species_degradation[species]
        plt.plot(noise_levels, degradation, 'o-', label=species, linewidth=2, markersize=6)
    
    plt.title('Деградация точности по видам\n(Топ-10 стабильных)', fontsize=12, fontweight='bold')
    plt.xlabel('Уровень шума (%)', fontsize=10)
    plt.ylabel('Точность вида', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Subplot 5: Сравнение с Alexnet
    plt.subplot(2, 3, 5)
    
    # Данные Alexnet (из предыдущих результатов)
    alexnet_accuracies = [0.993, 0.972, 0.648, 0.337, 0.123]
    extra_trees_accuracies = accuracies
    
    x = np.arange(len(noise_levels))
    width = 0.35
    
    plt.bar(x - width/2, alexnet_accuracies, width, label='1D Alexnet', color='orange', alpha=0.8)
    plt.bar(x + width/2, extra_trees_accuracies, width, label='Extra Trees', color='green', alpha=0.8)
    
    plt.title('Сравнение: Alexnet vs Extra Trees\nУстойчивость к шуму', fontsize=12, fontweight='bold')
    plt.xlabel('Уровень шума (%)', fontsize=10)
    plt.ylabel('Точность', fontsize=10)
    plt.xticks(x, noise_levels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Добавляем аннотации
    for i, (alexnet_acc, et_acc) in enumerate(zip(alexnet_accuracies, extra_trees_accuracies)):
        plt.text(i - width/2, alexnet_acc + 0.02, f'{alexnet_acc:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.text(i + width/2, et_acc + 0.02, f'{et_acc:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Subplot 6: Feature importance (топ-20)
    plt.subplot(2, 3, 6)
    
    # Получаем feature importance из модели
    feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(20)
    
    plt.barh(range(len(importance_df)), importance_df['importance'], color='skyblue', alpha=0.8)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.title('Feature Importance\n(Топ-20 признаков)', fontsize=12, fontweight='bold')
    plt.xlabel('Важность признака', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем
    filename = f'extra_trees_20_species_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Сохранено: {filename}")
    return filename

def create_individual_confusion_matrices(confusion_matrices, species_names, timestamp):
    """Создает отдельные confusion matrices для каждого уровня шума"""
    
    print("📊 СОЗДАНИЕ ОТДЕЛЬНЫХ CONFUSION MATRICES...")
    
    created_files = []
    noise_levels = list(confusion_matrices.keys())
    
    for noise_level in noise_levels:
        print(f"   🎨 Создание confusion matrix для {noise_level}% шума...")
        
        cm = confusion_matrices[noise_level]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(18, 16))
        
        sns.heatmap(cm_normalized, 
                   xticklabels=species_names, 
                   yticklabels=species_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   square=True,
                   linewidths=0.5,
                   vmin=0, vmax=1,
                   cbar_kws={'shrink': 0.8, 'label': 'Вероятность'})
        
        # Вычисляем общую точность
        total_accuracy = np.trace(cm) / np.sum(cm)
        
        plt.title(f'EXTRA TREES: НОРМАЛИЗОВАННАЯ CONFUSION MATRIX\n' +
                 f'Уровень шума: {noise_level}% | Общая точность: {total_accuracy:.1%}\n' +
                 f'Каждая строка суммируется в 1.0 (100%)',
                 fontsize=16, fontweight='bold', pad=30)
        
        plt.xlabel('Предсказанный класс', fontsize=14)
        plt.ylabel('Истинный класс', fontsize=14)
        
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        
        plt.tight_layout()
        
        filename = f'extra_trees_20_normalized_confusion_matrix_{noise_level}percent_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        created_files.append(filename)
        print(f"     ✅ {filename}")
    
    return created_files

def save_results(model, scaler, label_encoder, results, confusion_matrices, species_names, timestamp):
    """Сохраняет результаты и модели"""
    
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    
    # Сохраняем модель
    model_filename = f'extra_trees_20_species_model_{timestamp}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"   📦 Модель: {model_filename}")
    
    # Сохраняем scaler
    scaler_filename = f'extra_trees_20_species_scaler_{timestamp}.pkl'
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   📦 Scaler: {scaler_filename}")
    
    # Сохраняем label encoder
    encoder_filename = f'extra_trees_20_species_label_encoder_{timestamp}.pkl'
    with open(encoder_filename, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   📦 Label Encoder: {encoder_filename}")
    
    # Сохраняем результаты шума
    results_filename = f'extra_trees_20_species_noise_results_{timestamp}.txt'
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("EXTRA TREES: АНАЛИЗ УСТОЙЧИВОСТИ К ШУМУ\n")
        f.write("20 видов деревьев\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ОБЩИЕ РЕЗУЛЬТАТЫ:\n")
        for noise_level, accuracy in results.items():
            f.write(f"  {noise_level}% шума: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        # Детальные результаты по видам для каждого уровня шума
        for noise_level in sorted(results.keys()):
            f.write(f"ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ - {noise_level}% ШУМА:\n")
            f.write("-" * 40 + "\n")
            
            cm = confusion_matrices[noise_level]
            
            # Точность по видам
            for i, species in enumerate(species_names):
                if cm[i, :].sum() > 0:
                    species_accuracy = cm[i, i] / cm[i, :].sum()
                    f.write(f"  {species}: {species_accuracy:.3f} ({species_accuracy*100:.1f}%)\n")
                else:
                    f.write(f"  {species}: 0.000 (0.0%)\n")
            
            f.write("\n")
    
    print(f"   📊 Результаты: {results_filename}")
    
    return model_filename, scaler_filename, encoder_filename, results_filename

def main():
    """Главная функция"""
    
    print("🌳" * 60)
    print("🌳 EXTRA TREES ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ")
    print("🌳 ПОЛНЫЙ АНАЛИЗ УСТОЙЧИВОСТИ К ШУМУ")
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
    y = label_encoder.fit_transform(labels)
    species_names = label_encoder.classes_
    
    print(f"\n📊 ФИНАЛЬНЫЕ ДАННЫЕ:")
    print(f"   🔢 Форма признаков: {X_features.shape}")
    print(f"   🏷️  Количество классов: {len(species_names)}")
    print(f"   📋 Виды: {list(species_names)}")
    
    # 5. Обработка NaN значений
    print("\n🔧 ОБРАБОТКА NaN ЗНАЧЕНИЙ...")
    imputer = SimpleImputer(strategy='mean')
    X_features_clean = imputer.fit_transform(X_features)
    
    # 6. Нормализация признаков
    print("⚖️ НОРМАЛИЗАЦИЯ ПРИЗНАКОВ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features_clean)
    
    # 7. Разделение на train/test
    print("✂️ РАЗДЕЛЕНИЕ НА TRAIN/TEST...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   📊 Train: {X_train.shape[0]} образцов")
    print(f"   📊 Test: {X_test.shape[0]} образцов")
    
    # 8. Обучение модели
    global model  # Для доступа в функции создания графиков
    model = train_extra_trees_model(X_train, y_train)
    
    # 9. Анализ устойчивости к шуму
    results, confusion_matrices = evaluate_model_with_noise(
        model, X_test, y_test, species_names
    )
    
    # 10. Создание визуализаций
    analysis_file = create_noise_analysis_visualizations(
        results, confusion_matrices, species_names, timestamp
    )
    
    # 11. Создание отдельных confusion matrices
    cm_files = create_individual_confusion_matrices(
        confusion_matrices, species_names, timestamp
    )
    
    # 12. Сохранение результатов
    model_file, scaler_file, encoder_file, results_file = save_results(
        model, scaler, label_encoder, results, confusion_matrices, species_names, timestamp
    )
    
    print(f"\n🎉 АНАЛИЗ EXTRA TREES ЗАВЕРШЕН!")
    print(f"📁 Созданные файлы:")
    print(f"   🌳 Модель: {model_file}")
    print(f"   ⚖️ Scaler: {scaler_file}")
    print(f"   🏷️  Label Encoder: {encoder_file}")
    print(f"   📊 Результаты: {results_file}")
    print(f"   📈 Общий анализ: {analysis_file}")
    print(f"   📊 Confusion matrices:")
    for cm_file in cm_files:
        noise_level = cm_file.split('_')[6].replace('percent', '')
        print(f"     📊 {noise_level}% шума: {cm_file}")
    
    print(f"\n🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ EXTRA TREES:")
    for noise_level, accuracy in results.items():
        print(f"   📊 {noise_level}% шума: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\n✨ Все файлы готовы для перемещения в папку results_extra_trees_20_species!")

if __name__ == "__main__":
    main() 