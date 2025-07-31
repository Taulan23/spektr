#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ ВЛИЯНИЯ ШУМА НА 1D ALEXNET ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ
Создание confusion матриц для разных уровней шума
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_20_species_data():
    """Загружает данные для всех 20 видов деревьев"""
    
    print("🌲 ЗАГРУЗКА ДАННЫХ 20 ВИДОВ...")
    
    spring_folder = "Спектры, весенний период, 20 видов"
    
    # Получаем все папки видов
    all_folders = [d for d in os.listdir(spring_folder) 
                   if os.path.isdir(os.path.join(spring_folder, d))]
    
    spring_data = []
    spring_labels = []
    
    for species in sorted(all_folders):
        folder_path = os.path.join(spring_folder, species)
        
        # Для клен_ам проверяем вложенную папку
        if species == "клен_ам":
            subfolder_path = os.path.join(folder_path, species)
            if os.path.exists(subfolder_path):
                folder_path = subfolder_path
        
        files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        print(f"   {species}: {len(files)} файлов")
        
        for file in files:
            try:
                df = pd.read_excel(file)
                if not df.empty and len(df.columns) >= 2:
                    spectrum = df.iloc[:, 1].values
                    if len(spectrum) > 100:
                        spring_data.append(spectrum)
                        spring_labels.append(species)
            except Exception as e:
                continue
    
    unique_labels = sorted(list(set(spring_labels)))
    print(f"✅ Загружено {len(spring_data)} образцов по {len(unique_labels)} видам")
    
    return spring_data, spring_labels

def preprocess_spectra(spectra_list):
    """Предобработка спектров для CNN"""
    
    # Находим минимальную длину
    min_length = min(len(spectrum) for spectrum in spectra_list)
    
    # Обрезаем все спектры до минимальной длины и очищаем от NaN
    processed_spectra = []
    for spectrum in spectra_list:
        spectrum_clean = spectrum[~np.isnan(spectrum)]
        if len(spectrum_clean) >= min_length:
            processed_spectra.append(spectrum_clean[:min_length])
    
    # Преобразуем в numpy массив
    X = np.array(processed_spectra)
    
    # Добавляем размерность канала для CNN
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X

def add_noise(X, noise_level):
    """Добавляет Gaussian шум к спектрам"""
    if noise_level == 0:
        return X
    
    X_noisy = X.copy()
    for i in range(X.shape[0]):
        spectrum = X[i, :, 0]
        noise = np.random.normal(0, noise_level * np.std(spectrum), spectrum.shape)
        X_noisy[i, :, 0] = spectrum + noise
    
    return X_noisy

def load_saved_models():
    """Загружает сохраненные модели и обработчики"""
    
    print("📁 ЗАГРУЗКА СОХРАНЕННЫХ МОДЕЛЕЙ...")
    
    # Находим последние файлы
    model_files = glob.glob("alexnet_20_species_final_*.keras")
    encoder_files = glob.glob("alexnet_20_species_label_encoder_*.pkl")
    scaler_files = glob.glob("alexnet_20_species_scaler_*.pkl")
    
    if not model_files or not encoder_files or not scaler_files:
        raise Exception("Не найдены сохраненные модели!")
    
    # Берем последние файлы
    model_file = sorted(model_files)[-1]
    encoder_file = sorted(encoder_files)[-1]
    scaler_file = sorted(scaler_files)[-1]
    
    print(f"   Модель: {model_file}")
    print(f"   Encoder: {encoder_file}")
    print(f"   Scaler: {scaler_file}")
    
    # Загружаем
    model = keras.models.load_model(model_file)
    label_encoder = joblib.load(encoder_file)
    scaler = joblib.load(scaler_file)
    
    return model, label_encoder, scaler

def create_noise_confusion_matrices(model, X_test, y_test, species_names, noise_levels):
    """Создает confusion матрицы для разных уровней шума"""
    
    print("🔊 АНАЛИЗ ВЛИЯНИЯ ШУМА...")
    
    results = {}
    confusion_matrices = {}
    
    for noise_level in noise_levels:
        print(f"   Тестирование с шумом {noise_level*100:.0f}%...")
        
        # Добавляем шум
        X_noisy = add_noise(X_test, noise_level)
        
        # Предсказания
        y_pred_proba = model.predict(X_noisy, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Точность
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        results[noise_level] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized
        }
        
        print(f"     Точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return results

def plot_noise_confusion_matrices(results, species_names, noise_levels):
    """Создает визуализацию confusion матриц для разных уровней шума"""
    
    print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ...")
    
    n_levels = len(noise_levels)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('🔊 Confusion Matrices для разных уровней шума\n1D Alexnet - 20 видов деревьев', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Убираем лишние subplot'ы
    if n_levels < 6:
        for i in range(n_levels, 6):
            fig.delaxes(axes.flatten()[i])
    
    for idx, noise_level in enumerate(noise_levels):
        if idx >= 6:  # Максимум 6 графиков
            break
            
        ax = axes.flatten()[idx]
        cm_norm = results[noise_level]['confusion_matrix_normalized']
        accuracy = results[noise_level]['accuracy']
        
        # Создаем heatmap
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Настройки осей
        ax.set_xticks(range(len(species_names)))
        ax.set_yticks(range(len(species_names)))
        ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(species_names, fontsize=8)
        
        # Заголовок с точностью
        ax.set_title(f'Шум {noise_level*100:.0f}%\nТочность: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        
        # Добавляем значения в ячейки (только диагональные для читаемости)
        for i in range(len(species_names)):
            value = cm_norm[i, i]
            color = 'white' if value > 0.5 else 'black'
            ax.text(i, i, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontweight='bold', fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Вероятность', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Сохраняем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'alexnet_20_noise_confusion_matrices_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

def create_accuracy_degradation_plot(results, noise_levels):
    """Создает график деградации точности от шума"""
    
    print("📈 СОЗДАНИЕ ГРАФИКА ДЕГРАДАЦИИ ТОЧНОСТИ...")
    
    accuracies = [results[noise]['accuracy'] for noise in noise_levels]
    noise_percentages = [noise * 100 for noise in noise_levels]
    
    plt.figure(figsize=(12, 8))
    
    # Основной график
    plt.plot(noise_percentages, accuracies, 'o-', linewidth=3, markersize=8, 
             color='red', markerfacecolor='darkred', markeredgecolor='white', markeredgewidth=2)
    
    # Добавляем точки с подписями
    for i, (noise, acc) in enumerate(zip(noise_percentages, accuracies)):
        plt.annotate(f'{acc:.3f}', (noise, acc), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontweight='bold', fontsize=12)
    
    # Настройки
    plt.title('🔊 Влияние шума на точность 1D Alexnet (20 видов)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Уровень шума (%)', fontsize=14)
    plt.ylabel('Точность', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Цветовые зоны
    plt.axhspan(0.9, 1.0, alpha=0.2, color='green', label='Отличная точность (>90%)')
    plt.axhspan(0.7, 0.9, alpha=0.2, color='yellow', label='Хорошая точность (70-90%)')
    plt.axhspan(0.0, 0.7, alpha=0.2, color='red', label='Низкая точность (<70%)')
    
    plt.legend(loc='lower right')
    
    # Сохраняем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'alexnet_20_accuracy_degradation_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

def create_detailed_report(results, species_names, noise_levels):
    """Создает детальный отчет"""
    
    print("📋 СОЗДАНИЕ ДЕТАЛЬНОГО ОТЧЕТА...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_lines = [
        "🔊 АНАЛИЗ ВЛИЯНИЯ ШУМА НА 1D ALEXNET (20 ВИДОВ)",
        "=" * 60,
        "",
        f"⏰ Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"🌲 Количество видов: {len(species_names)}",
        f"🔢 Тестируемые уровни шума: {[f'{n*100:.0f}%' for n in noise_levels]}",
        "",
        "📊 РЕЗУЛЬТАТЫ ПО УРОВНЯМ ШУМА:",
        "-" * 40,
    ]
    
    for noise_level in noise_levels:
        accuracy = results[noise_level]['accuracy']
        report_lines.extend([
            f"",
            f"🔊 Шум {noise_level*100:.0f}%:",
            f"   Общая точность: {accuracy:.3f} ({accuracy*100:.1f}%)",
            f"   Статус: {'ОТЛИЧНО' if accuracy > 0.9 else 'ХОРОШО' if accuracy > 0.7 else 'ПЛОХО'}",
        ])
    
    # Добавляем точность по видам для каждого уровня шума
    report_lines.extend([
        "",
        "📋 ДЕТАЛИЗАЦИЯ ПО ВИДАМ:",
        "-" * 40,
    ])
    
    for species_idx, species in enumerate(species_names):
        report_lines.append(f"\n{species}:")
        for noise_level in noise_levels:
            cm_norm = results[noise_level]['confusion_matrix_normalized']
            species_accuracy = cm_norm[species_idx, species_idx]
            report_lines.append(f"   {noise_level*100:2.0f}% шума: {species_accuracy:.3f}")
    
    # Анализ устойчивости видов
    report_lines.extend([
        "",
        "🛡️  АНАЛИЗ УСТОЙЧИВОСТИ К ШУМУ:",
        "-" * 40,
    ])
    
    for species_idx, species in enumerate(species_names):
        accuracies = [results[noise]['confusion_matrix_normalized'][species_idx, species_idx] 
                     for noise in noise_levels]
        degradation = accuracies[0] - accuracies[-1]  # Разница между 0% и max шумом
        
        if degradation < 0.1:
            status = "ОЧЕНЬ УСТОЙЧИВ"
        elif degradation < 0.3:
            status = "УСТОЙЧИВ"
        elif degradation < 0.5:
            status = "УМЕРЕННО УСТОЙЧИВ"
        else:
            status = "ЧУВСТВИТЕЛЕН"
        
        report_lines.append(f"   {species:25} Деградация: {degradation:.3f} ({status})")
    
    # Сохраняем отчет
    filename = f'alexnet_20_noise_analysis_report_{timestamp}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   Отчет сохранен: {filename}")
    return filename

def main():
    """Главная функция"""
    
    print("🔊" * 25)
    print("🔊 АНАЛИЗ ВЛИЯНИЯ ШУМА НА 1D ALEXNET")
    print("🔊" * 25)
    
    # Уровни шума для тестирования
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]  # 0%, 1%, 5%, 10%, 20%
    
    try:
        # 1. Загрузка данных
        spring_data, spring_labels = load_20_species_data()
        
        # 2. Предобработка
        X = preprocess_spectra(spring_data)
        
        # 3. Подготовка меток (используем те же настройки, что и при обучении)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(spring_labels)
        species_names = label_encoder.classes_
        
        # 4. Загрузка обученной модели
        model, saved_label_encoder, saved_scaler = load_saved_models()
        
        # Используем сохраненный label_encoder
        species_names = saved_label_encoder.classes_
        y_encoded = saved_label_encoder.transform(spring_labels)
        
        from tensorflow.keras.utils import to_categorical
        y_categorical = to_categorical(y_encoded)
        
        # 5. Разделение данных (те же настройки)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_encoded
        )
        
        # 6. Нормализация тестовых данных
        print("🔢 НОРМАЛИЗАЦИЯ ТЕСТОВЫХ ДАННЫХ...")
        scaler = StandardScaler()
        X_test_scaled = X_test.copy()
        
        for i in range(X_test.shape[0]):
            X_test_scaled[i, :, 0] = scaler.fit_transform(X_test[i, :, 0].reshape(-1, 1)).flatten()
        
        print(f"📊 Тестовая выборка: {len(X_test_scaled)} образцов")
        print(f"🌲 Видов для анализа: {len(species_names)}")
        
        # 7. Анализ влияния шума
        results = create_noise_confusion_matrices(model, X_test_scaled, y_test, species_names, noise_levels)
        
        # 8. Создание визуализаций
        confusion_file = plot_noise_confusion_matrices(results, species_names, noise_levels)
        degradation_file = create_accuracy_degradation_plot(results, noise_levels)
        
        # 9. Создание отчета
        report_file = create_detailed_report(results, species_names, noise_levels)
        
        # 10. Финальная сводка
        print(f"\n🎉 АНАЛИЗ ШУМА ЗАВЕРШЕН!")
        print(f"📁 Созданные файлы:")
        print(f"   📊 Confusion матрицы: {confusion_file}")
        print(f"   📈 График деградации: {degradation_file}")
        print(f"   📋 Детальный отчет: {report_file}")
        
        # Краткая сводка
        print(f"\n📋 КРАТКАЯ СВОДКА:")
        for noise_level in noise_levels:
            accuracy = results[noise_level]['accuracy']
            print(f"   {noise_level*100:2.0f}% шума: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 