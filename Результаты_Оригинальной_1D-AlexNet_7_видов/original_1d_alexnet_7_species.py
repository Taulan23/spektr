#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОРИГИНАЛЬНАЯ 1D-AlexNet для классификации 7 весенних видов деревьев
Архитектура согласно оригинальной научной статье
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

def load_spring_7_species_data():
    """Загружает данные для 7 весенних видов"""
    
    spring_folder = "Исходные_данные/Спектры, весенний период, 7 видов"
    
    print("🌱 ЗАГРУЗКА ДАННЫХ 7 ВЕСЕННИХ ВИДОВ...")
    
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    
    all_data = []
    all_labels = []
    species_counts = {}
    
    for species in tree_types:
        species_folder = os.path.join(spring_folder, species)
        if not os.path.exists(species_folder):
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

def create_original_1d_alexnet_model(input_shape, num_classes):
    """Создает ОРИГИНАЛЬНУЮ 1D-AlexNet согласно научной статье"""
    
    print("🏗️ СОЗДАНИЕ ОРИГИНАЛЬНОЙ 1D-AlexNet МОДЕЛИ...")
    
    model = keras.Sequential([
        # Группа 1: Первая свертка + пулинг
        layers.Conv1D(filters=96, kernel_size=11, strides=4, padding='same', 
                     activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        
        # Группа 2: Вторая свертка + пулинг
        layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', 
                     activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        
        # Группа 3: Три свертки подряд + пулинг
        layers.Conv1D(filters=384, kernel_size=3, strides=1, padding='same', 
                     activation='relu'),
        layers.Conv1D(filters=384, kernel_size=3, strides=1, padding='same', 
                     activation='relu'),
        layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', 
                     activation='relu'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        
        # Flatten для перехода к полносвязным слоям
        layers.Flatten(),
        
        # Полносвязные слои (оригинальная архитектура)
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Компиляция модели с оригинальными параметрами
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   📊 ОРИГИНАЛЬНАЯ архитектура модели:")
    model.summary()
    
    return model

def add_noise(X, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return X
    noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
    return X + noise

def evaluate_with_noise(model, X_test, y_test, tree_types, noise_levels=[1, 5, 10]):
    """Оценивает модель с различными уровнями шума"""
    
    print(f"\n🔊 ОЦЕНКА ОРИГИНАЛЬНОЙ МОДЕЛИ С ШУМОМ...")
    
    results = {}
    confusion_matrices = {}
    
    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"Тестирование с уровнем шума: {noise_level}%")
        print(f"{'='*60}")
        
        # Добавляем шум к тестовым данным
        X_test_noisy = add_noise(X_test, noise_level / 100.0)
        
        # Предсказание
        y_pred_proba = model.predict(X_test_noisy, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Точность
        accuracy = accuracy_score(y_test_classes, y_pred)
        results[noise_level] = accuracy
        
        print(f"Точность при {noise_level}% шуме: {accuracy:.7f}")
        
        # Анализ вероятностей
        print(f"\n📊 АНАЛИЗ ВЕРОЯТНОСТЕЙ:")
        for i, species in enumerate(tree_types):
            species_probs = y_pred_proba[y_test_classes == i]
            if len(species_probs) > 0:
                max_probs = np.max(species_probs, axis=1)
                mean_max_prob = np.mean(max_probs)
                std_max_prob = np.std(max_probs)
                print(f"   {species}: средняя макс. вероятность = {mean_max_prob:.4f} ± {std_max_prob:.4f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y_test_classes, y_pred)
        confusion_matrices[noise_level] = cm
        
        print(f"\nОтчет о классификации:")
        print(classification_report(y_test_classes, y_pred, target_names=tree_types, digits=7))
        
        # Создаем тепловую карту матрицы ошибок
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=tree_types, yticklabels=tree_types)
        plt.title(f'ОРИГИНАЛЬНАЯ 1D-AlexNet - Матрица ошибок {noise_level}% шума')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.tight_layout()
        plt.savefig(f'original_confusion_matrix_{noise_level}percent.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Нормализованная матрица (вероятности по столбцам = 1)
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        cm_normalized = np.nan_to_num(cm_normalized)  # Заменяем NaN на 0
        
        # Сохраняем нормализованную матрицу
        np.save(f'original_confusion_matrix_{noise_level}percent_normalized.npy', cm_normalized)
        
        # Визуализация нормализованной матрицы
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.7f', cmap='Blues', 
                   xticklabels=tree_types, yticklabels=tree_types)
        plt.title(f'ОРИГИНАЛЬНАЯ 1D-AlexNet - Нормализованная матрица {noise_level}% шума')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.tight_layout()
        plt.savefig(f'original_confusion_matrix_{noise_level}percent_normalized.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results, confusion_matrices

def save_network_params(model, tree_types, timestamp):
    """Сохраняет параметры сети"""
    
    params = {
        'architecture': 'ОРИГИНАЛЬНАЯ 1D-AlexNet',
        'timestamp': timestamp,
        'input_shape': list(model.input_shape[1:]),
        'num_classes': len(tree_types),
        'tree_types': list(tree_types),
        'layers': []
    }
    
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'config': layer.get_config()
        }
        params['layers'].append(layer_info)
    
    filename = f'original_network_params_{timestamp}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Параметры сети сохранены: {filename}")

def main():
    """Основная функция"""
    
    print("🚀 ЗАПУСК ОРИГИНАЛЬНОЙ 1D-AlexNet ДЛЯ 7 ВЕСЕННИХ ВИДОВ")
    print("="*70)
    
    # Загрузка данных
    all_data, all_labels, species_counts = load_spring_7_species_data()
    
    if not all_data:
        print("❌ Нет данных для обучения!")
        return
    
    # Предобработка
    X = preprocess_spectra(all_data)
    
    # Добавляем размерность канала для 1D свертки
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Кодирование меток
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    tree_types = le.classes_
    
    # One-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=len(tree_types))
    
    print(f"\n📊 ФИНАЛЬНЫЕ ДАННЫЕ:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y_onehot.shape}")
    print(f"   Классы: {tree_types}")
    
    # Разделение данных 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"   Тестовая выборка: {X_test.shape[0]} образцов")
    
    # Создание модели
    model = create_original_1d_alexnet_model((X_train.shape[1], 1), len(tree_types))
    
    # Обучение
    print(f"\n🎯 ОБУЧЕНИЕ ОРИГИНАЛЬНОЙ МОДЕЛИ...")
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Оценка на чистых данных
    print(f"\n📊 ОЦЕНКА НА ЧИСТЫХ ДАННЫХ...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точность на чистых данных: {test_accuracy:.7f}")
    
    # Оценка с шумом
    results, confusion_matrices = evaluate_with_noise(model, X_test, y_test, tree_types)
    
    # Сохранение параметров
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_network_params(model, tree_types, timestamp)
    
    # Сохранение модели
    model_filename = f'original_1d_alexnet_7_species_{timestamp}.h5'
    model.save(model_filename)
    print(f"✅ Модель сохранена: {model_filename}")
    
    # Визуализация истории обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучающая точность')
    plt.plot(history.history['val_accuracy'], label='Валидационная точность')
    plt.title('ОРИГИНАЛЬНАЯ 1D-AlexNet - Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучающая ошибка')
    plt.plot(history.history['val_loss'], label='Валидационная ошибка')
    plt.title('ОРИГИНАЛЬНАЯ 1D-AlexNet - Ошибка')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'original_training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 ОРИГИНАЛЬНАЯ 1D-AlexNet ГОТОВА!")
    print(f"📁 Все файлы сохранены с префиксом 'original_'")

if __name__ == "__main__":
    main() 