#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D-AlexNet для классификации 7 весенних видов деревьев
Архитектура согласно изображению с 3 группами сверточных слоев
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
    
    spring_folder = "../Исходные_данные/Спектры, весенний период, 7 видов"
    
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

def create_1d_alexnet_model(input_shape, num_classes):
    """Создает РЕАЛИСТИЧНУЮ 1D-AlexNet с разными вероятностями для разных классов"""
    
    print("🏗️ СОЗДАНИЕ РЕАЛИСТИЧНОЙ 1D-AlexNet МОДЕЛИ...")
    
    model = keras.Sequential([
        # Группа 1: Более сложная первая свертка
        layers.Conv1D(filters=32, kernel_size=50, strides=4, padding='same', 
                     activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Dropout(0.25),
        
        # Группа 2: Увеличиваем количество фильтров
        layers.Conv1D(filters=64, kernel_size=50, strides=1, padding='same', 
                     activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Dropout(0.25),
        
        # Группа 3: Более сложные свертки
        layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', 
                     activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', 
                     activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', 
                     activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Dropout(0.25),
        
        # Flatten для перехода к полносвязным слоям
        layers.Flatten(),
        
        # Более сложные полносвязные слои
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Компиляция модели с более сложным оптимизатором
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   📊 РЕАЛИСТИЧНАЯ архитектура модели:")
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
        y_pred_proba = model.predict(X_test_noisy, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Точность
        accuracy = accuracy_score(y_test_classes, y_pred)
        results[noise_level] = accuracy
        
        print(f"Точность при {noise_level}% шуме: {accuracy:.7f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y_test_classes, y_pred)
        confusion_matrices[noise_level] = cm
        
        print(f"Отчет о классификации:")
        print(classification_report(y_test_classes, y_pred, target_names=tree_types, digits=7))
        
        # Создаем тепловую карту матрицы ошибок
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=tree_types, yticklabels=tree_types)
        plt.title(f'Матрица ошибок - {noise_level}% шума')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
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
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.7f', cmap='Blues', 
                   xticklabels=tree_types, yticklabels=tree_types)
        plt.title(f'Нормализованная матрица ошибок - {noise_level}% шума')
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{noise_level}percent_normalized.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results, confusion_matrices

def save_network_params(model, tree_types, timestamp):
    """Сохраняет параметры сети"""
    
    params = {
        'architecture': '1D-AlexNet',
        'timestamp': timestamp,
        'input_shape': list(model.input_shape[1:]),
        'num_classes': len(tree_types),
        'tree_types': list(tree_types),
        'layers': []
    }
    
    for i, layer in enumerate(model.layers):
        layer_info = {
            'layer_number': i + 1,
            'type': layer.__class__.__name__,
            'config': layer.get_config()
        }
        params['layers'].append(layer_info)
    
    # Сохраняем параметры
    with open(f'network_params_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Параметры сети сохранены: network_params_{timestamp}.json")

def main():
    """Основная функция"""
    
    print("🌳" * 60)
    print("🌳 1D-AlexNet ДЛЯ 7 ВЕСЕННИХ ВИДОВ")
    print("🌳 КЛАССИФИКАЦИЯ С ШУМОМ")
    print("🌳" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Загрузка данных
    spectra_list, labels, species_counts = load_spring_7_species_data()
    
    if len(spectra_list) == 0:
        print("❌ Ошибка: данные не загружены!")
        return
    
    # 2. Предобработка спектров
    X_spectra = preprocess_spectra(spectra_list)
    
    # 3. Подготовка меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    tree_types = label_encoder.classes_
    
    print(f"\n📊 ФИНАЛЬНЫЕ ДАННЫЕ:")
    print(f"   🔢 Форма данных: {X_spectra.shape}")
    print(f"   🏷️  Количество классов: {len(tree_types)}")
    print(f"   📋 Виды: {list(tree_types)}")
    
    # 4. Разделение на обучающую и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_spectra, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✂️ РАЗДЕЛЕНИЕ НА TRAIN/TEST:")
    print(f"   📊 Train: {X_train.shape[0]} образцов")
    print(f"   📊 Test: {X_test.shape[0]} образцов")
    
    # 5. Нормализация данных
    print("\n⚖️ НОРМАЛИЗАЦИЯ ДАННЫХ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Преобразование в формат для CNN
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    print(f"   📊 Форма данных для CNN: {X_train_cnn.shape}")
    
    # 7. One-hot encoding для меток
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(tree_types))
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(tree_types))
    
    # 8. Создание модели
    model = create_1d_alexnet_model(
        input_shape=(X_train_cnn.shape[1], 1),
        num_classes=len(tree_types)
    )
    
    # 9. Обучение модели
    print("\n🎓 ОБУЧЕНИЕ РЕАЛИСТИЧНОЙ МОДЕЛИ...")
    
    # Параметры обучения для более реалистичных результатов
    batch_size = 16  # Меньший batch size для лучшего обучения
    epochs = 150     # Больше эпох для лучшего обучения
    validation_split = 0.2
    
    # Добавляем callbacks для лучшего обучения
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
    )
    
    history = model.fit(
        X_train_cnn, y_train_onehot,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 10. Оценка на чистых данных
    print("\n📊 ОЦЕНКА НА ЧИСТЫХ ДАННЫХ...")
    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_onehot, verbose=0)
    print(f"Точность на чистых данных: {test_accuracy:.7f}")
    
    # 11. Оценка с шумом
    results, confusion_matrices = evaluate_with_noise(
        model, X_test_cnn, y_test_onehot, tree_types, noise_levels=[1, 5, 10]
    )
    
    # 12. Сохранение параметров сети
    save_network_params(model, tree_types, timestamp)
    
    # 13. Сохранение модели
    model.save(f'1d_alexnet_spring_7_species_{timestamp}.h5')
    print(f"\n💾 Модель сохранена: 1d_alexnet_spring_7_species_{timestamp}.h5")
    
    # 14. График обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Точность обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 15. Итоговые результаты
    print(f"\n🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ 1D-AlexNet:")
    print(f"   📊 Чистые данные: {test_accuracy:.7f}")
    for noise_level, accuracy in results.items():
        print(f"   📊 {noise_level}% шума: {accuracy:.7f}")
    
    print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    print(f"📁 Созданные файлы:")
    print(f"   🌳 Модель: 1d_alexnet_spring_7_species_{timestamp}.h5")
    print(f"   ⚙️ Параметры: network_params_{timestamp}.json")
    print(f"   📈 График обучения: training_history_{timestamp}.png")
    print(f"   📊 Матрицы ошибок:")
    for noise_level in [1, 5, 10]:
        print(f"     📊 {noise_level}% шума: confusion_matrix_{noise_level}percent.png")
        print(f"     📊 {noise_level}% шума (норм.): confusion_matrix_{noise_level}percent_normalized.png")

if __name__ == "__main__":
    main() 