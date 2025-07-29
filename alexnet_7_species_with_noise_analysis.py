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
from datetime import datetime

def load_spectral_data_7_species():
    """Загружает данные для 7 видов деревьев"""
    data = []
    labels = []
    
    # Список 7 видов
    species = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    
    for species_name in species:
        print(f"Загрузка данных для {species_name}...")
        
        folder_path = f'Спектры, весенний период, 7 видов/{species_name}'
        files = glob.glob(f'{folder_path}/*.xlsx')
        
        # Берем первые 30 файлов для каждого вида
        for file in files[:30]:
            try:
                df = pd.read_excel(file)
                # Предполагаем, что спектральные данные начинаются со второй колонки
                spectral_data = df.iloc[:, 1:].values.flatten()
                if len(spectral_data) > 0:
                    data.append(spectral_data)
                    labels.append(species_name)
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def create_1d_alexnet_model(input_shape, num_classes):
    """Создает 1D версию AlexNet"""
    model = keras.Sequential([
        # Первый сверточный блок
        layers.Conv1D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3, strides=2),
        
        # Второй сверточный блок
        layers.Conv1D(256, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3, strides=2),
        
        # Третий сверточный блок
        layers.Conv1D(384, 3, padding='same', activation='relu'),
        
        # Четвертый сверточный блок
        layers.Conv1D(384, 3, padding='same', activation='relu'),
        
        # Пятый сверточный блок
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(3, strides=2),
        
        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def add_noise(X, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return X
    
    # Шум как процент от стандартного отклонения данных
    noise_std = noise_level / 100.0 * np.std(X)
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Создает матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalized_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Создает нормализованную матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Нормализация по строкам (каждый класс)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_parameters_to_file(filename):
    """Сохраняет параметры модели в текстовый файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ПАРАМЕТРЫ МОДЕЛИ 1D-ALEXNET ДЛЯ 7 ВИДОВ\n")
        f.write("=" * 50 + "\n\n")
        f.write("Модель: 1D-AlexNet (CNN)\n")
        f.write("Архитектура:\n")
        f.write("- Conv1D(96, 11, strides=4) + BatchNorm + MaxPool(3,2)\n")
        f.write("- Conv1D(256, 5, padding='same') + BatchNorm + MaxPool(3,2)\n")
        f.write("- Conv1D(384, 3, padding='same')\n")
        f.write("- Conv1D(384, 3, padding='same')\n")
        f.write("- Conv1D(256, 3, padding='same') + MaxPool(3,2)\n")
        f.write("- Dense(4096) + Dropout(0.5)\n")
        f.write("- Dense(4096) + Dropout(0.5)\n")
        f.write("- Dense(num_classes, softmax)\n\n")
        
        f.write("ПАРАМЕТРЫ ОБУЧЕНИЯ:\n")
        f.write("-" * 20 + "\n")
        f.write("Оптимизатор: Adam\n")
        f.write("Learning Rate: 0.001\n")
        f.write("Loss: categorical_crossentropy\n")
        f.write("Эпохи: 100\n")
        f.write("Batch Size: 32\n")
        f.write("Validation Split: 0.2\n\n")
        
        f.write("ПАРАМЕТРЫ ДАННЫХ:\n")
        f.write("-" * 20 + "\n")
        f.write("Количество видов: 7\n")
        f.write("Виды: береза, дуб, ель, клен, липа, осина, сосна\n")
        f.write("Файлов на вид: 30\n")
        f.write("Разделение данных: 80% обучение, 20% тест\n")
        f.write("Стратификация: Да\n")
        f.write("Предобработка: StandardScaler\n\n")
        
        f.write("ПАРАМЕТРЫ ШУМА:\n")
        f.write("-" * 20 + "\n")
        f.write("Тип шума: Аддитивный гауссовский\n")
        f.write("Среднее: 0\n")
        f.write("Стандартное отклонение: процент от std данных\n")
        f.write("Уровни шума: 0%, 1%, 5%, 10%\n\n")
        
        f.write("ВОСПРОИЗВОДИМОСТЬ:\n")
        f.write("-" * 20 + "\n")
        f.write("np.random.seed(42)\n")
        f.write("tf.random.set_seed(42)\n")
        f.write("random_state=42 в train_test_split\n")

def main():
    """Основная функция"""
    print("КЛАССИФИКАЦИЯ 7 ВИДОВ ДЕРЕВЬЕВ - 1D-ALEXNET С АНАЛИЗОМ ШУМА")
    print("=" * 70)
    
    # Установка seed для воспроизводимости
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Загрузка данных
    X, y = load_spectral_data_7_species()
    
    if len(X) == 0:
        print("Не удалось загрузить данные!")
        return
    
    print(f"Загружено {len(X)} спектров для {len(np.unique(y))} видов")
    
    # Предобработка
    lengths = [len(s) for s in X]
    target_length = min(lengths)
    X_processed = np.array([spectrum[:target_length] for spectrum in X], dtype=np.float32)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"Форма данных: {X_processed.shape}")
    print(f"Классы: {class_names}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Преобразование в формат для CNN
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # One-hot encoding для меток
    y_train_onehot = keras.utils.to_categorical(y_train)
    y_test_onehot = keras.utils.to_categorical(y_test)
    
    # Создание модели
    model = create_1d_alexnet_model((X_train_cnn.shape[1], 1), len(class_names))
    
    # Компиляция модели
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nАрхитектура модели:")
    model.summary()
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
    
    # Обучение модели
    print("\nОбучение модели...")
    history = model.fit(
        X_train_cnn, y_train_onehot,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Тестирование на разных уровнях шума
    noise_levels = [0, 1, 5, 10]
    results = {}
    
    for noise_level in noise_levels:
        print(f"\nТестирование с {noise_level}% шумом...")
        
        # Добавление шума к тестовым данным
        X_test_noisy = add_noise(X_test_scaled, noise_level)
        X_test_noisy_cnn = X_test_noisy.reshape(X_test_noisy.shape[0], X_test_noisy.shape[1], 1)
        
        # Предсказание
        y_pred_proba = model.predict(X_test_noisy_cnn)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[noise_level] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true': y_test
        }
        
        print(f"Точность: {accuracy:.4f}")
        
        # Создание матриц ошибок
        if noise_level in [0, 1, 5, 10]:
            # Обычная матрица
            title = f'Матрица ошибок - {noise_level}% шум (точность: {accuracy:.4f})'
            filename = f'alexnet_7_species_confusion_matrix_{noise_level}percent.png'
            plot_confusion_matrix(y_test, y_pred, class_names, title, filename)
            print(f"Матрица сохранена: {filename}")
            
            # Нормализованная матрица
            title_norm = f'Нормализованная матрица ошибок - {noise_level}% шум'
            filename_norm = f'alexnet_7_species_normalized_confusion_matrix_{noise_level}percent.png'
            plot_normalized_confusion_matrix(y_test, y_pred, class_names, title_norm, filename_norm)
            print(f"Нормализованная матрица сохранена: {filename_norm}")
    
    # Сохранение параметров
    save_parameters_to_file('parameters_7_species_alexnet.txt')
    print("\nПараметры сохранены: parameters_7_species_alexnet.txt")
    
    # Итоговый отчет
    print("\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("-" * 30)
    for noise_level in noise_levels:
        acc = results[noise_level]['accuracy']
        print(f"{noise_level}% шум: {acc:.4f}")
    
    print(f"\nФайлы созданы:")
    for noise_level in [0, 1, 5, 10]:
        print(f"- alexnet_7_species_confusion_matrix_{noise_level}percent.png")
        print(f"- alexnet_7_species_normalized_confusion_matrix_{noise_level}percent.png")
    print("- parameters_7_species_alexnet.txt")

if __name__ == "__main__":
    main() 