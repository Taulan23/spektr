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

def plot_training_history(history, filename):
    """Создает график истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Обучающая точность')
    ax1.plot(history.history['val_accuracy'], label='Валидационная точность')
    ax1.set_title('Точность модели', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Точность', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График потерь
    ax2.plot(history.history['loss'], label='Обучающие потери')
    ax2.plot(history.history['val_loss'], label='Валидационные потери')
    ax2.set_title('Потери модели', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Потери', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_parameters_to_file(filename):
    """Сохраняет параметры модели в текстовый файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ПАРАМЕТРЫ МОДЕЛИ 1D-ALEXNET ДЛЯ 7 ВИДОВ (ДИССЕРТАЦИЯ)\n")
        f.write("=" * 60 + "\n\n")
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
        f.write("-" * 30 + "\n")
        f.write("Оптимизатор: Adam\n")
        f.write("Learning Rate: 0.001\n")
        f.write("Loss: categorical_crossentropy\n")
        f.write("Эпохи: 100\n")
        f.write("Batch Size: 32\n")
        f.write("Validation Split: 0.2\n")
        f.write("Аугментация шума: НЕТ\n\n")
        
        f.write("ПАРАМЕТРЫ ДАННЫХ:\n")
        f.write("-" * 30 + "\n")
        f.write("Количество видов: 7\n")
        f.write("Виды: береза, дуб, ель, клен, липа, осина, сосна\n")
        f.write("Файлов на вид: 30\n")
        f.write("Разделение данных: 80% обучение, 20% тест\n")
        f.write("Стратификация: Да\n")
        f.write("Предобработка: StandardScaler\n")
        f.write("Аугментация данных: НЕТ\n\n")
        
        f.write("ВОСПРОИЗВОДИМОСТЬ:\n")
        f.write("-" * 30 + "\n")
        f.write("np.random.seed(42)\n")
        f.write("tf.random.set_seed(42)\n")
        f.write("random_state=42 в train_test_split\n\n")
        
        f.write("ОСОБЕННОСТИ ДЛЯ ДИССЕРТАЦИИ:\n")
        f.write("-" * 30 + "\n")
        f.write("- Без аугментации шума\n")
        f.write("- Чистые данные без модификации\n")
        f.write("- Реалистичные условия классификации\n")

def main():
    """Основная функция для диссертации"""
    print("КЛАССИФИКАЦИЯ 7 ВИДОВ ДЕРЕВЬЕВ - 1D-ALEXNET (ДИССЕРТАЦИЯ)")
    print("=" * 70)
    print("БЕЗ АУГМЕНТАЦИИ ШУМА - ЧИСТАЯ МОДЕЛЬ")
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
    print("\nОбучение модели (без аугментации)...")
    history = model.fit(
        X_train_cnn, y_train_onehot,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Тестирование на чистых данных
    print("\nТестирование на чистых данных...")
    
    # Предсказание
    y_pred_proba = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Точность на тестовых данных: {accuracy:.4f}")
    
    # Создание матриц ошибок
    title = f'Матрица ошибок - Чистые данные (точность: {accuracy:.4f})'
    filename = f'alexnet_7_species_dissertation_confusion_matrix.png'
    plot_confusion_matrix(y_test, y_pred, class_names, title, filename)
    print(f"Матрица сохранена: {filename}")
    
    # Нормализованная матрица
    title_norm = f'Нормализованная матрица ошибок - Чистые данные'
    filename_norm = f'alexnet_7_species_dissertation_normalized_confusion_matrix.png'
    plot_normalized_confusion_matrix(y_test, y_pred, class_names, title_norm, filename_norm)
    print(f"Нормализованная матрица сохранена: {filename_norm}")
    
    # График обучения
    history_filename = f'alexnet_7_species_dissertation_training_history.png'
    plot_training_history(history, history_filename)
    print(f"График обучения сохранен: {history_filename}")
    
    # Сохранение параметров
    save_parameters_to_file('parameters_7_species_dissertation.txt')
    print("\nПараметры сохранены: parameters_7_species_dissertation.txt")
    
    # Детальный отчет по классам
    print("\nДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    print("-" * 40)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    for class_name in class_names:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        print(f"{class_name}:")
        print(f"  Точность: {precision:.3f}")
        print(f"  Полнота: {recall:.3f}")
        print(f"  F1-мера: {f1:.3f}")
        print()
    
    print(f"ОБЩАЯ ТОЧНОСТЬ: {accuracy:.4f}")
    
    print(f"\nФайлы созданы:")
    print(f"- {filename}")
    print(f"- {filename_norm}")
    print(f"- {history_filename}")
    print(f"- parameters_7_species_dissertation.txt")

if __name__ == "__main__":
    main() 