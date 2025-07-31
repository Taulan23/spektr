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
    """Загружает данные для 7 видов деревьев с улучшенной обработкой"""
    data = []
    labels = []
    
    # Список 7 видов
    species = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    
    for species_name in species:
        print(f"Загрузка данных для {species_name}...")
        
        folder_path = f'Спектры, весенний период, 7 видов/{species_name}'
        files = glob.glob(f'{folder_path}/*.xlsx')
        
        # Берем больше файлов для лучшего обучения
        for file in files[:50]:  # Увеличили с 30 до 50
            try:
                df = pd.read_excel(file)
                # Улучшенная обработка спектральных данных
                spectral_data = df.iloc[:, 1:].values.flatten()
                
                # Проверяем качество данных
                if len(spectral_data) > 0 and not np.any(np.isnan(spectral_data)):
                    # Нормализуем данные в диапазоне [0, 1]
                    spectral_data = (spectral_data - np.min(spectral_data)) / (np.max(spectral_data) - np.min(spectral_data))
                    data.append(spectral_data)
                    labels.append(species_name)
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def create_improved_1d_cnn(input_shape, num_classes):
    """Создает улучшенную 1D CNN модель с регуляризацией"""
    model = keras.Sequential([
        # Первый сверточный блок
        layers.Conv1D(64, 7, strides=2, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(3, strides=2),
        
        # Второй сверточный блок
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(3, strides=2),
        
        # Третий сверточный блок
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(3, strides=2),
        
        # Полносвязные слои с регуляризацией
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Создает матрицу ошибок с улучшенной визуализацией"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    
    # Нормализованная матрица
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем подграфики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Матрица ошибок (абсолютные значения)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Предсказанный класс', fontsize=12)
    ax1.set_ylabel('Истинный класс', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Матрица ошибок (нормализованные значения)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Предсказанный класс', fontsize=12)
    ax2.set_ylabel('Истинный класс', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, filename):
    """Создает график истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Обучающая точность', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Валидационная точность', linewidth=2)
    ax1.set_title('Точность модели', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График потерь
    ax2.plot(history.history['loss'], label='Обучающие потери', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Валидационные потери', linewidth=2)
    ax2.set_title('Потери модели', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_class_performance(y_true, y_pred, class_names):
    """Анализирует производительность по классам"""
    print("\nДетальный анализ по классам:")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        class_correct = np.sum((y_true == i) & (y_pred == i))
        class_total = np.sum(class_mask)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        
        print(f"{class_name}:")
        print(f"  Правильно классифицировано: {class_correct}/{class_total}")
        print(f"  Точность: {class_accuracy:.3f}")
        
        # Показываем ошибки
        if class_total > class_correct:
            wrong_predictions = y_pred[class_mask & (y_pred != i)]
            if len(wrong_predictions) > 0:
                unique_wrong, counts = np.unique(wrong_predictions, return_counts=True)
                print(f"  Ошибки классификации:")
                for wrong_class, count in zip(unique_wrong, counts):
                    print(f"    -> {class_names[wrong_class]}: {count} раз")
        print()

def main():
    """Основная функция анализа с улучшениями"""
    print("Загрузка данных для 7 видов деревьев (улучшенная версия)...")
    X, y = load_spectral_data_7_species()
    
    print(f"Загружено {len(X)} спектров")
    print(f"Распределение классов:")
    unique, counts = np.unique(y, return_counts=True)
    for species, count in zip(unique, counts):
        print(f"  {species}: {count}")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные (80% на обучение)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Изменяем форму данных для 1D свертки
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Создаем улучшенную модель
    print("Создание улучшенной модели 1D-CNN...")
    model = create_improved_1d_cnn(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Компилируем модель с улучшенными параметрами
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Архитектура модели:")
    model.summary()
    
    # Обучаем модель с ранней остановкой
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    print("Обучение модели...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,  # Увеличили количество эпох
        batch_size=16,  # Уменьшили размер батча
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Оцениваем модель на тестовых данных
    print("\nРезультаты классификации:")
    print("-" * 50)
    
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Точность
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на тестовых данных: {accuracy:.4f}")
    
    # Отчет о классификации
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Детальный анализ по классам
    analyze_class_performance(y_test, y_pred, label_encoder.classes_)
    
    # Создаем матрицу ошибок
    print("Создание матрицы ошибок...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_confusion_matrix(
        y_test, y_pred, 
        label_encoder.classes_,
        f"Матрица ошибок улучшенной 1D-CNN (7 видов)",
        f"alexnet_7_species_improved_confusion_matrix_{timestamp}.png"
    )
    
    # График истории обучения
    plot_training_history(
        history,
        f"alexnet_7_species_improved_training_history_{timestamp}.png"
    )
    
    # Сохраняем модель и препроцессоры
    model.save(f'alexnet_7_species_improved_model_{timestamp}.h5')
    
    import joblib
    joblib.dump(scaler, f'alexnet_7_species_improved_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'alexnet_7_species_improved_label_encoder_{timestamp}.pkl')
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    print("\nУлучшенный анализ 1D-CNN для 7 видов завершен!")

if __name__ == "__main__":
    main() 