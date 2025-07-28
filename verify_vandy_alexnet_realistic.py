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
    """Загружает данные для 7 видов деревьев (все доступные файлы)"""
    data = []
    labels = []
    
    species_folders = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    
    for species in species_folders:
        print(f"Загрузка данных для {species}...")
        folder_path = f'Спектры, весенний период, 7 видов/{species}'
        
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена, пропускаем...")
            continue
            
        files = glob.glob(f'{folder_path}/*.xlsx')
        print(f"  Найдено {len(files)} файлов")
        
        # Загружаем ВСЕ файлы для каждого вида
        for file in files:
            try:
                df = pd.read_excel(file)
                spectral_data = df.iloc[:, 1:].values.flatten()
                
                if len(spectral_data) > 0 and not np.any(np.isnan(spectral_data)):
                    # Нормализация к [0,1]
                    spectral_data = (spectral_data - np.min(spectral_data)) / (np.max(spectral_data) - np.min(spectral_data))
                    data.append(spectral_data)
                    labels.append(species)
            except Exception as e:
                print(f"  Ошибка при чтении файла {file}: {e}")
                continue
    
    return np.array(data), np.array(labels)

def create_vandy_alexnet_model(input_shape, num_classes):
    """Создает оригинальную модель Vandy AlexNet"""
    model = keras.Sequential([
        # Первый сверточный блок
        layers.Conv1D(96, 11, strides=1, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3, strides=2),
        
        # Второй сверточный блок
        layers.Conv1D(256, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3, strides=2),
        
        # Третий сверточный блок
        layers.Conv1D(384, 3, padding='same', activation='relu'),
        layers.Conv1D(384, 3, padding='same', activation='relu'),
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

def analyze_data_quality(X, y, label_encoder):
    """Анализирует качество данных"""
    print("\nАНАЛИЗ КАЧЕСТВА ДАННЫХ:")
    print("=" * 50)
    
    # Анализ спектральных данных
    print(f"Размер данных: {X.shape}")
    print(f"Количество классов: {len(label_encoder.classes_)}")
    
    # Проверяем уникальность спектров
    unique_spectra = np.unique(X, axis=0)
    print(f"Уникальных спектров: {len(unique_spectra)} из {len(X)}")
    
    # Анализ по классам
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = (y == i)
        class_data = X[class_mask]
        
        # Проверяем уникальность внутри класса
        unique_in_class = np.unique(class_data, axis=0)
        print(f"{class_name}: {len(unique_in_class)} уникальных из {len(class_data)}")
        
        # Проверяем стандартное отклонение
        std_dev = np.std(class_data, axis=0)
        print(f"  Среднее стд. отклонение: {np.mean(std_dev):.6f}")
        
        # Проверяем корреляцию между спектрами
        if len(class_data) > 1:
            correlations = []
            for j in range(len(class_data)):
                for k in range(j+1, len(class_data)):
                    corr = np.corrcoef(class_data[j], class_data[k])[0,1]
                    correlations.append(corr)
            print(f"  Средняя корреляция между спектрами: {np.mean(correlations):.4f}")

def cross_validate_model(X, y, label_encoder, n_splits=5):
    """Проводит кросс-валидацию для проверки стабильности результатов"""
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\nКРОСС-ВАЛИДАЦИЯ ({n_splits} фолдов):")
    print("=" * 50)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nФолд {fold + 1}:")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Масштабируем данные
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Изменяем форму для CNN
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # Создаем и обучаем модель
        model = create_vandy_alexnet_model(
            input_shape=(X_train_reshaped.shape[1], 1),
            num_classes=len(label_encoder.classes_)
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Обучаем с ранней остановкой
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_reshaped, y_train_fold,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Оцениваем
        y_pred = np.argmax(model.predict(X_test_reshaped, verbose=0), axis=1)
        accuracy = accuracy_score(y_test_fold, y_pred)
        accuracies.append(accuracy)
        
        print(f"  Точность: {accuracy:.4f}")
        print(f"  Эпох обучения: {len(history.history['loss'])}")
    
    print(f"\nРезультаты кросс-валидации:")
    print(f"  Средняя точность: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Минимальная точность: {np.min(accuracies):.4f}")
    print(f"  Максимальная точность: {np.max(accuracies):.4f}")
    
    return accuracies

def main():
    """Основная функция для проверки реалистичности результатов"""
    print("ПРОВЕРКА РЕАЛИСТИЧНОСТИ РЕЗУЛЬТАТОВ VANDY ALEXNET")
    print("=" * 60)
    
    # Загружаем данные
    X, y = load_spectral_data_7_species()
    
    print(f"\nЗагружено {len(X)} спектров")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Анализируем качество данных
    analyze_data_quality(X, y_encoded, label_encoder)
    
    # Проводим кросс-валидацию
    cv_accuracies = cross_validate_model(X, y_encoded, label_encoder)
    
    # Проверяем, реалистичны ли результаты
    mean_accuracy = np.mean(cv_accuracies)
    
    print(f"\nОЦЕНКА РЕАЛИСТИЧНОСТИ:")
    print("=" * 50)
    
    if mean_accuracy > 0.95:
        print("⚠️  ВНИМАНИЕ: Очень высокая точность (>95%) может указывать на:")
        print("   - Переобучение модели")
        print("   - Проблемы с разделением данных")
        print("   - Дублирование или очень похожие спектры")
        print("   - Недостаточную сложность задачи")
    elif mean_accuracy > 0.85:
        print("✅ Результаты выглядят реалистично (85-95%)")
    elif mean_accuracy > 0.75:
        print("✅ Результаты выглядят реалистично (75-85%)")
    else:
        print("⚠️  Низкая точность может указывать на проблемы с данными или моделью")
    
    print(f"\nРекомендация: {mean_accuracy:.1%} точность {'реалистична' if 0.75 <= mean_accuracy <= 0.95 else 'требует проверки'}")

if __name__ == "__main__":
    main() 