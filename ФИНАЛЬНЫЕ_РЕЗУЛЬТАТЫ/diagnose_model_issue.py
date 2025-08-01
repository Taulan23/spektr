import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_spectral_data_7_species():
    """Загрузка данных для 7 видов"""
    base_path = "Исходные_данные/Спектры, весенний период, 7 видов"
    species_folders = ["береза", "дуб", "ель", "клен", "липа", "осина", "сосна"]
    
    all_data = []
    all_labels = []
    
    for species in species_folders:
        species_path = f"{base_path}/{species}"
        try:
            import os
            import glob
            excel_files = glob.glob(f"{species_path}/*_vis.xlsx")
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        spectral_data = df[numeric_cols[0]].values
                        if len(spectral_data) > 0:
                            all_data.append(spectral_data)
                            all_labels.append(species)
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Ошибка при загрузке {species}: {e}")
            continue
    
    if len(all_data) == 0:
        print("Не удалось загрузить данные!")
        return np.array([]), np.array([])
    
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"Загружено {len(X)} образцов для {len(np.unique(y))} видов")
    return X, y

def create_original_1d_alexnet(input_shape, num_classes):
    """Создание ОРИГИНАЛЬНОЙ 1D-AlexNet БЕЗ Dropout"""
    model = Sequential([
        Conv1D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(3, strides=2),
        
        Conv1D(256, 5, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2),
        
        Conv1D(384, 3, padding='same', activation='relu'),
        Conv1D(384, 3, padding='same', activation='relu'),
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def diagnose_model_problem():
    """Диагностика проблемы с моделью"""
    print("=== ДИАГНОСТИКА ПРОБЛЕМЫ С МОДЕЛЬЮ ===")
    
    # Загрузка данных
    print("1. Загрузка данных...")
    data, labels = load_spectral_data_7_species()
    if len(data) == 0:
        return
    
    # Preprocessing
    print("2. Предобработка данных...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество классов: {len(label_encoder.classes_)}")
    print(f"Классы: {label_encoder.classes_}")
    
    # Создание модели
    print("3. Создание модели...")
    model = create_original_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    # Обучение модели
    print("4. Обучение модели...")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Диагностика предсказаний
    print("5. Диагностика предсказаний...")
    y_pred_proba = model.predict(X_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    print(f"\n📊 АНАЛИЗ ПРЕДСКАЗАНИЙ:")
    print(f"Форма предсказаний: {y_pred_proba.shape}")
    print(f"Уникальные значения в предсказаниях: {np.unique(y_pred_proba)}")
    print(f"Среднее по каждому классу:")
    for i, class_name in enumerate(label_encoder.classes_):
        mean_prob = np.mean(y_pred_proba[:, i])
        print(f"  {class_name}: {mean_prob:.6f}")
    
    print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ:")
    print(f"Максимальные вероятности:")
    max_probs = np.max(y_pred_proba, axis=1)
    print(f"  Среднее: {np.mean(max_probs):.6f}")
    print(f"  Стандартное отклонение: {np.std(max_probs):.6f}")
    print(f"  Минимум: {np.min(max_probs):.6f}")
    print(f"  Максимум: {np.max(max_probs):.6f}")
    
    print(f"\n🎯 ПРОБЛЕМА ОБНАРУЖЕНА:")
    if np.std(max_probs) < 1e-6:
        print("❌ ВСЕ ПРЕДСКАЗАНИЯ ОДИНАКОВЫЕ!")
        print("   Это указывает на проблему с архитектурой или обучением")
        print("   Возможные причины:")
        print("   1. Слишком большие полносвязные слои (4096 нейронов)")
        print("   2. Отсутствие регуляризации (нет Dropout)")
        print("   3. Проблема с размером батча")
        print("   4. Переобучение на малом количестве данных")
    
    # Тестирование с разными размерами батча
    print(f"\n🧪 ТЕСТИРОВАНИЕ С РАЗНЫМИ РАЗМЕРАМИ БАТЧА:")
    batch_sizes = [1, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\nРазмер батча: {batch_size}")
        
        # Создаем новую модель
        test_model = create_original_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
        
        # Обучаем с новым размером батча
        test_model.fit(
            X_train_cnn, y_train,
            epochs=5,  # Меньше эпох для быстрого теста
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Тестируем
        test_pred_proba = test_model.predict(X_test_cnn, verbose=0)
        test_max_probs = np.max(test_pred_proba, axis=1)
        test_accuracy = accuracy_score(y_test, np.argmax(test_pred_proba, axis=1))
        
        print(f"  Средняя макс. вероятность: {np.mean(test_max_probs):.6f}")
        print(f"  Стандартное отклонение: {np.std(test_max_probs):.6f}")
        print(f"  Точность: {test_accuracy*100:.2f}%")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print(f"1. Уменьшить размер полносвязных слоев (4096 -> 512)")
    print(f"2. Добавить Dropout для регуляризации")
    print(f"3. Уменьшить размер батча до 8-16")
    print(f"4. Добавить BatchNormalization")
    print(f"5. Использовать меньшую архитектуру для малого количества данных")

if __name__ == "__main__":
    diagnose_model_problem() 