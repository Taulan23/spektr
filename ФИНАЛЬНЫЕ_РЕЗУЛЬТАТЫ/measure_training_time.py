import time
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

def create_1d_alexnet(input_shape, num_classes):
    """Создание 1D-AlexNet"""
    model = Sequential([
        Conv1D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        Conv1D(256, 5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        Conv1D(384, 3, padding='same', activation='relu'),
        Conv1D(384, 3, padding='same', activation='relu'),
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def measure_training_time():
    """Измерение времени обучения оригинальной модели"""
    print("=== ИЗМЕРЕНИЕ ВРЕМЕНИ ОБУЧЕНИЯ ОРИГИНАЛЬНОЙ МОДЕЛИ ===")
    
    # Засекаем общее время
    total_start_time = time.time()
    
    # Загрузка данных
    print("1. Загрузка данных...")
    data_start_time = time.time()
    data, labels = load_spectral_data_7_species()
    if len(data) == 0:
        return
    data_time = time.time() - data_start_time
    print(f"   Время загрузки данных: {data_time:.2f} секунд")
    
    # Preprocessing
    print("2. Предобработка данных...")
    preprocess_start_time = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    preprocess_time = time.time() - preprocess_start_time
    print(f"   Время предобработки: {preprocess_time:.2f} секунд")
    
    # Создание модели
    print("3. Создание модели...")
    model_start_time = time.time()
    model = create_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    model_time = time.time() - model_start_time
    print(f"   Время создания модели: {model_time:.2f} секунд")
    
    # Обучение модели
    print("4. Обучение модели...")
    training_start_time = time.time()
    
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
    training_time = time.time() - training_start_time
    print(f"   Время обучения: {training_time:.2f} секунд")
    
    # Тестирование
    print("5. Тестирование модели...")
    test_start_time = time.time()
    y_pred = model.predict(X_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    test_time = time.time() - test_start_time
    print(f"   Время тестирования: {test_time:.2f} секунд")
    
    # Общее время
    total_time = time.time() - total_start_time
    
    # Результаты
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ИЗМЕРЕНИЯ ВРЕМЕНИ:")
    print("="*60)
    print(f"Загрузка данных:     {data_time:8.2f} сек")
    print(f"Предобработка:       {preprocess_time:8.2f} сек")
    print(f"Создание модели:     {model_time:8.2f} сек")
    print(f"Обучение модели:     {training_time:8.2f} сек")
    print(f"Тестирование:        {test_time:8.2f} сек")
    print("-" * 60)
    print(f"ОБЩЕЕ ВРЕМЯ:         {total_time:8.2f} сек")
    print(f"Точность модели:     {accuracy*100:8.2f}%")
    print("="*60)
    
    # Дополнительная информация
    print(f"\n📈 ДЕТАЛИ:")
    print(f"Размер обучающей выборки: {len(X_train)} образцов")
    print(f"Размер тестовой выборки:  {len(X_test)} образцов")
    print(f"Количество эпох:          {len(history.history['accuracy'])}")
    print(f"Финальная точность:       {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Финальная валидационная:  {history.history['val_accuracy'][-1]*100:.2f}%")
    
    # Сохранение результатов
    results = {
        'Операция': ['Загрузка данных', 'Предобработка', 'Создание модели', 'Обучение', 'Тестирование', 'ОБЩЕЕ'],
        'Время (сек)': [data_time, preprocess_time, model_time, training_time, test_time, total_time]
    }
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/время_обучения_модели.csv', index=False)
    print(f"\n✅ Результаты сохранены в: ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/время_обучения_модели.csv")

if __name__ == "__main__":
    measure_training_time() 