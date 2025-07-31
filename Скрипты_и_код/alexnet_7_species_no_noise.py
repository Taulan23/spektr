import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Установка русского шрифта
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']

def load_spectral_data_7_species():
    """Загрузка данных для 7 видов деревьев без аугментации шума"""
    print("Загрузка данных для 7 видов деревьев (без аугментации шума)...")
    
    species_folders = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    data = []
    labels = []
    
    for species in species_folders:
        print(f"Загрузка данных для {species}...")
        folder_path = f'Спектры, весенний период, 7 видов/{species}'
        
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена, пропускаем...")
            continue
            
        files = glob.glob(f'{folder_path}/*.xlsx')
        files = files[:30]  # Берем первые 30 файлов
        
        for file in files:
            try:
                df = pd.read_excel(file)
                # Предполагаем, что спектральные данные находятся в определенных столбцах
                spectral_data = df.iloc[:, 1:].values.flatten()  # Исключаем первый столбец (длины волн)
                
                if len(spectral_data) > 0:
                    data.append(spectral_data)
                    labels.append(species)
            except Exception as e:
                print(f"Ошибка при чтении файла {file}: {e}")
                continue
    
    print(f"Загружено {len(data)} спектров")
    
    # Выводим распределение классов
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Распределение классов:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    return np.array(data), np.array(labels)

def create_1d_alexnet_model(input_shape, num_classes):
    """Создание модели 1D-AlexNet"""
    model = keras.Sequential([
        layers.Conv1D(96, 11, strides=1, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3, strides=2),
        
        layers.Conv1D(256, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(3, strides=2),
        
        layers.Conv1D(384, 3, padding='same', activation='relu'),
        layers.Conv1D(384, 3, padding='same', activation='relu'),
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(3, strides=2),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Создание матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, filename):
    """Создание графика истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Обучающая точность')
    ax1.plot(history.history['val_accuracy'], label='Валидационная точность')
    ax1.set_title('Точность модели', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График потерь
    ax2.plot(history.history['loss'], label='Обучающие потери')
    ax2.plot(history.history['val_loss'], label='Валидационные потери')
    ax2.set_title('Потери модели', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Загрузка данных
    X, y = load_spectral_data_7_species()
    
    if len(X) == 0:
        print("Не удалось загрузить данные!")
        return
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Стандартизация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Изменение формы для CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    print(f"Форма обучающих данных: {X_train_reshaped.shape}")
    print(f"Форма тестовых данных: {X_test_reshaped.shape}")
    
    # Создание модели
    print("Создание модели 1D-AlexNet...")
    model = create_1d_alexnet_model(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Обучение модели
    print("Обучение модели...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Оценка модели на тестовых данных
    print("\nРезультаты классификации:")
    print("-" * 50)
    
    # Предсказания на тестовых данных
    y_pred_proba = model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Точность
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на тестовых данных: {accuracy:.4f}")
    
    # Отчет о классификации
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Создание матрицы ошибок
    print("Создание матрицы ошибок...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_confusion_matrix(
        y_test, y_pred, 
        label_encoder.classes_,
        f"Матрица ошибок 1D-AlexNet (7 видов, без аугментации шума)",
        f"alexnet_7_species_no_noise_confusion_matrix_{timestamp}.png"
    )
    
    # График истории обучения
    plot_training_history(
        history,
        f"alexnet_7_species_no_noise_training_history_{timestamp}.png"
    )
    
    # Сохранение модели и препроцессоров
    model.save(f'alexnet_7_species_no_noise_model_{timestamp}.h5')
    
    import joblib
    joblib.dump(scaler, f'alexnet_7_species_no_noise_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'alexnet_7_species_no_noise_label_encoder_{timestamp}.pkl')
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    print("\nАнализ 1D-AlexNet для 7 видов (без аугментации шума) завершен!")

if __name__ == "__main__":
    main() 