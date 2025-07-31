import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
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
    """Добавляет гауссов шум к данным"""
    if noise_level == 0:
        return X
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def evaluate_model_with_noise(model, X_test, y_test, noise_levels):
    """Оценивает модель на разных уровнях шума"""
    results = {}
    
    for noise_level in noise_levels:
        X_test_noisy = add_noise(X_test, noise_level)
        y_pred = model.predict(X_test_noisy, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        results[noise_level] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'predicted_classes': y_pred_classes
        }
        
        print(f"Шум {noise_level*100:.0f}%: Точность = {accuracy:.4f}")
    
    return results

def create_confusion_matrices(results, y_test, label_encoder, noise_levels):
    """Создает матрицы ошибок для всех уровней шума"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, noise_level in enumerate(noise_levels):
        cm = confusion_matrix(y_test, results[noise_level]['predicted_classes'])
        
        # Нормализуем матрицу
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[i])
        axes[i].set_title(f'Матрица ошибок (шум {noise_level*100:.0f}%)')
        axes[i].set_xlabel('Предсказанный класс')
        axes[i].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'alexnet_7_species_confusion_matrices_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_7_species_classification():
    """Основная функция анализа"""
    print("Загрузка данных для 7 видов деревьев...")
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
    
    # Создаем модель
    print("Создание модели 1D-AlexNet...")
    model = create_1d_alexnet_model(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Компилируем модель
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Архитектура модели:")
    model.summary()
    
    # Обучаем модель
    print("Обучение модели...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Тестируем на разных уровнях шума
    noise_levels = [0, 0.01, 0.05, 0.10]  # 0%, 1%, 5%, 10%
    
    print("\nРезультаты классификации:")
    print("-" * 50)
    
    results = evaluate_model_with_noise(model, X_test_reshaped, y_test, noise_levels)
    
    # Создаем матрицы ошибок
    print("\nСоздание матриц ошибок...")
    create_confusion_matrices(results, y_test, label_encoder, noise_levels)
    
    # Сохраняем модель и препроцессоры
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model.save(f'alexnet_7_species_model_{timestamp}.h5')
    
    import joblib
    joblib.dump(scaler, f'alexnet_7_species_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'alexnet_7_species_label_encoder_{timestamp}.pkl')
    
    # Сохраняем результаты
    results_summary = {
        'noise_levels': noise_levels,
        'accuracies': [results[noise]['accuracy'] for noise in noise_levels],
        'model_architecture': model.get_config(),
        'training_history': history.history
    }
    
    joblib.dump(results_summary, f'alexnet_7_species_results_{timestamp}.pkl')
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    
    # Создаем график обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучающая точность')
    plt.plot(history.history['val_accuracy'], label='Валидационная точность')
    plt.title('Точность модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучающая ошибка')
    plt.plot(history.history['val_loss'], label='Валидационная ошибка')
    plt.title('Ошибка модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'alexnet_7_species_training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, scaler, label_encoder, results

if __name__ == "__main__":
    model, scaler, label_encoder, results = analyze_7_species_classification()
    print(f"\nАнализ 1D-AlexNet для 7 видов завершен!") 