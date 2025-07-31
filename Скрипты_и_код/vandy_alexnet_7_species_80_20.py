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

def add_noise(data, noise_level):
    """Добавляет шум к данным"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Создает матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Нормализованная матрица
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем подграфики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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

def test_with_specific_noise(model, X_test, y_test, class_names, noise_level=0.011):
    """Тестирует модель с конкретным уровнем шума (1.1%)"""
    print(f"\nТЕСТИРОВАНИЕ С ШУМОМ {noise_level*100}%:")
    print("=" * 50)
    
    # Добавляем шум к тестовым данным
    X_test_noisy = add_noise(X_test, noise_level)
    
    # Получаем предсказания
    probabilities = model.predict(X_test_noisy, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    
    # Точность
    accuracy = accuracy_score(y_test, predictions)
    
    # Анализ вероятностей
    max_probabilities = np.max(probabilities, axis=1)
    mean_confidence = np.mean(max_probabilities)
    std_confidence = np.std(max_probabilities)
    
    print(f"Точность: {accuracy:.4f}")
    print(f"Средняя уверенность: {mean_confidence:.4f} ± {std_confidence:.4f}")
    
    # Детальный анализ по классам
    print("\nДетальный анализ по классам с шумом:")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_correct = np.sum((y_test == i) & (predictions == i))
            class_total = np.sum(class_mask)
            class_accuracy = class_correct / class_total
            class_conf = np.mean(max_probabilities[class_mask])
            
            print(f"{class_name}:")
            print(f"  Правильно классифицировано: {class_correct}/{class_total}")
            print(f"  Точность: {class_accuracy:.3f}")
            print(f"  Средняя уверенность: {class_conf:.3f}")
            
            # Показываем ошибки
            if class_total > class_correct:
                wrong_predictions = predictions[class_mask & (predictions != i)]
                if len(wrong_predictions) > 0:
                    unique_wrong, counts = np.unique(wrong_predictions, return_counts=True)
                    print(f"  Ошибки классификации:")
                    for wrong_class, count in zip(unique_wrong, counts):
                        print(f"    -> {class_names[wrong_class]}: {count} раз")
            print()
    
    return {
        'accuracy': accuracy,
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'predictions': predictions,
        'probabilities': probabilities
    }

def main():
    """Основная функция с оригинальной Vandy AlexNet архитектурой"""
    print("ОБУЧЕНИЕ VANDY ALEXNET С РАЗДЕЛЕНИЕМ 80/20 (7 ВИДОВ)")
    print("=" * 60)
    
    # Загружаем ВСЕ данные
    X, y = load_spectral_data_7_species()
    
    print(f"\nЗагружено {len(X)} спектров")
    print(f"Распределение классов:")
    unique, counts = np.unique(y, return_counts=True)
    for species, count in zip(unique, counts):
        print(f"  {species}: {count}")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {len(X_train)} (80%)")
    print(f"  Тестовая выборка: {len(X_test)} (20%)")
    
    # Проверяем распределение в тестовой выборке
    print(f"\nРаспределение в тестовой выборке:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for i, count in enumerate(counts_test):
        class_name = label_encoder.classes_[i]
        print(f"  {class_name}: {count}")
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Изменяем форму для CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Создаем оригинальную Vandy AlexNet модель
    print("\nСоздание оригинальной Vandy AlexNet модели...")
    model = create_vandy_alexnet_model(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Компилируем модель
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
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7
    )
    
    print("\nОбучение модели...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Оцениваем модель на чистых тестовых данных
    print("\nРезультаты классификации (без шума):")
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
    
    # Тестируем с 1.1% шумом
    noise_results = test_with_specific_noise(model, X_test_reshaped, y_test, label_encoder.classes_, 0.011)
    
    # Создаем матрицу ошибок
    print("\nСоздание матрицы ошибок...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_confusion_matrix(
        y_test, y_pred, 
        label_encoder.classes_,
        f"Матрица ошибок Vandy AlexNet (7 видов, 80/20 разделение)",
        f"vandy_alexnet_7_species_80_20_confusion_matrix_{timestamp}.png"
    )
    
    # График истории обучения
    plot_training_history(
        history,
        f"vandy_alexnet_7_species_80_20_training_history_{timestamp}.png"
    )
    
    # Сохраняем модель и препроцессоры
    model.save(f'vandy_alexnet_7_species_80_20_model_{timestamp}.h5')
    
    import joblib
    joblib.dump(scaler, f'vandy_alexnet_7_species_80_20_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'vandy_alexnet_7_species_80_20_label_encoder_{timestamp}.pkl')
    
    # Сохраняем результаты тестирования с шумом
    noise_df = pd.DataFrame([
        {
            'Уровень шума (%)': '1.1%',
            'Точность': f'{noise_results["accuracy"]:.4f}',
            'Средняя уверенность': f'{noise_results["mean_confidence"]:.4f}',
            'Стд уверенность': f'{noise_results["std_confidence"]:.4f}'
        }
    ])
    
    noise_df.to_csv(f'vandy_alexnet_noise_test_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    print("\nАнализ Vandy AlexNet для 7 видов (80/20 разделение) завершен!")
    
    # Открываем результаты
    import subprocess
    subprocess.run(['open', f'vandy_alexnet_7_species_80_20_confusion_matrix_{timestamp}.png'])
    subprocess.run(['open', f'vandy_alexnet_7_species_80_20_training_history_{timestamp}.png'])

if __name__ == "__main__":
    main() 