import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
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

def create_improved_vandy_alexnet_model(input_shape, num_classes):
    """Создает улучшенную модель Vandy AlexNet для достижения 80% точности"""
    model = keras.Sequential([
        # Первый сверточный блок - увеличенное количество фильтров
        layers.Conv1D(128, 11, strides=1, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling1D(3, strides=2),
        
        # Второй сверточный блок - улучшенная архитектура
        layers.Conv1D(384, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling1D(3, strides=2),
        
        # Третий сверточный блок - больше слоев
        layers.Conv1D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(384, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling1D(3, strides=2),
        
        # Четвертый сверточный блок - дополнительный слой
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling1D(3, strides=2),
        
        # Полносвязные слои - оптимизированные
        layers.Flatten(),
        layers.Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def add_data_augmentation(X, y, noise_level=0.01, num_augmented=2):
    """Добавляет аугментацию данных для увеличения вариативности"""
    print(f"Добавление аугментации данных (шум {noise_level*100}%, {num_augmented} копий)...")
    
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Оригинальные данные
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Аугментированные данные
        for j in range(num_augmented):
            noise = np.random.normal(0, noise_level, X[i].shape)
            augmented_sample = X[i] + noise
            # Ограничиваем значения в диапазоне [0, 1]
            augmented_sample = np.clip(augmented_sample, 0, 1)
            
            augmented_X.append(augmented_sample)
            augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def cross_validate_improved_model(X, y, label_encoder, n_splits=5):
    """Проводит кросс-валидацию улучшенной модели"""
    print(f"\nКРОСС-ВАЛИДАЦИЯ УЛУЧШЕННОЙ МОДЕЛИ ({n_splits} фолдов):")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nФолд {fold + 1}:")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Добавляем аугментацию к обучающим данным
        X_train_aug, y_train_aug = add_data_augmentation(X_train_fold, y_train_fold, noise_level=0.005, num_augmented=1)
        
        # Масштабируем данные
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Изменяем форму для CNN
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # Создаем и обучаем улучшенную модель
        model = create_improved_vandy_alexnet_model(
            input_shape=(X_train_reshaped.shape[1], 1),
            num_classes=len(label_encoder.classes_)
        )
        
        # Используем legacy optimizer для лучшей производительности на M1/M2
        optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Обучаем с улучшенными callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-7
        )
        
        history = model.fit(
            X_train_reshaped, y_train_aug,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Оцениваем
        y_pred = np.argmax(model.predict(X_test_reshaped, verbose=0), axis=1)
        accuracy = accuracy_score(y_test_fold, y_pred)
        accuracies.append(accuracy)
        
        print(f"  Точность: {accuracy:.4f}")
        print(f"  Эпох обучения: {len(history.history['loss'])}")
        print(f"  Размер обучающей выборки: {len(X_train_aug)} (с аугментацией)")
    
    print(f"\nРезультаты кросс-валидации:")
    print(f"  Средняя точность: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Минимальная точность: {np.min(accuracies):.4f}")
    print(f"  Максимальная точность: {np.max(accuracies):.4f}")
    
    return accuracies

def train_final_improved_model(X, y, label_encoder):
    """Обучает финальную улучшенную модель"""
    print(f"\nОБУЧЕНИЕ ФИНАЛЬНОЙ УЛУЧШЕННОЙ МОДЕЛИ:")
    print("=" * 50)
    
    # Разделяем данные 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Разделение данных:")
    print(f"  Обучающая выборка: {len(X_train)} (80%)")
    print(f"  Тестовая выборка: {len(X_test)} (20%)")
    
    # Добавляем аугментацию к обучающим данным
    X_train_aug, y_train_aug = add_data_augmentation(X_train, y_train, noise_level=0.005, num_augmented=2)
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)
    
    # Изменяем форму для CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Создаем улучшенную модель
    model = create_improved_vandy_alexnet_model(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Компилируем модель
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Архитектура улучшенной модели:")
    model.summary()
    
    # Обучаем модель
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=10,
        min_lr=1e-7
    )
    
    print("\nОбучение модели...")
    history = model.fit(
        X_train_reshaped, y_train_aug,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Оцениваем модель
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
    
    # Создаем матрицу ошибок
    print("\nСоздание матрицы ошибок...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax1)
    ax1.set_title('Матрица ошибок (абсолютные значения)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Предсказанный класс', fontsize=12)
    ax1.set_ylabel('Истинный класс', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax2)
    ax2.set_title('Матрица ошибок (нормализованные значения)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Предсказанный класс', fontsize=12)
    ax2.set_ylabel('Истинный класс', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'improved_vandy_alexnet_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График истории обучения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Обучающая точность', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Валидационная точность', linewidth=2)
    ax1.set_title('Точность модели', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Обучающие потери', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Валидационные потери', linewidth=2)
    ax2.set_title('Потери модели', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'improved_vandy_alexnet_training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем модель
    model.save(f'improved_vandy_alexnet_model_{timestamp}.h5')
    
    import joblib
    joblib.dump(scaler, f'improved_vandy_alexnet_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'improved_vandy_alexnet_label_encoder_{timestamp}.pkl')
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    
    return accuracy, model, scaler, label_encoder

def main():
    """Основная функция для достижения 80% точности"""
    print("УЛУЧШЕННАЯ VANDY ALEXNET ДЛЯ ДОСТИЖЕНИЯ 80% ТОЧНОСТИ")
    print("=" * 60)
    
    # Загружаем данные
    X, y = load_spectral_data_7_species()
    
    print(f"\nЗагружено {len(X)} спектров")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Проводим кросс-валидацию улучшенной модели
    cv_accuracies = cross_validate_improved_model(X, y_encoded, label_encoder)
    
    # Обучаем финальную модель
    final_accuracy, model, scaler, label_encoder = train_final_improved_model(X, y_encoded, label_encoder)
    
    # Итоговые результаты
    print(f"\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 50)
    print(f"Кросс-валидация: {np.mean(cv_accuracies):.1%} ± {np.std(cv_accuracies):.1%}")
    print(f"Финальная точность: {final_accuracy:.1%}")
    
    if final_accuracy >= 0.80:
        print("✅ Цель достигнута! Точность ≥ 80%")
    else:
        print("⚠️ Цель не достигнута. Требуется дополнительная оптимизация.")
    
    # Открываем результаты
    import subprocess
    subprocess.run(['open', f'improved_vandy_alexnet_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'])
    subprocess.run(['open', f'improved_vandy_alexnet_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'])

if __name__ == "__main__":
    main() 