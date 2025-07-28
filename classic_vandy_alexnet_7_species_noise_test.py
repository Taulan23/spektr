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

def create_classic_vandy_alexnet_model(input_shape, num_classes):
    """Создает классическую модель Vandy AlexNet"""
    model = keras.Sequential([
        # Первый сверточный блок
        layers.Conv1D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(3, strides=2),
        
        # Второй сверточный блок
        layers.Conv1D(256, 5, padding='same', activation='relu'),
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

def add_noise_to_data(X, noise_level):
    """Добавляет шум к данным"""
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    # Ограничиваем значения в диапазоне [0, 1]
    noisy_X = np.clip(noisy_X, 0, 1)
    return noisy_X

def test_with_noise(model, X_test, y_test, scaler, label_encoder, noise_levels=[0.0, 0.01, 0.05, 0.10]):
    """Тестирует модель на данных с разными уровнями шума"""
    print(f"\nТЕСТИРОВАНИЕ С ШУМОМ:")
    print("=" * 50)
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\nТестирование с шумом {noise_level*100}%:")
        
        # Добавляем шум к тестовым данным
        if noise_level > 0:
            X_test_noisy = add_noise_to_data(X_test, noise_level)
        else:
            X_test_noisy = X_test.copy()
        
        # Масштабируем данные
        X_test_scaled = scaler.transform(X_test_noisy)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # Предсказания
        y_pred_proba = model.predict(X_test_reshaped, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Точность
        accuracy = accuracy_score(y_test, y_pred)
        
        # Средняя уверенность
        confidence = np.mean(np.max(y_pred_proba, axis=1))
        confidence_std = np.std(np.max(y_pred_proba, axis=1))
        
        print(f"  Точность: {accuracy:.4f}")
        print(f"  Средняя уверенность: {confidence:.4f} ± {confidence_std:.4f}")
        
        # Анализ по классам
        print("  Точность по классам:")
        for i, class_name in enumerate(label_encoder.classes_):
            class_mask = (y_test == i)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_confidence = np.mean(np.max(y_pred_proba[class_mask], axis=1))
                print(f"    {class_name}: {class_accuracy:.3f} (уверенность: {class_confidence:.3f})")
        
        results.append({
            'noise_level': noise_level,
            'accuracy': accuracy,
            'confidence': confidence,
            'confidence_std': confidence_std
        })
    
    return results

def plot_confusion_matrix(y_true, y_pred, label_encoder, title, filename):
    """Создает матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax1)
    ax1.set_title(f'{title} (абсолютные значения)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Предсказанный класс', fontsize=12)
    ax1.set_ylabel('Истинный класс', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax2)
    ax2.set_title(f'{title} (нормализованные значения)', fontsize=14, fontweight='bold')
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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_noise_results(results, filename):
    """Создает график результатов тестирования с шумом"""
    noise_levels = [r['noise_level'] * 100 for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    confidences = [r['confidence'] * 100 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График точности
    ax1.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Уровень шума (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Точность (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Точность классификации при разных уровнях шума', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Добавляем значения на точки
    for i, (x, y) in enumerate(zip(noise_levels, accuracies)):
        ax1.annotate(f'{y:.1f}%', (x, y + 2), ha='center', va='bottom', fontweight='bold')
    
    # График уверенности
    ax2.plot(noise_levels, confidences, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Уровень шума (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Средняя уверенность (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Уверенность модели при разных уровнях шума', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Добавляем значения на точки
    for i, (x, y) in enumerate(zip(noise_levels, confidences)):
        ax2.annotate(f'{y:.1f}%', (x, y + 2), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Основная функция"""
    print("КЛАССИЧЕСКАЯ VANDY ALEXNET - 7 ВИДОВ С ТЕСТИРОВАНИЕМ ШУМА")
    print("=" * 60)
    
    # Загружаем данные
    X, y = load_spectral_data_7_species()
    
    print(f"\nЗагружено {len(X)} спектров")
    
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
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Изменяем форму для CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Создаем классическую модель Vandy AlexNet
    model = create_classic_vandy_alexnet_model(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Компилируем модель
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nАрхитектура классической Vandy AlexNet:")
    model.summary()
    
    # Обучаем модель
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
    
    print("\nОбучение модели...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Оцениваем модель на чистых данных
    print("\nРЕЗУЛЬТАТЫ НА ЧИСТЫХ ДАННЫХ:")
    print("-" * 50)
    
    y_pred_proba = model.predict(X_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на тестовых данных: {accuracy:.4f}")
    
    # Отчет о классификации
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Тестируем с шумом
    noise_results = test_with_noise(model, X_test, y_test, scaler, label_encoder)
    
    # Создаем графики
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Матрица ошибок для чистых данных
    plot_confusion_matrix(y_test, y_pred, label_encoder, 
                         f'Классическая Vandy AlexNet (точность: {accuracy:.1%})',
                         f'classic_vandy_alexnet_confusion_matrix_{timestamp}.png')
    
    # История обучения
    plot_training_history(history, f'classic_vandy_alexnet_training_history_{timestamp}.png')
    
    # Результаты с шумом
    plot_noise_results(noise_results, f'classic_vandy_alexnet_noise_results_{timestamp}.png')
    
    # Сохраняем модель
    model.save(f'classic_vandy_alexnet_model_{timestamp}.h5')
    
    import joblib
    joblib.dump(scaler, f'classic_vandy_alexnet_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'classic_vandy_alexnet_label_encoder_{timestamp}.pkl')
    
    # Сохраняем результаты в CSV
    results_df = pd.DataFrame(noise_results)
    results_df.to_csv(f'classic_vandy_alexnet_noise_results_{timestamp}.csv', index=False)
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    
    # Открываем результаты
    import subprocess
    subprocess.run(['open', f'classic_vandy_alexnet_confusion_matrix_{timestamp}.png'])
    subprocess.run(['open', f'classic_vandy_alexnet_noise_results_{timestamp}.png'])
    
    return accuracy, model, scaler, label_encoder

if __name__ == "__main__":
    main() 