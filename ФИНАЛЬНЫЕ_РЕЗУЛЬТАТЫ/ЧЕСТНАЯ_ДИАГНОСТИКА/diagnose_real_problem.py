import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import os

# Устанавливаем seeds
np.random.seed(42)
tf.random.set_seed(42)

def create_diagnostic_alexnet(input_shape, num_classes):
    """Простая диагностическая модель"""
    model = Sequential([
        Conv1D(32, 10, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def add_gaussian_noise(data, noise_level):
    """Добавление гауссового шума"""
    if noise_level == 0:
        return data
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise

def create_realistic_synthetic_data():
    """Создание ДЕЙСТВИТЕЛЬНО различающихся синтетических данных"""
    print("Создание реалистичных данных с ЯВНЫМИ различиями...")
    
    species_names = ["береза", "дуб", "ель", "клен", "липа", "осина", "сосна"]
    all_data = []
    all_labels = []
    
    samples_per_species = 30
    spectrum_length = 300
    
    for i, species in enumerate(species_names):
        print(f"Создание {samples_per_species} образцов для {species}...")
        
        for j in range(samples_per_species):
            # Создаём УНИКАЛЬНЫЕ спектральные характеристики для каждого вида
            base_freq = 2 + i * 0.5  # Разная базовая частота для каждого вида
            amplitude = 0.3 + i * 0.1  # Разная амплитуда
            
            # Основной сигнал
            x = np.linspace(0, 10, spectrum_length)
            spectrum = 0.5 + amplitude * np.sin(base_freq * x)
            
            # Добавляем уникальные пики для каждого вида
            if species == "береза":
                spectrum += 0.2 * np.sin(8 * x) * np.exp(-((x-3)**2)/2)
            elif species == "дуб":
                spectrum += 0.3 * np.sin(12 * x) * np.exp(-((x-5)**2)/3)
            elif species == "ель":
                spectrum += 0.25 * np.sin(6 * x) * np.exp(-((x-7)**2)/2.5)
            elif species == "клен":
                spectrum += 0.2 * np.sin(15 * x) * np.exp(-((x-2)**2)/1.5)
            elif species == "липа":
                spectrum += 0.35 * np.sin(10 * x) * np.exp(-((x-8)**2)/3)
            elif species == "осина":
                spectrum += 0.3 * np.sin(4 * x) * np.exp(-((x-4)**2)/2)
            elif species == "сосна":
                spectrum += 0.25 * np.sin(20 * x) * np.exp(-((x-6)**2)/2)
            
            # Добавляем небольшой случайный шум
            spectrum += np.random.normal(0, 0.02, spectrum_length)
            
            # Нормализуем
            spectrum = np.clip(spectrum, 0.1, 1.0)
            
            all_data.append(spectrum)
            all_labels.append(species)
    
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"✅ Создано {X.shape[0]} образцов")
    print(f"✅ Размер спектра: {X.shape[1]}")
    print(f"✅ Виды: {np.unique(y)}")
    
    # Проверяем различия между видами
    print("\n🔍 ПРОВЕРКА РАЗЛИЧИЙ МЕЖДУ ВИДАМИ:")
    for i, species in enumerate(species_names):
        species_data = X[y == species]
        mean_spectrum = np.mean(species_data, axis=0)
        print(f"{species}: среднее={np.mean(mean_spectrum):.3f}, std={np.std(mean_spectrum):.3f}")
    
    return X, y

def honest_diagnosis():
    """ЧЕСТНАЯ диагностика проблемы"""
    print("🔍 ЧЕСТНАЯ ДИАГНОСТИКА ПРОБЛЕМЫ...")
    print("="*60)
    
    # Создаём данные с реальными различиями
    X, y = create_realistic_synthetic_data()
    
    # БЕЗ агрессивной нормализации
    print("\nИспользуем сырые данные БЕЗ нормализации...")
    X_processed = X  # Используем как есть
    
    print(f"Данные: min={np.min(X_processed):.3f}, max={np.max(X_processed):.3f}, std={np.std(X_processed):.3f}")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Подготавливаем для CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"\nОбучающая выборка: {X_train_cnn.shape}")
    print(f"Тестовая выборка: {X_test_cnn.shape}")
    print(f"Распределение в тесте: {np.bincount(y_test)}")
    
    # Создаём простую модель
    print("\nСоздание простой диагностической модели...")
    model = create_diagnostic_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    print("Архитектура модели:")
    model.summary()
    
    # Обучение
    print("\nОбучение модели...")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Тестирование БЕЗ шума
    print("\n" + "="*60)
    print("📊 ТЕСТИРОВАНИЕ БЕЗ ШУМА:")
    print("="*60)
    
    y_pred_proba = model.predict(X_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    print(f"Точность: {accuracy*100:.2f}%")
    
    # Анализируем предсказания
    pred_distribution = np.bincount(y_pred_classes, minlength=len(label_encoder.classes_))
    print(f"Распределение предсказаний: {pred_distribution}")
    
    for i, species in enumerate(label_encoder.classes_):
        print(f"{species}: {pred_distribution[i]} предсказаний")
    
    # Матрица ошибок БЕЗ шума
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    print(f"\nМатрица ошибок БЕЗ шума (нормализованная):")
    cm_df = pd.DataFrame(cm_normalized, 
                         index=label_encoder.classes_, 
                         columns=label_encoder.classes_)
    print(cm_df.round(3))
    
    # Тестирование С ШУМОМ
    print("\n" + "="*60)
    print("🔊 ТЕСТИРОВАНИЕ С РАЗНЫМИ УРОВНЯМИ ШУМА:")
    print("="*60)
    
    noise_levels = [0, 0.05, 0.1, 0.2]
    noise_results = []
    
    for noise_level in noise_levels:
        print(f"\n--- ШУМ {noise_level*100}% ---")
        
        # Добавляем шум
        X_test_noisy = add_gaussian_noise(X_test_cnn, noise_level)
        
        # Предсказания
        y_pred_proba_noisy = model.predict(X_test_noisy, verbose=0)
        y_pred_classes_noisy = np.argmax(y_pred_proba_noisy, axis=1)
        accuracy_noisy = accuracy_score(y_test, y_pred_classes_noisy)
        
        # Распределение
        pred_dist_noisy = np.bincount(y_pred_classes_noisy, minlength=len(label_encoder.classes_))
        
        print(f"Точность: {accuracy_noisy*100:.2f}%")
        print(f"Распределение: {pred_dist_noisy}")
        
        # Матрица ошибок
        cm_noisy = confusion_matrix(y_test, y_pred_classes_noisy)
        cm_noisy_norm = cm_noisy.astype('float') / cm_noisy.sum(axis=1)[:, np.newaxis]
        cm_noisy_norm = np.nan_to_num(cm_noisy_norm)
        
        noise_results.append({
            'noise_level': noise_level,
            'accuracy': accuracy_noisy,
            'matrix': cm_noisy_norm,
            'distribution': pred_dist_noisy
        })
    
    # Анализ влияния шума
    print("\n" + "="*60)
    print("📈 АНАЛИЗ ВЛИЯНИЯ ШУМА:")
    print("="*60)
    
    base_accuracy = noise_results[0]['accuracy']
    print(f"Базовая точность (0% шума): {base_accuracy*100:.2f}%")
    
    for result in noise_results[1:]:
        change = result['accuracy'] - base_accuracy
        print(f"Шум {result['noise_level']*100}%: {result['accuracy']*100:.2f}% (изменение: {change*100:+.2f}%)")
    
    # Сохраняем ЧЕСТНЫЕ результаты
    os.makedirs('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ЧЕСТНАЯ_ДИАГНОСТИКА', exist_ok=True)
    
    # Создаём честные матрицы
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, result in enumerate(noise_results):
        sns.heatmap(result['matrix'], 
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[i])
        
        axes[i].set_title(f'ШУМ: {result["noise_level"]*100}%\nТочность: {result["accuracy"]*100:.1f}%')
        axes[i].set_xlabel('Предсказанный класс')
        axes[i].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ЧЕСТНАЯ_ДИАГНОСТИКА/honest_confusion_matrices.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем честный отчёт
    with open('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ЧЕСТНАЯ_ДИАГНОСТИКА/honest_report.txt', 'w', encoding='utf-8') as f:
        f.write("ЧЕСТНЫЙ ОТЧЁТ О ДИАГНОСТИКЕ\n")
        f.write("="*50 + "\n\n")
        f.write(f"Базовая точность: {base_accuracy*100:.2f}%\n")
        f.write(f"Параметров модели: {model.count_params():,}\n")
        f.write(f"Эпох обучения: {len(history.history['accuracy'])}\n\n")
        
        f.write("ВЛИЯНИЕ ШУМА НА ТОЧНОСТЬ:\n")
        f.write("-"*30 + "\n")
        for result in noise_results:
            change = result['accuracy'] - base_accuracy
            f.write(f"Шум {result['noise_level']*100:4.1f}%: точность {result['accuracy']*100:5.1f}% (изменение: {change*100:+5.1f}%)\n")
        
        f.write(f"\nРАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:\n")
        f.write("-"*30 + "\n")
        for result in noise_results:
            f.write(f"Шум {result['noise_level']*100:4.1f}%: {result['distribution']}\n")
    
    print(f"\n✅ ЧЕСТНАЯ диагностика завершена!")
    print(f"📁 Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ЧЕСТНАЯ_ДИАГНОСТИКА/")
    print(f"🎯 Базовая точность: {base_accuracy*100:.2f}%")
    print(f"🔢 Параметров модели: {model.count_params():,}")
    
    return noise_results

if __name__ == "__main__":
    results = honest_diagnosis()
    print("\n🎯 ЧЕСТНАЯ ДИАГНОСТИКА ЗАВЕРШЕНА!")