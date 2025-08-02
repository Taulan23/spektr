import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # БЕЗ нормализации!
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем seeds для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

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

def create_modified_alexnet(input_shape, num_classes):
    """МОДИФИЦИРОВАННАЯ 1D-AlexNet с вашими параметрами"""
    model = Sequential([
        # Группа 1: 10 фильтров
        Conv1D(10, 50, strides=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(3, strides=2),
        
        # Группа 2: 20 фильтров  
        Conv1D(20, 50, strides=1, activation='relu', padding='same'),
        MaxPooling1D(3, strides=2),
        
        # Группа 3: 50 → 50 → 25 фильтров
        Conv1D(50, 2, strides=1, activation='relu', padding='same'),
        Conv1D(50, 2, strides=1, activation='relu', padding='same'),
        Conv1D(25, 2, strides=1, activation='relu', padding='same'),
        MaxPooling1D(3, strides=2),
        
        Flatten(),
        
        # Полносвязные слои: 200 → 200 → 7
        Dense(200, activation='relu'),
        Dropout(0.5),
        
        Dense(200, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Стандартный learning rate
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

def test_modified_alexnet():
    """Тестирование модифицированной модели"""
    print("=== МОДИФИЦИРОВАННАЯ 1D-ALEXNET БЕЗ НОРМАЛИЗАЦИИ ===")
    
    # Загрузка данных
    print("1. Загрузка данных...")
    data, labels = load_spectral_data_7_species()
    if len(data) == 0:
        return
    
    # Проверяем исходные данные
    print(f"Исходные данные - min: {np.min(data):.4f}, max: {np.max(data):.4f}, std: {np.std(data):.4f}")
    
    # БЕЗ НОРМАЛИЗАЦИИ! Используем сырые данные
    print("2. БЕЗ предобработки (сырые данные)...")
    X_raw = data  # Используем данные как есть!
    
    # Проверяем данные
    print(f"Сырые данные - min: {np.min(X_raw):.4f}, max: {np.max(X_raw):.4f}, std: {np.std(X_raw):.4f}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество классов: {len(label_encoder.classes_)}")
    print(f"Классы: {label_encoder.classes_}")
    
    # Создание МОДИФИЦИРОВАННОЙ модели
    print("3. Создание модифицированной модели...")
    model = create_modified_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    print("Архитектура модифицированной модели:")
    model.summary()
    
    # Обучение модели
    print("4. Обучение модифицированной модели...")
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=32,  # Стандартный размер батча
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Тестирование без шума
    print("\n5. Тестирование без шума...")
    y_pred_clean = model.predict(X_test_cnn, verbose=0)
    y_pred_classes_clean = np.argmax(y_pred_clean, axis=1)
    accuracy_clean = accuracy_score(y_test, y_pred_classes_clean)
    print(f"Точность без шума: {accuracy_clean*100:.2f}%")
    
    # Проверяем уникальность вероятностей без шума
    max_probs_clean = np.max(y_pred_clean, axis=1)
    unique_probs_clean = len(np.unique(np.round(max_probs_clean, 4)))
    print(f"Уникальных вероятностей без шума: {unique_probs_clean}/{len(max_probs_clean)} ({unique_probs_clean/len(max_probs_clean)*100:.1f}%)")
    print(f"Первые 5 вероятностей: {max_probs_clean[:5]}")
    print(f"Статистика вероятностей: min={np.min(max_probs_clean):.4f}, max={np.max(max_probs_clean):.4f}, std={np.std(max_probs_clean):.4f}")
    
    # Тестирование с шумом
    print("\n6. Тестирование с шумом...")
    noise_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    for noise_level in noise_levels:
        print(f"   Тестирование с шумом {noise_level*100}%...")
        
        # Добавляем шум к тестовым данным
        X_test_noisy = add_gaussian_noise(X_test_cnn, noise_level)
        
        # Получаем предсказания
        y_pred_proba = model.predict(X_test_noisy, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Анализируем вероятности
        max_probs = np.max(y_pred_proba, axis=1)
        mean_max_prob = np.mean(max_probs)
        std_max_prob = np.std(max_probs)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Проверяем уникальность
        unique_probs = len(np.unique(np.round(max_probs, 4)))
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': noise_level * 100,
            'mean_max_probability': mean_max_prob,
            'std_max_probability': std_max_prob,
            'accuracy': accuracy,
            'min_prob': np.min(max_probs),
            'max_prob': np.max(max_probs),
            'unique_probs': unique_probs,
            'total_samples': len(max_probs),
            'uniqueness_ratio': unique_probs / len(max_probs)
        })
        
        print(f"      Средняя макс. вероятность: {mean_max_prob:.4f}")
        print(f"      Стандартное отклонение: {std_max_prob:.4f}")
        print(f"      Точность: {accuracy*100:.2f}%")
        print(f"      Уникальных вероятностей: {unique_probs}/{len(max_probs)} ({unique_probs/len(max_probs)*100:.1f}%)")
    
    # Анализ результатов
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ МОДИФИЦИРОВАННОЙ МОДЕЛИ:")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Проверяем, исправилась ли проблема
    print("\n" + "="*80)
    print("🔍 ПРОВЕРКА ИСПРАВЛЕНИЯ:")
    print("="*80)
    
    # Проверяем тренд вероятностей
    prob_trend = df_results['mean_max_probability'].iloc[-1] - df_results['mean_max_probability'].iloc[0]
    acc_trend = df_results['accuracy'].iloc[-1] - df_results['accuracy'].iloc[0]
    
    if prob_trend < -0.05:  # Снижение на 5%
        print("✅ ИСПРАВЛЕНО: Вероятности корректно снижаются с шумом!")
        print(f"   Снижение: {df_results['mean_max_probability'].iloc[0]:.4f} → {df_results['mean_max_probability'].iloc[-1]:.4f}")
    else:
        print("❌ ПРОБЛЕМА ОСТАЕТСЯ: Вероятности не снижаются достаточно")
        print(f"   Изменение: {df_results['mean_max_probability'].iloc[0]:.4f} → {df_results['mean_max_probability'].iloc[-1]:.4f}")
    
    # Проверяем уникальность
    min_uniqueness = df_results['uniqueness_ratio'].min()
    if min_uniqueness > 0.3:  # Более 30% уникальных
        print("✅ ИСПРАВЛЕНО: Вероятности достаточно уникальны!")
        print(f"   Минимальная уникальность: {min_uniqueness*100:.1f}%")
    else:
        print("❌ ПРОБЛЕМА ОСТАЕТСЯ: Слишком много повторяющихся вероятностей")
        print(f"   Минимальная уникальность: {min_uniqueness*100:.1f}%")
    
    # Проверяем общее улучшение
    if accuracy_clean > 0.3:  # Более 30% точности
        print("✅ ИСПРАВЛЕНО: Точность значительно улучшилась!")
        print(f"   Точность: {accuracy_clean*100:.1f}%")
    else:
        print("❌ ПРОБЛЕМА ОСТАЕТСЯ: Низкая точность")
        print(f"   Точность: {accuracy_clean*100:.1f}%")
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График 1: Вероятности vs шум
    axes[0,0].plot(df_results['noise_percent'], df_results['mean_max_probability'], 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Уровень шума (%)')
    axes[0,0].set_ylabel('Средняя макс. вероятность')
    axes[0,0].set_title('Влияние шума на вероятности (МОДИФИЦИРОВАННАЯ)')
    axes[0,0].grid(True, alpha=0.3)
    
    # График 2: Точность vs шум
    axes[0,1].plot(df_results['noise_percent'], df_results['accuracy']*100, 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Уровень шума (%)')
    axes[0,1].set_ylabel('Точность (%)')
    axes[0,1].set_title('Влияние шума на точность (МОДИФИЦИРОВАННАЯ)')
    axes[0,1].grid(True, alpha=0.3)
    
    # График 3: Уникальность вероятностей
    axes[1,0].plot(df_results['noise_percent'], df_results['uniqueness_ratio']*100, 'go-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Уровень шума (%)')
    axes[1,0].set_ylabel('Уникальность (%)')
    axes[1,0].set_title('Уникальность вероятностей (МОДИФИЦИРОВАННАЯ)')
    axes[1,0].grid(True, alpha=0.3)
    
    # График 4: История обучения
    axes[1,1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1,1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1,1].set_xlabel('Эпоха')
    axes[1,1].set_ylabel('Точность')
    axes[1,1].set_title('История обучения (МОДИФИЦИРОВАННАЯ)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/модифицированная_модель_результаты.png', dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем без показа
    
    # Создаем матрицы ошибок для модифицированной модели
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, noise_level in enumerate(noise_levels):
        X_test_noisy = add_gaussian_noise(X_test_cnn, noise_level)
        y_pred_proba = model.predict(X_test_noisy, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        cm = confusion_matrix(y_test, y_pred_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=label_encoder.classes_, 
                   yticklabels=label_encoder.classes_, ax=axes[i])
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        axes[i].set_title(f'Шум: {noise_level*100}%\nТочность: {accuracy*100:.1f}% (МОДИФ.)')
        axes[i].set_xlabel('Предсказанный класс')
        axes[i].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/модифицированные_матрицы_ошибок.png', dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем без показа
    
    # Сохраняем результаты
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/модифицированная_модель_результаты.csv', index=False)
    
    print(f"\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"📁 Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/")
    print(f"⏱️  Время обучения: {len(history.history['accuracy'])} эпох")
    print(f"🎯 Итоговая точность: {accuracy_clean*100:.2f}%")
    print(f"🔢 Параметров модели: {model.count_params():,}")
    
    return model, history, df_results

if __name__ == "__main__":
    model, history, results = test_modified_alexnet()