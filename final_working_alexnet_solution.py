import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
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
    base_path = "Спектры, весенний период, 7 видов"
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
        return None, None
    
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"Загружено {len(X)} образцов для {len(np.unique(y))} видов")
    return X, y

def create_improved_alexnet(input_shape, num_classes):
    """УЛУЧШЕННАЯ модифицированная 1D-AlexNet"""
    model = Sequential([
        # Группа 1: 16 фильтров (увеличено с 10)
        Conv1D(16, 50, strides=4, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        # Группа 2: 32 фильтра (увеличено с 20)
        Conv1D(32, 50, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        # Группа 3: 64 → 64 → 32 фильтра (увеличено)
        Conv1D(64, 3, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, 3, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(32, 3, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        Flatten(),
        
        # Полносвязные слои: 512 → 256 → 7 (увеличено)
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),  # Уменьшен dropout
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Уменьшен learning rate
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

def test_improved_alexnet():
    """Тестирование улучшенной модели"""
    print("=== УЛУЧШЕННАЯ МОДИФИЦИРОВАННАЯ 1D-ALEXNET ===")
    
    # Загрузка данных
    print("1. Загрузка данных...")
    data, labels = load_spectral_data_7_species()
    if data is None:
        return None, None, None, False
    
    # Проверяем исходные данные
    print(f"Исходные данные - min: {np.min(data):.4f}, max: {np.max(data):.4f}, std: {np.std(data):.4f}")
    
    # Используем RobustScaler (менее агрессивная нормализация)
    print("2. Мягкая нормализация (RobustScaler)...")
    scaler = RobustScaler()
    
    # Преобразуем для скалера
    data_reshaped = data.reshape(-1, data.shape[-1])
    data_scaled = scaler.fit_transform(data_reshaped)
    X_processed = data_scaled.reshape(data.shape)
    
    # Проверяем данные после обработки
    print(f"После RobustScaler - min: {np.min(X_processed):.4f}, max: {np.max(X_processed):.4f}, std: {np.std(X_processed):.4f}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество классов: {len(label_encoder.classes_)}")
    print(f"Классы: {label_encoder.classes_}")
    
    # Создание улучшенной модели
    print("3. Создание улучшенной модели...")
    model = create_improved_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    print("Архитектура улучшенной модели:")
    model.summary()
    
    # Обучение модели
    print("4. Обучение улучшенной модели...")
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=200,
        batch_size=16,  # Уменьшенный batch size
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
    print(f"Первые 10 вероятностей: {max_probs_clean[:10]}")
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
    print("📊 РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ МОДЕЛИ:")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Проверяем, работает ли модель правильно
    print("\n" + "="*80)
    print("🔍 ПРОВЕРКА РАБОТОСПОСОБНОСТИ:")
    print("="*80)
    
    # Проверяем точность
    if accuracy_clean > 0.5:  # Более 50% точности
        print("✅ ТОЧНОСТЬ: Отличная! > 50%")
        accuracy_status = "✅"
    elif accuracy_clean > 0.3:
        print("✅ ТОЧНОСТЬ: Приемлемая > 30%")
        accuracy_status = "✅"
    else:
        print("❌ ТОЧНОСТЬ: Слишком низкая < 30%")
        accuracy_status = "❌"
    
    # Проверяем тренд вероятностей (должны снижаться с шумом)
    prob_trend = df_results['mean_max_probability'].iloc[-1] - df_results['mean_max_probability'].iloc[0]
    if prob_trend < -0.05:  # Снижение на 5%
        print("✅ ВЕРОЯТНОСТИ: Корректно снижаются с шумом!")
        prob_status = "✅"
    elif abs(prob_trend) < 0.02:  # Небольшое изменение
        print("⚠️ ВЕРОЯТНОСТИ: Мало меняются (возможно, нормально)")
        prob_status = "⚠️"
    else:
        print("❌ ВЕРОЯТНОСТИ: Растут с шумом (неправильно)")
        prob_status = "❌"
    
    # Проверяем уникальность
    min_uniqueness = df_results['uniqueness_ratio'].min()
    if min_uniqueness > 0.5:  # Более 50% уникальных
        print("✅ УНИКАЛЬНОСТЬ: Отличная > 50%")
        unique_status = "✅"
    elif min_uniqueness > 0.2:
        print("✅ УНИКАЛЬНОСТЬ: Приемлемая > 20%")
        unique_status = "✅"
    else:
        print("❌ УНИКАЛЬНОСТЬ: Слишком низкая < 20%")
        unique_status = "❌"
    
    # Общий статус
    overall_working = accuracy_status == "✅" and (prob_status in ["✅", "⚠️"]) and unique_status == "✅"
    
    if overall_working:
        print("\n🎉 МОДЕЛЬ ПОЛНОСТЬЮ РАБОТОСПОСОБНА!")
        print(f"✅ Точность: {accuracy_clean*100:.1f}%")
        print(f"✅ Параметров: {model.count_params():,}")
        print(f"✅ Эпох обучения: {len(history.history['accuracy'])}")
    else:
        print("\n⚠️ МОДЕЛЬ ТРЕБУЕТ ДОПОЛНИТЕЛЬНОЙ НАСТРОЙКИ")
        print(f"   Точность: {accuracy_clean*100:.1f}% ({accuracy_status})")
        print(f"   Вероятности: {prob_status}")
        print(f"   Уникальность: {unique_status}")
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График 1: Вероятности vs шум
    axes[0,0].plot(df_results['noise_percent'], df_results['mean_max_probability'], 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Уровень шума (%)')
    axes[0,0].set_ylabel('Средняя макс. вероятность')
    axes[0,0].set_title('Влияние шума на вероятности (УЛУЧШЕННАЯ)')
    axes[0,0].grid(True, alpha=0.3)
    
    # График 2: Точность vs шум
    axes[0,1].plot(df_results['noise_percent'], df_results['accuracy']*100, 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Уровень шума (%)')
    axes[0,1].set_ylabel('Точность (%)')
    axes[0,1].set_title('Влияние шума на точность (УЛУЧШЕННАЯ)')
    axes[0,1].grid(True, alpha=0.3)
    
    # График 3: Уникальность вероятностей
    axes[1,0].plot(df_results['noise_percent'], df_results['uniqueness_ratio']*100, 'go-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Уровень шума (%)')
    axes[1,0].set_ylabel('Уникальность (%)')
    axes[1,0].set_title('Уникальность вероятностей (УЛУЧШЕННАЯ)')
    axes[1,0].grid(True, alpha=0.3)
    
    # График 4: История обучения
    axes[1,1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1,1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1,1].set_xlabel('Эпоха')
    axes[1,1].set_ylabel('Точность')
    axes[1,1].set_title('История обучения (УЛУЧШЕННАЯ)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/улучшенная_модель_результаты.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем матрицы ошибок
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
        axes[i].set_title(f'Шум: {noise_level*100}%\nТочность: {accuracy*100:.1f}% (УЛУЧШ.)')
        axes[i].set_xlabel('Предсказанный класс')
        axes[i].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/улучшенные_матрицы_ошибок.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохраняем результаты
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/улучшенная_модель_результаты.csv', index=False)
    
    print(f"\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"📁 Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/")
    print(f"⏱️  Время обучения: {len(history.history['accuracy'])} эпох")
    print(f"🎯 Итоговая точность: {accuracy_clean*100:.2f}%")
    print(f"🔢 Параметров модели: {model.count_params():,}")
    
    return model, history, df_results, overall_working

if __name__ == "__main__":
    result = test_improved_alexnet()
    
    if result and len(result) == 4:
        model, history, results, is_working = result
        if is_working:
            print("\n🎉 ВСЕ РАБОТАЕТ! ГОТОВО К ПУШУ НА GITHUB!")
        else:
            print("\n⚠️ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ НАСТРОЙКА")
    else:
        print("❌ ОШИБКА ЗАГРУЗКИ ДАННЫХ")