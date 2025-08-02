import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
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

def create_original_1d_alexnet_with_dropout(input_shape, num_classes):
    """ТОЧНО ОРИГИНАЛЬНАЯ модель из статьи + только дропауты"""
    model = Sequential([
        # ТОЧНО как в статье
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
        Dropout(0.5),  # ТОЛЬКО эти дропауты добавлены
        Dense(4096, activation='relu'),
        Dropout(0.5),  # ТОЛЬКО эти дропауты добавлены
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
    
    # Правильная формула шума
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise

def analyze_noise_effect_on_spectra(X_test, noise_levels):
    """Анализ влияния шума на сами спектры"""
    print("\n🔍 АНАЛИЗ ВЛИЯНИЯ ШУМА НА СПЕКТРЫ:")
    print("="*50)
    
    for noise_level in noise_levels:
        X_noisy = add_gaussian_noise(X_test, noise_level)
        
        # Статистика спектров
        original_std = np.std(X_test)
        noisy_std = np.std(X_noisy)
        
        # Корреляция между оригинальными и зашумленными спектрами
        correlations = []
        for i in range(min(50, len(X_test))):  # Первые 50 образцов
            corr = np.corrcoef(X_test[i], X_noisy[i])[0, 1]
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        
        print(f"Шум {noise_level*100:3.0f}%: std_orig={original_std:.4f}, std_noisy={noisy_std:.4f}, корреляция={mean_corr:.4f}")

def debug_probability_behavior():
    """Детальная диагностика поведения вероятностей"""
    print("=== ДИАГНОСТИКА ОРИГИНАЛЬНОЙ 1D-ALEXNET ===")
    
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
    
    # Создание ОРИГИНАЛЬНОЙ модели
    print("3. Создание ОРИГИНАЛЬНОЙ модели из статьи...")
    model = create_original_1d_alexnet_with_dropout((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    print("Архитектура модели:")
    model.summary()
    
    # Обучение модели
    print("4. Обучение модели...")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=32,  # Оригинальный размер батча
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Анализ влияния шума на спектры
    noise_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    analyze_noise_effect_on_spectra(X_test_cnn, noise_levels)
    
    # Тестирование без шума
    print("\n5. Тестирование без шума...")
    y_pred_clean = model.predict(X_test_cnn, verbose=0)
    y_pred_classes_clean = np.argmax(y_pred_clean, axis=1)
    accuracy_clean = accuracy_score(y_test, y_pred_classes_clean)
    print(f"Точность без шума: {accuracy_clean*100:.2f}%")
    
    # Детальный анализ вероятностей с шумом
    print("\n6. Детальный анализ вероятностей с шумом...")
    results = []
    all_predictions = {}
    
    for noise_level in noise_levels:
        print(f"   Тестирование с шумом {noise_level*100}%...")
        
        # Добавляем шум к тестовым данным
        X_test_noisy = add_gaussian_noise(X_test_cnn, noise_level)
        
        # Получаем предсказания
        y_pred_proba = model.predict(X_test_noisy, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Сохраняем для анализа
        all_predictions[f'{noise_level*100:.0f}%'] = y_pred_proba
        
        # Анализируем вероятности
        max_probs = np.max(y_pred_proba, axis=1)
        mean_max_prob = np.mean(max_probs)
        std_max_prob = np.std(max_probs)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Проверяем уникальность вероятностей
        unique_probs = len(np.unique(np.round(max_probs, 4)))
        total_samples = len(max_probs)
        
        # Создаем матрицу ошибок
        cm = confusion_matrix(y_test, y_pred_classes)
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': noise_level * 100,
            'mean_max_probability': mean_max_prob,
            'std_max_probability': std_max_prob,
            'accuracy': accuracy,
            'min_prob': np.min(max_probs),
            'max_prob': np.max(max_probs),
            'unique_probs': unique_probs,
            'total_samples': total_samples,
            'uniqueness_ratio': unique_probs / total_samples
        })
        
        print(f"      Средняя макс. вероятность: {mean_max_prob:.6f}")
        print(f"      Стандартное отклонение: {std_max_prob:.6f}")
        print(f"      Точность: {accuracy*100:.2f}%")
        print(f"      Уникальных вероятностей: {unique_probs}/{total_samples} ({unique_probs/total_samples*100:.1f}%)")
        
        # Показываем первые 5 вероятностей для диагностики
        print(f"      Первые 5 макс. вероятностей: {max_probs[:5]}")
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Проверяем логичность поведения
    print("\n" + "="*80)
    print("🔍 АНАЛИЗ ЛОГИЧНОСТИ ПОВЕДЕНИЯ:")
    print("="*80)
    
    # Проверяем, падают ли вероятности с шумом
    prob_trend = df_results['mean_max_probability'].iloc[-1] - df_results['mean_max_probability'].iloc[0]
    acc_trend = df_results['accuracy'].iloc[-1] - df_results['accuracy'].iloc[0]
    
    if prob_trend < 0:
        print("✅ Вероятности корректно снижаются с шумом")
    else:
        print("❌ ПРОБЛЕМА: Вероятности растут с шумом!")
        print(f"   Изменение: {df_results['mean_max_probability'].iloc[0]:.6f} → {df_results['mean_max_probability'].iloc[-1]:.6f}")
    
    if acc_trend < 0:
        print("✅ Точность корректно снижается с шумом")
    else:
        print("❌ ПРОБЛЕМА: Точность растет с шумом!")
        print(f"   Изменение: {df_results['accuracy'].iloc[0]*100:.2f}% → {df_results['accuracy'].iloc[-1]*100:.2f}%")
    
    # Проверяем уникальность вероятностей
    min_uniqueness = df_results['uniqueness_ratio'].min()
    if min_uniqueness < 0.1:
        print("❌ ПРОБЛЕМА: Слишком много повторяющихся вероятностей!")
        print(f"   Минимальная уникальность: {min_uniqueness*100:.1f}%")
    else:
        print("✅ Вероятности достаточно уникальны")
    
    # Создаем графики для диагностики
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # График 1: Вероятности vs шум
    axes[0,0].plot(df_results['noise_percent'], df_results['mean_max_probability'], 'bo-', linewidth=2)
    axes[0,0].set_xlabel('Уровень шума (%)')
    axes[0,0].set_ylabel('Средняя макс. вероятность')
    axes[0,0].set_title('Влияние шума на вероятности')
    axes[0,0].grid(True, alpha=0.3)
    
    # График 2: Точность vs шум
    axes[0,1].plot(df_results['noise_percent'], df_results['accuracy']*100, 'ro-', linewidth=2)
    axes[0,1].set_xlabel('Уровень шума (%)')
    axes[0,1].set_ylabel('Точность (%)')
    axes[0,1].set_title('Влияние шума на точность')
    axes[0,1].grid(True, alpha=0.3)
    
    # График 3: Уникальность вероятностей
    axes[0,2].plot(df_results['noise_percent'], df_results['uniqueness_ratio']*100, 'go-', linewidth=2)
    axes[0,2].set_xlabel('Уровень шума (%)')
    axes[0,2].set_ylabel('Уникальность (%)')
    axes[0,2].set_title('Уникальность вероятностей')
    axes[0,2].grid(True, alpha=0.3)
    
    # График 4: Стандартное отклонение
    axes[1,0].plot(df_results['noise_percent'], df_results['std_max_probability'], 'mo-', linewidth=2)
    axes[1,0].set_xlabel('Уровень шума (%)')
    axes[1,0].set_ylabel('Стандартное отклонение')
    axes[1,0].set_title('Разброс вероятностей')
    axes[1,0].grid(True, alpha=0.3)
    
    # График 5: Диапазон вероятностей
    axes[1,1].fill_between(df_results['noise_percent'], 
                          df_results['min_prob'], 
                          df_results['max_prob'], 
                          alpha=0.3, color='orange')
    axes[1,1].plot(df_results['noise_percent'], df_results['mean_max_probability'], 'o-', linewidth=2)
    axes[1,1].set_xlabel('Уровень шума (%)')
    axes[1,1].set_ylabel('Вероятность')
    axes[1,1].set_title('Диапазон вероятностей')
    axes[1,1].grid(True, alpha=0.3)
    
    # График 6: История обучения
    axes[1,2].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1,2].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1,2].set_xlabel('Эпоха')
    axes[1,2].set_ylabel('Точность')
    axes[1,2].set_title('История обучения')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/диагностика_оригинальной_модели.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Сохраняем результаты
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/диагностика_результаты.csv', index=False)
    
    # Создаем матрицы ошибок для разных уровней шума
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
        axes[i].set_title(f'Шум: {noise_level*100}%\nТочность: {accuracy*100:.1f}%')
        axes[i].set_xlabel('Предсказанный класс')
        axes[i].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/матрицы_ошибок_диагностика.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ ДИАГНОСТИКА ЗАВЕРШЕНА!")
    print(f"📁 Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/")
    print(f"⏱️  Время обучения: ~{len(history.history['accuracy'])} эпох")

if __name__ == "__main__":
    debug_probability_behavior()