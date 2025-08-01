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

def create_original_1d_alexnet(input_shape, num_classes):
    """Создание ОРИГИНАЛЬНОЙ 1D-AlexNet БЕЗ Dropout"""
    model = Sequential([
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
        Dense(4096, activation='relu'),
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

def analyze_probability_behavior():
    """Анализ поведения вероятностей с шумом"""
    print("=== АНАЛИЗ ПРОБЛЕМЫ С ВЕРОЯТНОСТЯМИ ===")
    
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
    
    # Создание ОРИГИНАЛЬНОЙ модели
    print("3. Создание ОРИГИНАЛЬНОЙ модели...")
    model = create_original_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    # Обучение модели
    print("4. Обучение модели...")
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
    
    # Анализ вероятностей с разными уровнями шума
    print("5. Анализ вероятностей с шумом...")
    noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
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
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': noise_level * 100,
            'mean_max_probability': mean_max_prob,
            'std_max_probability': std_max_prob,
            'accuracy': accuracy,
            'min_prob': np.min(max_probs),
            'max_prob': np.max(max_probs)
        })
        
        print(f"      Средняя макс. вероятность: {mean_max_prob:.4f}")
        print(f"      Точность: {accuracy*100:.2f}%")
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА ВЕРОЯТНОСТЕЙ:")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Визуализация
    plt.figure(figsize=(15, 10))
    
    # График 1: Средняя максимальная вероятность vs шум
    plt.subplot(2, 2, 1)
    plt.plot(df_results['noise_percent'], df_results['mean_max_probability'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Средняя максимальная вероятность')
    plt.title('Влияние шума на максимальную вероятность')
    plt.grid(True, alpha=0.3)
    
    # График 2: Точность vs шум
    plt.subplot(2, 2, 2)
    plt.plot(df_results['noise_percent'], df_results['accuracy']*100, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность (%)')
    plt.title('Влияние шума на точность')
    plt.grid(True, alpha=0.3)
    
    # График 3: Диапазон вероятностей
    plt.subplot(2, 2, 3)
    plt.fill_between(df_results['noise_percent'], 
                     df_results['min_prob'], 
                     df_results['max_prob'], 
                     alpha=0.3, color='green')
    plt.plot(df_results['noise_percent'], df_results['mean_max_probability'], 'go-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Вероятность')
    plt.title('Диапазон максимальных вероятностей')
    plt.grid(True, alpha=0.3)
    
    # График 4: Стандартное отклонение
    plt.subplot(2, 2, 4)
    plt.plot(df_results['noise_percent'], df_results['std_max_probability'], 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Стандартное отклонение')
    plt.title('Изменчивость максимальных вероятностей')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/анализ_вероятностей_с_шумом.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Сохранение результатов
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/анализ_вероятностей_результаты.csv', index=False)
    
    # Анализ проблемы
    print("\n" + "="*60)
    print("🔍 АНАЛИЗ ПРОБЛЕМЫ:")
    print("="*60)
    
    if df_results['mean_max_probability'].iloc[-1] > df_results['mean_max_probability'].iloc[0]:
        print("❌ ПРОБЛЕМА: Вероятности растут с шумом!")
        print("   Это указывает на проблему в модели или данных")
    else:
        print("✅ Вероятности корректно снижаются с шумом")
    
    print(f"\n📈 ИЗМЕНЕНИЯ:")
    print(f"Без шума:     {df_results['mean_max_probability'].iloc[0]:.4f}")
    print(f"С максимальным шумом: {df_results['mean_max_probability'].iloc[-1]:.4f}")
    print(f"Изменение:    {df_results['mean_max_probability'].iloc[-1] - df_results['mean_max_probability'].iloc[0]:.4f}")
    
    print(f"\n🎯 ВЫВОДЫ:")
    print(f"- Модель: ОРИГИНАЛЬНАЯ 1D-AlexNet (без Dropout)")
    print(f"- Размер батча: 32")
    print(f"- Архитектура: Стандартная AlexNet")
    print(f"- Проблема может быть в переобучении или размере батча")

if __name__ == "__main__":
    analyze_probability_behavior() 