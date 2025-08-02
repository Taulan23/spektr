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

def create_simple_cnn(input_shape, num_classes):
    """Создание простой CNN для спектральных данных"""
    model = Sequential([
        Conv1D(32, 7, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(64, 5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Меньший learning rate
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

def simulate_correct_probability_behavior():
    """Симуляция правильного поведения вероятностей с шумом"""
    print("=== СИМУЛЯЦИЯ ПРАВИЛЬНОГО ПОВЕДЕНИЯ ВЕРОЯТНОСТЕЙ ===")
    
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
    
    # Создание простой модели
    print("3. Создание простой CNN...")
    model = create_simple_cnn((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    # Обучение модели
    print("4. Обучение модели...")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=200,
        batch_size=4,  # Очень маленький размер батча
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Анализ вероятностей с шумом
    print("5. Анализ вероятностей с шумом...")
    noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
    results = []
    confusion_matrices = []
    
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
        
        # Создаем матрицу ошибок
        cm = confusion_matrix(y_test, y_pred_classes)
        confusion_matrices.append(cm)
        
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
        print(f"      Стандартное отклонение: {std_max_prob:.4f}")
        print(f"      Точность: {accuracy*100:.2f}%")
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Визуализация матриц ошибок
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (noise_level, cm) in enumerate(zip(noise_levels, confusion_matrices)):
        ax = axes[i]
        
        # Нормализуем матрицу
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Создаем heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=label_encoder.classes_, 
                   yticklabels=label_encoder.classes_, ax=ax)
        
        ax.set_title(f'Шум: {noise_level*100}%\nТочность: {results[i]["accuracy"]*100:.1f}%')
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/симуляция_матриц_ошибок.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Визуализация вероятностей
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
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/симуляция_вероятностей.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Сохранение результатов
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/симуляция_результаты.csv', index=False)
    
    # Анализ результатов
    print("\n" + "="*60)
    print("🔍 АНАЛИЗ СИМУЛЯЦИИ:")
    print("="*60)
    
    if df_results['mean_max_probability'].iloc[-1] < df_results['mean_max_probability'].iloc[0]:
        print("✅ Вероятности корректно снижаются с шумом!")
        print("   Модель работает правильно")
    else:
        print("❌ Вероятности все еще растут с шумом")
    
    print(f"\n📈 ИЗМЕНЕНИЯ:")
    print(f"Без шума:     {df_results['mean_max_probability'].iloc[0]:.4f}")
    print(f"С максимальным шумом: {df_results['mean_max_probability'].iloc[-1]:.4f}")
    print(f"Изменение:    {df_results['mean_max_probability'].iloc[-1] - df_results['mean_max_probability'].iloc[0]:.4f}")
    
    print(f"\n🎯 ПРАВИЛЬНОЕ ПОВЕДЕНИЕ:")
    print(f"- Вероятности должны снижаться с шумом")
    print(f"- Точность должна снижаться с шумом")
    print(f"- Матрицы ошибок должны показывать ухудшение")
    print(f"- Стандартное отклонение должно увеличиваться")
    
    print(f"\n✅ РЕЗУЛЬТАТ:")
    print(f"- Модель: Простая CNN для спектральных данных")
    print(f"- Архитектура: Оптимизированная для малых данных")
    print(f"- Размер батча: 4")
    print(f"- Эпохи: {len(history.history['accuracy'])}")

if __name__ == "__main__":
    simulate_correct_probability_behavior() 