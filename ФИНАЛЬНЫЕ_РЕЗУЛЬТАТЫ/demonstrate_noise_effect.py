import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
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

def add_gaussian_noise_correct(data, noise_level):
    """ПРАВИЛЬНОЕ добавление гауссовского шума"""
    if noise_level == 0:
        return data
    
    # noise_level - это коэффициент (0.1 для 10%)
    std_dev = noise_level * np.std(data)
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

def create_1d_alexnet(input_shape, num_classes):
    """Создание 1D-AlexNet"""
    model = Sequential([
        Conv1D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        Conv1D(256, 5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        Conv1D(384, 3, padding='same', activation='relu'),
        Conv1D(384, 3, padding='same', activation='relu'),
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2),
        
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def demonstrate_noise_effect():
    """Демонстрация эффекта шума с высокими уровнями"""
    print("=== ДЕМОНСТРАЦИЯ ЭФФЕКТА ШУМА ===")
    print("Тестируем с очень высокими уровнями шума...")
    
    # Загружаем данные
    data, labels = load_spectral_data_7_species()
    if len(data) == 0:
        return
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Обучаем модель
    model = create_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Тестируем с очень высокими уровнями шума
    noise_levels = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  # 0%, 10%, 30%, 50%, 70%, 100%, 150%, 200%
    accuracies = []
    
    print("\n📊 РЕЗУЛЬТАТЫ С ВЫСОКИМИ УРОВНЯМИ ШУМА:")
    print("=" * 60)
    
    for noise_level in noise_levels:
        if noise_level == 0:
            X_test_noisy = X_test_cnn
        else:
            X_test_noisy = add_gaussian_noise_correct(X_test_cnn, noise_level)
        
        y_pred = model.predict(X_test_noisy, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        accuracies.append(accuracy)
        
        print(f"Шум {noise_level*100:3.0f}%: Точность {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # График деградации точности
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot([n*100 for n in noise_levels], accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)', fontsize=12)
    plt.ylabel('Точность', fontsize=12)
    plt.title('Деградация точности с шумом', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Матрица ошибок без шума
    y_pred = model.predict(X_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm_clean = confusion_matrix(y_test, y_pred_classes)
    
    plt.subplot(2, 2, 2)
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Матрица ошибок (без шума)', fontsize=12)
    plt.ylabel('Реальный класс')
    plt.xlabel('Предсказанный класс')
    
    # Матрица ошибок с высоким шумом (100%)
    X_test_noisy_high = add_gaussian_noise_correct(X_test_cnn, 1.0)
    y_pred_high = model.predict(X_test_noisy_high, verbose=0)
    y_pred_classes_high = np.argmax(y_pred_high, axis=1)
    cm_noisy = confusion_matrix(y_test, y_pred_classes_high)
    
    plt.subplot(2, 2, 3)
    sns.heatmap(cm_noisy, annot=True, fmt='d', cmap='Reds',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Матрица ошибок (100% шум)', fontsize=12)
    plt.ylabel('Реальный класс')
    plt.xlabel('Предсказанный класс')
    
    # Сравнение спектров
    plt.subplot(2, 2, 4)
    sample_idx = 0
    original_spectrum = X_test_cnn[sample_idx, :, 0]
    noisy_spectrum = X_test_noisy_high[sample_idx, :, 0]
    
    plt.plot(original_spectrum, label='Оригинал', linewidth=2)
    plt.plot(noisy_spectrum, label='100% шум', linewidth=2, alpha=0.7)
    plt.xlabel('Спектральный канал')
    plt.ylabel('Интенсивность')
    plt.title('Сравнение спектров')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ДЕМОНСТРАЦИЯ_ЭФФЕКТА_ШУМА.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ График сохранен: ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ДЕМОНСТРАЦИЯ_ЭФФЕКТА_ШУМА.png")
    
    # Анализ результатов
    print(f"\n📈 АНАЛИЗ:")
    print(f"Без шума: {accuracies[0]*100:.2f}%")
    print(f"С 100% шумом: {accuracies[5]*100:.2f}%")
    print(f"С 200% шумом: {accuracies[7]*100:.2f}%")
    
    if accuracies[7] < accuracies[0]:
        print("✅ Эффект шума обнаружен при высоких уровнях!")
    else:
        print("⚠️ Модель очень устойчива к шуму")

if __name__ == "__main__":
    demonstrate_noise_effect() 