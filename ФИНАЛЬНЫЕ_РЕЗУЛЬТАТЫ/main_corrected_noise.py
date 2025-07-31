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
    """Загрузка данных для 7 видов из правильных папок"""
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

def load_spectral_data_20_species():
    """Загрузка данных для 20 видов из правильных папок"""
    base_path = "Исходные_данные/Спектры, весенний период, 20 видов"
    species_folders = [
        "береза", "дуб", "ель", "ель_голубая", "ива", "каштан", 
        "клен", "клен_ам", "липа", "лиственница", "орех", "осина", 
        "рябина", "сирень", "сосна", "тополь_бальзамический", 
        "тополь_черный", "туя", "черемуха", "ясень"
    ]
    
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
        # Первый блок свертки
        Conv1D(96, 11, strides=4, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        # Второй блок свертки
        Conv1D(256, 5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(3, strides=2),
        
        # Третий блок свертки
        Conv1D(384, 3, padding='same', activation='relu'),
        
        # Четвертый блок свертки
        Conv1D(384, 3, padding='same', activation='relu'),
        
        # Пятый блок свертки
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2),
        
        # Полносвязные слои
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

def extra_trees_2_species():
    """ExtraTrees для 2 видов (осина и сирень) с ПРАВИЛЬНЫМ шумом"""
    print("=== EXTRA TREES ДЛЯ 2 ВИДОВ (ОСИНА И СИРЕНЬ) - ИСПРАВЛЕННЫЙ ШУМ ===")
    
    # Загружаем данные для 20 видов
    data, labels = load_spectral_data_20_species()
    if len(data) == 0:
        return
    
    # Фильтруем только осину и сирень
    osina_mask = labels == 'осина'
    siren_mask = labels == 'сирень'
    
    if not np.any(osina_mask) or not np.any(siren_mask):
        print("Не найдены данные для осины или сирени!")
        return
    
    # Выбираем данные только для осины и сирени
    X_filtered = data[osina_mask | siren_mask]
    y_filtered = labels[osina_mask | siren_mask]
    
    print(f"Отобрано {len(X_filtered)} образцов (осина: {np.sum(osina_mask)}, сирень: {np.sum(siren_mask)})")
    
    # Предобработка
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_filtered)
    
    # Разделение данных (80% на обучение)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")
    
    # Обучение ExtraTrees
    model = ExtraTreesClassifier(
        n_estimators=200, 
        max_depth=20, 
        min_samples_split=5, 
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Тестирование с ПРАВИЛЬНЫМ шумом
    noise_levels = [0, 0.01, 0.05, 0.1]  # 0%, 1%, 5%, 10%
    
    for noise_level in noise_levels:
        if noise_level == 0:
            X_test_noisy = X_test
        else:
            X_test_noisy = add_gaussian_noise_correct(X_test, noise_level)
        
        y_pred = model.predict(X_test_noisy)
        y_pred_proba = model.predict_proba(X_test_noisy)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Точность с {noise_level*100}% шумом: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Создание Excel файла с результатами (без шума)
    results = []
    for i in range(len(X_test)):
        max_prob_idx = np.argmax(y_pred_proba[i])
        row = [0] * len(label_encoder.classes_)
        row[max_prob_idx] = 1
        results.append(row)
    
    df_results = pd.DataFrame(results, columns=label_encoder.classes_)
    output_path = 'ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ExtraTrees_2_вида/extra_trees_2_species_results_CORRECTED.xlsx'
    df_results.to_excel(output_path, index=False)
    print(f"Результаты сохранены в {output_path}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Матрица ошибок - ExtraTrees (2 вида) - ИСПРАВЛЕННЫЙ ШУМ')
    plt.ylabel('Реальный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ExtraTrees_2_вида/confusion_matrix_2_species_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ExtraTrees_2_вида/")

def alexnet_7_species():
    """1D-AlexNet для 7 видов с ПРАВИЛЬНЫМ шумом"""
    print("=== 1D-ALEXNET ДЛЯ 7 ВИДОВ - ИСПРАВЛЕННЫЙ ШУМ ===")
    
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
    
    print(f"Обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = create_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
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
    
    # Тестирование с ПРАВИЛЬНЫМ шумом
    noise_levels = [0, 0.01, 0.05, 0.1]  # 0%, 1%, 5%, 10%
    
    for noise_level in noise_levels:
        if noise_level == 0:
            X_test_noisy = X_test_cnn
        else:
            X_test_noisy = add_gaussian_noise_correct(X_test_cnn, noise_level)
        
        y_pred = model.predict(X_test_noisy)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"Точность с {noise_level*100}% шумом: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Матрица ошибок (без шума)
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Матрица ошибок - 1D-AlexNet (7 видов) - ИСПРАВЛЕННЫЙ ШУМ')
    plt.ylabel('Реальный класс')
    plt.xlabel('Предсказанный класс')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_7_видов/confusion_matrix_7_species_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Графики обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучающая точность')
    plt.plot(history.history['val_accuracy'], label='Валидационная точность')
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучающая ошибка')
    plt.plot(history.history['val_loss'], label='Валидационная ошибка')
    plt.title('Ошибка')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_7_видов/training_history_7_species_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_7_видов/")

def alexnet_20_species():
    """1D-AlexNet для 20 видов с ПРАВИЛЬНЫМ шумом"""
    print("=== 1D-ALEXNET ДЛЯ 20 ВИДОВ - ИСПРАВЛЕННЫЙ ШУМ ===")
    
    # Загружаем данные
    data, labels = load_spectral_data_20_species()
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
    
    print(f"Обучающая выборка: {len(X_train)}, тестовая: {len(X_test)}")
    
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = create_1d_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Тестирование с ПРАВИЛЬНЫМ шумом
    noise_levels = [0, 0.01, 0.05, 0.1]  # 0%, 1%, 5%, 10%
    
    for noise_level in noise_levels:
        if noise_level == 0:
            X_test_noisy = X_test_cnn
        else:
            X_test_noisy = add_gaussian_noise_correct(X_test_cnn, noise_level)
        
        y_pred = model.predict(X_test_noisy)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"Точность с {noise_level*100}% шумом: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Матрица ошибок (без шума)
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Матрица ошибок - 1D-AlexNet (20 видов) - ИСПРАВЛЕННЫЙ ШУМ')
    plt.ylabel('Реальный класс')
    plt.xlabel('Предсказанный класс')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_20_видов/confusion_matrix_20_species_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Графики обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучающая точность')
    plt.plot(history.history['val_accuracy'], label='Валидационная точность')
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучающая ошибка')
    plt.plot(history.history['val_loss'], label='Валидационная ошибка')
    plt.title('Ошибка')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_20_видов/training_history_20_species_CORRECTED.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Результаты сохранены в ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_20_видов/")

def main():
    print("=== ИСПРАВЛЕННАЯ ВЕРСИЯ С ПРАВИЛЬНЫМ ШУМОМ ===")
    
    # Устанавливаем seed для воспроизводимости
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Создаем папки если их нет
    import os
    os.makedirs('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ExtraTrees_2_вида', exist_ok=True)
    os.makedirs('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_7_видов', exist_ok=True)
    os.makedirs('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/AlexNet_20_видов', exist_ok=True)
    
    print("\n🔧 ИСПРАВЛЕНИЕ: Шум теперь добавляется правильно!")
    print("📊 Старая формула: std_dev = noise_level / 100.0 * np.std(data)")
    print("✅ Новая формула: std_dev = noise_level * np.std(data)")
    print("🎯 Результат: Шум в 100 раз сильнее и реалистичнее!")
    
    # 1. ExtraTrees для 2 видов
    extra_trees_2_species()
    
    # 2. AlexNet для 7 видов
    alexnet_7_species()
    
    # 3. AlexNet для 20 видов
    alexnet_20_species()
    
    print("\n=== ВСЕ ЭКСПЕРИМЕНТЫ С ИСПРАВЛЕННЫМ ШУМОМ ЗАВЕРШЕНЫ ===")
    print("Результаты сохранены в папке ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/")

if __name__ == "__main__":
    main() 