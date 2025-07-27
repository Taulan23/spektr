import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка для использования GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_data_from_folders():
    """Загружает данные из папок с Excel файлами"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    print("Загрузка данных...")
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"Найдено {len(excel_files)} файлов для {tree_type}")
            
            for file_path in excel_files:
                try:
                    # Чтение Excel файла
                    df = pd.read_excel(file_path)
                    
                    # Предполагаем, что данные в первых двух столбцах (волновая длина и интенсивность)
                    if df.shape[1] >= 2:
                        # Берем только числовые значения интенсивности
                        spectrum_data = df.iloc[:, 1].values
                        
                        # Проверяем на валидность данных
                        if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                            # Удаляем NaN значения
                            spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                            
                            if len(spectrum_data) > 10:  # Минимум 10 точек для спектра
                                all_data.append(spectrum_data)
                                all_labels.append(tree_type)
                                
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
                    continue
        else:
            print(f"Папка {folder_path} не найдена")
    
    print(f"Загружено {len(all_data)} спектров")
    return all_data, all_labels, tree_types

def preprocess_data(all_data, all_labels):
    """Предобрабатывает данные"""
    print("Предобработка данных...")
    
    # Находим минимальную длину спектра
    min_length = min(len(spectrum) for spectrum in all_data)
    print(f"Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до одинаковой длины
    processed_data = []
    for spectrum in all_data:
        # Берем первые min_length точек
        truncated_spectrum = spectrum[:min_length]
        processed_data.append(truncated_spectrum)
    
    # Преобразуем в numpy массив
    X = np.array(processed_data, dtype=np.float32)
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(all_labels)
    
    print(f"Форма данных: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    print(f"Классы: {label_encoder.classes_}")
    
    return X, y, label_encoder

def create_model(input_shape, num_classes):
    """Создает модель нейронной сети"""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Строит графики обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Точность на обучении')
    ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
    ax1.set_title('Точность модели')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    
    # График потерь
    ax2.plot(history.history['loss'], label='Потери на обучении')
    ax2.plot(history.history['val_loss'], label='Потери на валидации')
    ax2.set_title('Потери модели')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model_with_noise(model, test_data_scaled, test_labels_onehot, tree_types, noise_levels, n_realizations=10):
    """Тестирует модель с различными уровнями шума"""
    print("\nТестирование модели с различными уровнями шума...")
    
    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"Тестирование с уровнем шума: {noise_level * 100}%")
        print(f"{'='*60}")
        
        accuracies = []
        
        with tf.device('/GPU:0'):
            for i in range(n_realizations):
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, test_data_scaled.shape).astype(np.float32)
                    X_test_noisy = test_data_scaled + noise
                else:
                    X_test_noisy = test_data_scaled
                
                test_loss, test_accuracy = model.evaluate(X_test_noisy, test_labels_onehot, verbose=0)
                accuracies.append(test_accuracy)
                
                if i == 0:
                    y_pred = model.predict(X_test_noisy, verbose=0)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_test_classes = np.argmax(test_labels_onehot, axis=1)

        mean_accuracy = np.mean(accuracies)
        print(f"Средняя точность тестирования (шум {noise_level * 100}%): {mean_accuracy:.4f} ± {np.std(accuracies):.4f}")
        print(f"Отчет о классификации (шум {noise_level * 100}%):")
        print(classification_report(y_test_classes, y_pred_classes, target_names=tree_types, digits=3))
        
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        print("Матрица ошибок:")
        print(cm)
        
        print("Коэффициент ложных срабатываний (FPR) для каждого класса:")
        for i, tree in enumerate(tree_types):
            FP = cm.sum(axis=0)[i] - cm[i, i]
            TN = cm.sum() - cm.sum(axis=0)[i] - cm.sum(axis=1)[i] + cm[i, i]
            FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
            print(f"{tree}: {FPR:.3f}")

def save_model_and_scaler(model, scaler, label_encoder):
    """Сохраняет модель и предобработчики"""
    model.save('tree_classification_model.h5')
    
    # Сохраняем scaler и label_encoder
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    print("Модель и предобработчики сохранены!")

def main():
    """Основная функция"""
    print("Начало программы классификации спектров деревьев")
    print("="*60)
    
    # Загрузка данных
    all_data, all_labels, tree_types = load_data_from_folders()
    
    if len(all_data) == 0:
        print("Не удалось загрузить данные. Проверьте наличие Excel файлов в папках.")
        return
    
    # Предобработка данных
    X, y, label_encoder = preprocess_data(all_data, all_labels)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Преобразование меток в one-hot encoding
    y_train_onehot = keras.utils.to_categorical(y_train)
    y_test_onehot = keras.utils.to_categorical(y_test)
    
    # Создание модели
    model = create_model(X_train_scaled.shape[1], len(tree_types))
    
    print("\nАрхитектура модели:")
    model.summary()
    
    # Обучение модели
    print("\nНачало обучения...")
    
    # Callbacks для улучшения обучения
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
    
    history = model.fit(
        X_train_scaled, y_train_onehot,
        batch_size=32,
        epochs=100,
        validation_data=(X_test_scaled, y_test_onehot),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Построение графиков обучения
    plot_training_history(history)
    
    # Базовая оценка модели
    print("\nБазовая оценка модели:")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
    
    # Тестирование с различными уровнями шума
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
    test_model_with_noise(model, X_test_scaled, y_test_onehot, tree_types, noise_levels)
    
    # Сохранение модели
    save_model_and_scaler(model, scaler, label_encoder)
    
    print("\n" + "="*60)
    print("Программа завершена успешно!")

if __name__ == "__main__":
    main() 