import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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

def load_spring_data():
    """Загружает весенние данные для обучения"""
    base_path = "Спектры, весенний период, 7 видов"
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    print("Загрузка весенних данных для обучения...")
    
    for tree_type in tree_types:
        folder_path = os.path.join(base_path, tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"Найдено {len(excel_files)} весенних файлов для {tree_type}")
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    
                    if df.shape[1] >= 2:
                        spectrum_data = df.iloc[:, 1].values
                        
                        if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                            spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                            
                            if len(spectrum_data) > 10:
                                all_data.append(spectrum_data)
                                all_labels.append(tree_type)
                                
                except Exception as e:
                    print(f"Ошибка при чтении весеннего файла {file_path}: {e}")
                    continue
        else:
            print(f"Папка {folder_path} не найдена")
    
    print(f"Загружено {len(all_data)} весенних спектров")
    return all_data, all_labels

def load_summer_data():
    """Загружает летние данные для тестирования"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    print("Загрузка летних данных для тестирования...")
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"Найдено {len(excel_files)} летних файлов для {tree_type}")
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    
                    if df.shape[1] >= 2:
                        spectrum_data = df.iloc[:, 1].values
                        
                        if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                            spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                            
                            if len(spectrum_data) > 10:
                                all_data.append(spectrum_data)
                                all_labels.append(tree_type)
                                
                except Exception as e:
                    print(f"Ошибка при чтении летнего файла {file_path}: {e}")
                    continue
        else:
            print(f"Папка {folder_path} не найдена")
    
    print(f"Загружено {len(all_data)} летних спектров")
    return all_data, all_labels

def preprocess_data(train_data, train_labels, test_data, test_labels):
    """Предобрабатывает данные"""
    print("Предобработка данных...")
    
    # Находим минимальную длину среди всех спектров
    all_spectra = train_data + test_data
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до одинаковой длины
    processed_train_data = []
    for spectrum in train_data:
        truncated_spectrum = spectrum[:min_length]
        processed_train_data.append(truncated_spectrum)
    
    processed_test_data = []
    for spectrum in test_data:
        truncated_spectrum = spectrum[:min_length]
        processed_test_data.append(truncated_spectrum)
    
    # Преобразуем в numpy массивы
    X_train = np.array(processed_train_data, dtype=np.float32)
    X_test = np.array(processed_test_data, dtype=np.float32)
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    print(f"Форма обучающих данных: {X_train.shape}")
    print(f"Форма тестовых данных: {X_test.shape}")
    print(f"Количество классов: {len(np.unique(y_train))}")
    print(f"Классы: {label_encoder.classes_}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def create_model(input_shape, num_classes):
    """Создает модель нейронной сети"""
    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Строит графики обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Точность на весенних данных (обучение)', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Точность на летних данных (валидация)', linewidth=2)
    ax1.set_title('Точность модели: Весна → Лето', fontsize=14)
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График потерь
    ax2.plot(history.history['loss'], label='Потери на весенних данных (обучение)', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Потери на летних данных (валидация)', linewidth=2)
    ax2.set_title('Потери модели: Весна → Лето', fontsize=14)
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spring2summer_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, title):
    """Строит тепловую карту матрицы ошибок"""
    plt.figure(figsize=(10, 8))
    
    # Нормализуем матрицу ошибок (по строкам для получения вероятностей)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Вероятность'})
    
    plt.title(f'{title}\n(Обучение на весенних данных, тест на летних)', fontsize=14)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_spring2summer.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm_normalized

def analyze_performance(y_true, y_pred, class_names):
    """Анализирует производительность модели"""
    # Отчет о классификации
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print("\nДетальный отчет о классификации:")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    print("\nМатрица ошибок (абсолютные значения):")
    print(cm)
    
    # Нормализованная матрица ошибок
    cm_normalized = plot_confusion_matrix(cm, class_names, 'Матрица ошибок')
    
    print("\nМатрица ошибок (вероятности по строкам):")
    print("Каждая строка показывает распределение предсказаний для истинного класса")
    print("Сумма по каждой строке = 1.0")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        row_sum = np.sum(cm_normalized[i])
        print(f"{class_name:>10}: {cm_normalized[i]} | Сумма: {row_sum:.3f}")
    
    # Сохраняем результаты
    save_results(report, cm, cm_normalized, class_names)
    
    return report, cm, cm_normalized

def save_results(report, cm, cm_normalized, class_names):
    """Сохраняет результаты анализа"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем отчет о классификации
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report_spring2summer.csv')
    
    # Сохраняем матрицы ошибок в текстовый файл
    with open('spring2summer_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ: ОБУЧЕНИЕ НА ВЕСЕННИХ ДАННЫХ, ТЕСТ НА ЛЕТНИХ\n")
        f.write("="*80 + "\n\n")
        
        f.write("ОБЩАЯ ТОЧНОСТЬ:\n")
        f.write(f"Точность: {report['accuracy']:.4f}\n\n")
        
        f.write("МАТРИЦА ОШИБОК (абсолютные значения):\n")
        f.write("Строки - истинные классы, столбцы - предсказанные классы\n")
        header = "".join([f"{name:>12}" for name in class_names])
        f.write(f"{'':>12}" + header + "\n")
        for i, class_name in enumerate(class_names):
            row = "".join([f"{cm[i][j]:>12}" for j in range(len(class_names))])
            f.write(f"{class_name:>12}" + row + "\n")
        
        f.write("\nМАТРИЦА ОШИБОК (вероятности):\n")
        f.write("Каждая строка показывает распределение предсказаний для истинного класса\n")
        f.write("Сумма по каждой строке = 1.0\n")
        header = "".join([f"{name:>12}" for name in class_names])
        f.write(f"{'':>12}" + header + f"{'Сумма':>12}\n")
        for i, class_name in enumerate(class_names):
            row = "".join([f"{cm_normalized[i][j]:>12.3f}" for j in range(len(class_names))])
            row_sum = np.sum(cm_normalized[i])
            f.write(f"{class_name:>12}" + row + f"{row_sum:>12.3f}\n")
        
        f.write("\nВЕРОЯТНОСТИ ПРАВИЛЬНОЙ КЛАССИФИКАЦИИ ПО КЛАССАМ:\n")
        for i, class_name in enumerate(class_names):
            accuracy = cm_normalized[i][i]
            f.write(f"{class_name}: {accuracy:.3f} ({accuracy*100:.1f}%)\n")

def plot_metrics_comparison(report, class_names):
    """Строит график сравнения метрик по классам"""
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report[class_name][metric] for class_name in class_names]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Виды деревьев')
    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение метрик по классам\n(Обучение на весенних данных, тест на летних)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_spring2summer.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("КЛАССИФИКАЦИЯ СПЕКТРОВ ДЕРЕВЬЕВ: ВЕСНА → ЛЕТО")
    print("="*60)
    print("Обучение на весенних данных, тестирование на летних данных")
    print("="*60)
    
    # Загрузка данных
    train_data, train_labels = load_spring_data()
    test_data, test_labels = load_summer_data()
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Ошибка: Не удалось загрузить данные. Проверьте наличие файлов.")
        return
    
    # Предобработка данных
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(
        train_data, train_labels, test_data, test_labels
    )
    
    # Нормализация данных
    print("Нормализация данных...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Преобразование меток для обучения (используем sparse_categorical_crossentropy)
    # y_train и y_test уже в правильном формате
    
    # Создание модели
    model = create_model(X_train_scaled.shape[1], len(label_encoder.classes_))
    
    print("\nАрхитектура модели:")
    model.summary()
    
    # Обучение модели
    print(f"\nНачало обучения на {len(X_train)} весенних спектрах...")
    print(f"Валидация на {len(X_test)} летних спектрах...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_spring2summer_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=32,
        epochs=150,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Построение графиков обучения
    plot_training_history(history)
    
    # Финальная оценка модели
    print("\n" + "="*60)
    print("ФИНАЛЬНАЯ ОЦЕНКА МОДЕЛИ")
    print("="*60)
    
    final_loss, final_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Финальная точность на летних данных: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Предсказания
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Детальный анализ производительности
    report, cm, cm_normalized = analyze_performance(y_test, y_pred, label_encoder.classes_)
    
    # График сравнения метрик
    plot_metrics_comparison(report, label_encoder.classes_)
    
    # Сохранение модели и предобработчиков
    print("\nСохранение модели и предобработчиков...")
    import joblib
    joblib.dump(scaler, 'spring2summer_scaler.pkl')
    joblib.dump(label_encoder, 'spring2summer_label_encoder.pkl')
    model.save('spring2summer_model.h5')
    
    print("\n" + "="*60)
    print("ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print("Результаты сохранены в файлы:")
    print("- spring2summer_training_progress.png")
    print("- confusion_matrix_spring2summer.png")
    print("- metrics_spring2summer.png")
    print("- classification_report_spring2summer.csv")
    print("- spring2summer_results.txt")
    print("- best_spring2summer_model.keras")
    print("- spring2summer_model.h5")
    print("- spring2summer_scaler.pkl")
    print("- spring2summer_label_encoder.pkl")

if __name__ == "__main__":
    main() 