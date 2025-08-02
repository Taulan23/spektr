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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Устанавливаем seeds для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

def load_real_spectral_data():
    """Загрузка РЕАЛЬНЫХ спектральных данных из Excel файлов"""
    print("🔍 Загрузка РЕАЛЬНЫХ спектральных данных...")
    
    # Маппинг папок к названиям видов
    species_mapping = {
        'береза': 'береза',
        'дуб': 'дуб', 
        'ель': 'ель',
        'клен': 'клен',
        'липа': 'липа',
        'осина': 'осина',
        'сосна': 'сосна'
    }
    
    all_data = []
    all_labels = []
    successful_loads = 0
    failed_loads = 0
    
    for folder_name, species_name in species_mapping.items():
        folder_path = os.path.join('Исходные_данные', folder_name)
        if not os.path.exists(folder_path):
            print(f"❌ Папка {folder_path} не найдена")
            continue
            
        print(f"📁 Обработка папки: {folder_name} -> {species_name}")
        
        # Получаем все Excel файлы в папке
        excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        print(f"   Найдено {len(excel_files)} Excel файлов")
        
        species_count = 0
        for file_path in excel_files[:30]:  # Ограничиваем до 30 файлов на вид
            try:
                # Читаем Excel файл
                df = pd.read_excel(file_path)
                
                # Берём все строки второй колонки (интенсивности)
                if df.shape[1] >= 2 and df.shape[0] >= 50:  # Минимум 50 точек
                    # Вторая колонка содержит интенсивности
                    spectrum = df.iloc[:, 1].values
                    
                    # Удаляем NaN и inf значения
                    spectrum = spectrum[~np.isnan(spectrum)]
                    spectrum = spectrum[~np.isinf(spectrum)]
                    
                    if len(spectrum) > 50:  # Проверяем, что осталось достаточно данных
                        # Нормализуем спектр
                        if np.std(spectrum) > 0:
                            # Приводим к длине 300 точек
                            if len(spectrum) > 300:
                                spectrum = spectrum[:300]
                            elif len(spectrum) < 300:
                                # Дополняем средним значением
                                spectrum = np.pad(spectrum, (0, 300 - len(spectrum)), 'mean')
                            
                            all_data.append(spectrum)
                            all_labels.append(species_name)
                            species_count += 1
                            successful_loads += 1
                            
            except Exception as e:
                failed_loads += 1
                continue
        
        print(f"   ✅ Загружено {species_count} образцов для {species_name}")
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА ЗАГРУЗКИ:")
    print(f"✅ Успешно загружено: {successful_loads} образцов")
    print(f"❌ Ошибок загрузки: {failed_loads}")
    
    if len(all_data) == 0:
        print("❌ НЕ УДАЛОСЬ ЗАГРУЗИТЬ ДАННЫЕ!")
        return None, None
    
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"📈 Форма данных: {X.shape}")
    print(f"🏷️ Уникальные виды: {np.unique(y)}")
    
    # Статистика по видам
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО ВИДАМ:")
    for species, count in zip(unique, counts):
        print(f"   {species}: {count} образцов")
    
    return X, y

def create_real_alexnet(input_shape, num_classes):
    """Создание упрощённой AlexNet архитектуры для РЕАЛЬНЫХ данных"""
    model = Sequential([
        # Группа 1
        Conv1D(32, 10, strides=2, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2, strides=2),
        
        # Группа 2
        Conv1D(64, 5, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2, strides=2),
        
        # Группа 3
        Conv1D(128, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2, strides=2),
        
        # Полносвязные слои
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def add_gaussian_noise(data, noise_level):
    """Добавление гауссового шума к данным"""
    if noise_level == 0:
        return data
    
    # Вычисляем стандартное отклонение для каждого образца отдельно
    noise = np.zeros_like(data)
    for i in range(data.shape[0]):
        std_dev = np.std(data[i])
        noise[i] = np.random.normal(0, noise_level * std_dev, data[i].shape)
    
    return data + noise

def plot_confusion_matrices_with_precision(matrices_data, class_names, save_path, precision=7):
    """Создание матриц ошибок с заданной точностью"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, data in enumerate(matrices_data):
        noise_level = data['noise_level']
        cm_normalized = data['matrix']
        accuracy = data['accuracy']
        
        # Создаём аннотации с заданной точностью
        annotations = []
        for row in range(cm_normalized.shape[0]):
            ann_row = []
            for col in range(cm_normalized.shape[1]):
                value = cm_normalized[row, col]
                ann_row.append(f"{value:.{precision}f}")
            annotations.append(ann_row)
        
        # Создаём heatmap
        sns.heatmap(cm_normalized, 
                   annot=annotations,
                   fmt='',  # Используем собственные аннотации
                   cmap='Blues',
                   cbar=True,
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=axes[i],
                   square=True)
        
        axes[i].set_title(f'ШУМ: {noise_level:.1f}% | ТОЧНОСТЬ: {accuracy*100:.4f}%', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Предсказанный класс', fontsize=12)
        axes[i].set_ylabel('Истинный класс', fontsize=12)
        
        # Поворачиваем метки
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_matrices(matrices_data, class_names, save_path, precision=7):
    """Сохранение детальных матриц в текстовый файл"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("ДЕТАЛЬНЫЕ МАТРИЦЫ ОШИБОК С ВЫСОКОЙ ТОЧНОСТЬЮ\n")
        f.write("=" * 80 + "\n\n")
        
        for data in matrices_data:
            noise_level = data['noise_level']
            cm_normalized = data['matrix']
            accuracy = data['accuracy']
            
            f.write(f"ШУМ: {noise_level:.1f}% | ТОЧНОСТЬ: {accuracy*100:.7f}%\n")
            f.write("-" * 70 + "\n")
            
            # Заголовок таблицы
            header = "        "
            for class_name in class_names:
                header += f"{class_name:>12s} "
            f.write(header + "\n")
            
            # Строки матрицы
            for i, true_class in enumerate(class_names):
                row = f"{true_class:>8s}"
                for j in range(len(class_names)):
                    value = cm_normalized[i, j]
                    row += f" {value:.{precision}f}"
                f.write(row + "\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

def main():
    """Основная функция для обучения и тестирования модели"""
    print("🚀 НАЧАЛО ОБУЧЕНИЯ С РЕАЛЬНЫМИ ДАННЫМИ")
    print("=" * 60)
    
    # Загружаем реальные данные
    X, y = load_real_spectral_data()
    
    if X is None:
        print("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    # Проверяем качество данных
    print(f"\n🔍 АНАЛИЗ ЗАГРУЖЕННЫХ ДАННЫХ:")
    print(f"Форма данных: {X.shape}")
    print(f"Диапазон значений: [{np.min(X):.3f}, {np.max(X):.3f}]")
    print(f"Среднее значение: {np.mean(X):.3f}")
    print(f"Стандартное отклонение: {np.std(X):.3f}")
    
    # Проверяем различия между видами
    unique_species = np.unique(y)
    print(f"\n📊 СТАТИСТИКА ПО ВИДАМ:")
    for species in unique_species:
        species_data = X[y == species]
        print(f"{species:>8s}: {len(species_data):>3d} образцов, "
              f"среднее={np.mean(species_data):.3f}, std={np.std(species_data):.3f}")
    
    # Минимальное количество образцов на класс
    unique, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)
    
    if min_samples < 10:
        print(f"⚠️ Мало данных! Минимум образцов на класс: {min_samples}")
        print("Используем все доступные данные...")
    
    # Подготавливаем данные
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"После нормализации: [{np.min(X_scaled):.3f}, {np.max(X_scaled):.3f}], std={np.std(X_scaled):.3f}")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные
    test_size = 0.3 if len(X) > 50 else 0.2  # Адаптивный размер тестовой выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # Подготавливаем для CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"\n📋 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"Обучающая выборка: {X_train_cnn.shape}")
    print(f"Тестовая выборка: {X_test_cnn.shape}")
    print(f"Классы: {len(label_encoder.classes_)}")
    
    # Создаём модель
    print(f"\n🧠 СОЗДАНИЕ МОДЕЛИ...")
    model = create_real_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    print(f"Параметров модели: {model.count_params():,}")
    
    # Настраиваем callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Обучение
    print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ...")
    batch_size = min(16, len(X_train) // 4)  # Адаптивный размер батча
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print(f"✅ Обучение завершено за {len(history.history['accuracy'])} эпох")
    
    # Тестирование с разными уровнями шума
    print(f"\n🔊 ТЕСТИРОВАНИЕ С РАЗЛИЧНЫМИ УРОВНЯМИ ШУМА:")
    print("=" * 60)
    
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    results = []
    
    for noise_level in noise_levels:
        print(f"\n--- Тестирование с шумом {noise_level*100:.1f}% ---")
        
        # Добавляем шум к тестовым данным
        X_test_noisy = add_gaussian_noise(X_test_cnn, noise_level)
        
        # Предсказания
        y_pred_proba = model.predict(X_test_noisy, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred_classes)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        cm_normalized = np.nan_to_num(cm_normalized)
        
        results.append({
            'noise_level': noise_level * 100,
            'accuracy': accuracy,
            'matrix': cm_normalized,
            'raw_matrix': cm
        })
        
        print(f"Точность: {accuracy*100:.4f}%")
        
        # Распределение предсказаний
        pred_counts = np.bincount(y_pred_classes, minlength=len(label_encoder.classes_))
        print("Распределение предсказаний:")
        for i, (species, count) in enumerate(zip(label_encoder.classes_, pred_counts)):
            print(f"  {species}: {count} предсказаний")
    
    # Создаём папку для результатов
    output_dir = "ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/РЕАЛЬНАЯ_ALEXNET_7_ВИДОВ"
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем матрицы с высокой точностью
    matrices_path = os.path.join(output_dir, "real_alexnet_confusion_matrices_7_digits.png")
    plot_confusion_matrices_with_precision(results, label_encoder.classes_, matrices_path, precision=7)
    
    # Сохраняем детальные матрицы в текстовый файл
    detailed_path = os.path.join(output_dir, "detailed_matrices_7_digits.txt")
    save_detailed_matrices(results, label_encoder.classes_, detailed_path, precision=7)
    
    # Сохраняем отчёт
    report_path = os.path.join(output_dir, "real_alexnet_classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ПО КЛАССИФИКАЦИИ С РЕАЛЬНЫМИ ДАННЫМИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Всего образцов: {len(X)}\n")
        f.write(f"Обучающая выборка: {len(X_train)}\n")
        f.write(f"Тестовая выборка: {len(X_test)}\n")
        f.write(f"Количество классов: {len(label_encoder.classes_)}\n")
        f.write(f"Классы: {', '.join(label_encoder.classes_)}\n")
        f.write(f"Параметров модели: {model.count_params():,}\n")
        f.write(f"Эпох обучения: {len(history.history['accuracy'])}\n\n")
        
        f.write("РЕЗУЛЬТАТЫ ПО ШУМУ:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"Шум {result['noise_level']:5.1f}%: точность {result['accuracy']*100:7.4f}%\n")
    
    print(f"\n✅ ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {output_dir}")
    print(f"📊 Матрицы: {matrices_path}")
    print(f"📄 Детали: {detailed_path}")
    print(f"📋 Отчёт: {report_path}")
    
    # Финальная статистика
    base_accuracy = results[0]['accuracy']
    final_accuracy = results[-1]['accuracy']
    accuracy_drop = (base_accuracy - final_accuracy) * 100
    
    print(f"\n🎯 ФИНАЛЬНАЯ СТАТИСТИКА:")
    print(f"Базовая точность (0% шума): {base_accuracy*100:.4f}%")
    print(f"Точность с шумом 10%: {final_accuracy*100:.4f}%")
    print(f"Снижение точности: {accuracy_drop:.4f} процентных пунктов")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\n🎉 ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")