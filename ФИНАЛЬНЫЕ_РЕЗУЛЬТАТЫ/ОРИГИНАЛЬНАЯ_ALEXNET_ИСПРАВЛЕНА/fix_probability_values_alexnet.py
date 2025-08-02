import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
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
        # Используем весенние данные с большей длиной спектра
        folder_path = os.path.join('Исходные_данные', 'Спектры, весенний период, 7 видов', folder_name)
        if not os.path.exists(folder_path):
            print(f"❌ Папка {folder_path} не найдена")
            continue
            
        print(f"📁 Обработка папки: {folder_name} -> {species_name}")
        
        # Получаем все Excel файлы в папке
        excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        print(f"   Найдено {len(excel_files)} Excel файлов")
        
        species_count = 0
        for file_path in excel_files[:50]:  # Увеличим до 50 файлов на вид
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

def create_original_alexnet(input_shape, num_classes):
    """ОРИГИНАЛЬНАЯ модифицированная AlexNet архитектура с дропаутами (МИНИМАЛЬНЫЕ ИЗМЕНЕНИЯ!)"""
    model = Sequential([
        # Группа 1 - уменьшим stride для 300 точек
        Conv1D(10, 50, strides=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(3, strides=2),
        
        # Группа 2 - уменьшим размер ядра
        Conv1D(20, 25, strides=1, activation='relu'),
        MaxPooling1D(3, strides=2),
        
        # Группа 3
        Conv1D(50, 2, strides=1, activation='relu'),
        Conv1D(50, 2, strides=1, activation='relu'),
        Conv1D(25, 2, strides=1, activation='relu'),
        MaxPooling1D(3, strides=2),
        
        # Полносвязные слои
        Flatten(),
        Dense(200, activation='relu'),
        Dropout(0.5),
        Dense(200, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Стандартная скорость
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

def plot_confusion_matrices_like_table(matrices_data, class_names, save_path):
    """Создание матриц ошибок с вероятностями как в таблице"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, data in enumerate(matrices_data):
        noise_level = data['noise_level']
        cm_normalized = data['matrix']
        accuracy = data['accuracy']
        
        # Создаём аннотации с 3 знаками как в таблице
        annotations = []
        for row in range(cm_normalized.shape[0]):
            ann_row = []
            for col in range(cm_normalized.shape[1]):
                value = cm_normalized[row, col]
                ann_row.append(f"{value:.3f}")
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
        
        axes[i].set_title(f'δ = {noise_level:.0f}% | Точность: {accuracy:.3f}', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Предсказанный класс', fontsize=12)
        axes[i].set_ylabel('Истинный класс', fontsize=12)
        
        # Поворачиваем метки
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_table_format_matrices(matrices_data, class_names, save_path):
    """Сохранение матриц в формате таблицы как у вас"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Результаты классификации по гиперспектральным данным. Разрешение 2 нм\n")
        f.write("=" * 80 + "\n\n")
        
        # Заголовок таблицы
        header = f"{'Порода':>10s} |"
        for data in matrices_data:
            noise_level = data['noise_level']
            header += f" δ = {noise_level:2.0f}% |" * 2
        f.write(header + "\n")
        
        # Подзаголовки
        subheader = f"{'дерева':>10s} |"
        for data in matrices_data:
            subheader += f"   Pd    |   Pf    |"
        f.write(subheader + "\n")
        f.write("-" * len(subheader) + "\n")
        
        # Строки для каждого вида
        for i, true_class in enumerate(class_names):
            row = f"{true_class:>10s} |"
            for data in matrices_data:
                cm_normalized = data['matrix']
                pd_value = cm_normalized[i, i]  # Правильная классификация (диагональ)
                # Pf - вероятность ложной тревоги (среднее по строке исключая диагональ)
                pf_values = [cm_normalized[j, i] for j in range(len(class_names)) if j != i]
                pf_value = np.mean(pf_values) if pf_values else 0.0
                
                row += f" {pd_value:6.3f} | {pf_value:6.3f} |"
            f.write(row + "\n")

def main():
    """Основная функция для обучения и тестирования модели"""
    print("🚀 НАЧАЛО ОБУЧЕНИЯ С ОРИГИНАЛЬНОЙ АРХИТЕКТУРОЙ")
    print("=" * 60)
    
    # Загружаем реальные данные
    X, y = load_real_spectral_data()
    
    if X is None:
        print("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    # Подготавливаем данные БЕЗ агрессивной нормализации
    print(f"\n🔍 ИСПОЛЬЗУЕМ МЯГКУЮ НОРМАЛИЗАЦИЮ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"После нормализации: [{np.min(X_scaled):.3f}, {np.max(X_scaled):.3f}], std={np.std(X_scaled):.3f}")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, 
        test_size=0.25,  # Меньшая тестовая выборка
        random_state=42, 
        stratify=y_encoded
    )
    
    # Подготавливаем для CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"\n📋 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"Обучающая выборка: {X_train_cnn.shape}")
    print(f"Тестовая выборка: {X_test_cnn.shape}")
    
    # Проверяем размер для архитектуры
    if X_train_cnn.shape[1] < 200:
        print("❌ Недостаточно точек для оригинальной архитектуры")
        return
    
    # Создаём ОРИГИНАЛЬНУЮ модель
    print(f"\n🧠 СОЗДАНИЕ ОРИГИНАЛЬНОЙ МОДЕЛИ...")
    model = create_original_alexnet((X_train_cnn.shape[1], 1), len(label_encoder.classes_))
    
    print(f"Параметров модели: {model.count_params():,}")
    
    # Настраиваем callbacks с БОЛЬШИМ терпением
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=25,  # Увеличенное терпение
        restore_best_weights=True,
        verbose=1
    )
    
    # Обучение с УВЕЛИЧЕННЫМ batch_size и эпохами
    print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ...")
    batch_size = 32  # Увеличенный batch size
    
    history = model.fit(
        X_train_cnn, y_train,
        epochs=150,  # Больше эпох
        batch_size=batch_size,
        validation_split=0.15,  # Меньше валидации, больше для обучения
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(f"✅ Обучение завершено за {len(history.history['accuracy'])} эпох")
    
    # Тестирование с разными уровнями шума
    print(f"\n🔊 ТЕСТИРОВАНИЕ С РАЗЛИЧНЫМИ УРОВНЯМИ ШУМА:")
    print("=" * 60)
    
    noise_levels = [1.0, 5.0, 10.0]  # Как в вашей таблице
    results = []
    
    for noise_level in noise_levels:
        print(f"\n--- Тестирование с шумом {noise_level:.1f}% ---")
        
        # Добавляем шум к тестовым данным
        X_test_noisy = add_gaussian_noise(X_test_cnn, noise_level/100.0)
        
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
            'noise_level': noise_level,
            'accuracy': accuracy,
            'matrix': cm_normalized,
            'raw_matrix': cm
        })
        
        print(f"Точность: {accuracy:.3f}")
        
        # Показываем главную диагональ (Pd значения)
        print("Pd значения (правильная классификация):")
        for i, species in enumerate(label_encoder.classes_):
            pd_value = cm_normalized[i, i]
            print(f"  {species}: {pd_value:.3f}")
    
    # Создаём папку для результатов
    output_dir = "ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/ОРИГИНАЛЬНАЯ_ALEXNET_ИСПРАВЛЕНА"
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем матрицы в стиле таблицы
    matrices_path = os.path.join(output_dir, "alexnet_confusion_matrices_table_style.png")
    plot_confusion_matrices_like_table(results, label_encoder.classes_, matrices_path)
    
    # Сохраняем в формате таблицы как у вас
    table_path = os.path.join(output_dir, "classification_results_table_format.txt")
    save_table_format_matrices(results, label_encoder.classes_, table_path)
    
    # Сохраняем отчёт
    report_path = os.path.join(output_dir, "original_alexnet_fixed_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ПО ОРИГИНАЛЬНОЙ ALEXNET С ИСПРАВЛЕННЫМИ ПАРАМЕТРАМИ\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"ИЗМЕНЕНИЯ (БЕЗ ИЗМЕНЕНИЯ АРХИТЕКТУРЫ):\n")
        f.write(f"- Увеличен batch_size: 32\n")
        f.write(f"- Увеличены эпохи: 150\n")
        f.write(f"- Увеличено терпение: 25\n")
        f.write(f"- Уменьшена валидация: 15%\n")
        f.write(f"- Мягкая нормализация: StandardScaler\n\n")
        f.write(f"РЕЗУЛЬТАТЫ ПО ШУМУ:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"δ = {result['noise_level']:4.1f}%: точность {result['accuracy']:.3f}\n")
    
    print(f"\n✅ ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {output_dir}")
    print(f"📊 Матрицы: {matrices_path}")
    print(f"📄 Таблица: {table_path}")
    print(f"📋 Отчёт: {report_path}")
    
    # Финальная статистика
    print(f"\n🎯 ФИНАЛЬНЫЕ Pd ЗНАЧЕНИЯ:")
    for result in results:
        print(f"\nδ = {result['noise_level']:.0f}%:")
        for i, species in enumerate(label_encoder.classes_):
            pd_value = result['matrix'][i, i]
            print(f"  {species}: Pd = {pd_value:.3f}")
    
    return results

if __name__ == "__main__":
    results = main()
    print("\n🎉 ОБУЧЕНИЕ С ОРИГИНАЛЬНОЙ АРХИТЕКТУРОЙ ЗАВЕРШЕНО!")