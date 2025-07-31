import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
                    continue
    
    print(f"Загружено {len(all_data)} весенних спектров")
    return all_data, all_labels

def load_summer_data_with_new_maple():
    """Загружает летние данные + новые данные клена для тестирования"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    print("Загрузка летних данных для тестирования...")
    
    for tree_type in tree_types:
        if tree_type == 'клен':
            # Для клена загружаем оригинальные летние + новые данные
            
            # Оригинальные летние данные клена
            folder_path = os.path.join('.', tree_type)
            if os.path.exists(folder_path):
                excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
                print(f"Найдено {len(excel_files)} летних файлов для {tree_type} (оригинальные)")
                
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
                        continue
            
            # Новые данные клена из папки клен_ам
            new_maple_path = "клен_ам"
            if os.path.exists(new_maple_path):
                excel_files = glob.glob(os.path.join(new_maple_path, '*.xlsx'))
                print(f"Найдено {len(excel_files)} новых файлов для {tree_type} (клен_ам)")
                
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
                        continue
        else:
            # Для остальных видов - обычные летние данные
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
                        continue
    
    print(f"Загружено {len(all_data)} летних спектров (включая новые данные клена)")
    return all_data, all_labels

def extract_enhanced_features_v2(spectra):
    """Улучшенная версия извлечения признаков с фокусом на клен"""
    features = []
    
    # Ключевые каналы для дуба и клена (обновленные на основе анализа)
    oak_channels = list(range(151, 161))  # 151-160
    maple_channels = list(range(172, 182)) + list(range(179, 186)) + [258, 276, 286]  # объединенные каналы
    
    for spectrum in spectra:
        spectrum = np.array(spectrum)
        feature_vector = []
        
        # 1. Исходные спектральные значения (более частая выборка)
        feature_vector.extend(spectrum[::8])  # каждый 8-й канал (было 10)
        
        # 2. Расширенные статистические признаки
        feature_vector.extend([
            np.mean(spectrum),
            np.std(spectrum),
            np.median(spectrum),
            np.min(spectrum),
            np.max(spectrum),
            np.percentile(spectrum, 10),
            np.percentile(spectrum, 25),
            np.percentile(spectrum, 75),
            np.percentile(spectrum, 90),
            np.ptp(spectrum),  # размах
            np.var(spectrum),  # дисперсия
        ])
        
        # 3. Производные и изменения
        derivative = np.diff(spectrum)
        feature_vector.extend([
            np.mean(derivative),
            np.std(derivative),
            np.max(np.abs(derivative)),
            np.sum(derivative > 0),  # количество возрастающих точек
            np.sum(derivative < 0),  # количество убывающих точек
        ])
        
        # 4. Улучшенные признаки для дуба
        if len(spectrum) > max(oak_channels):
            oak_region = spectrum[oak_channels]
            feature_vector.extend([
                np.mean(oak_region),
                np.std(oak_region),
                np.max(oak_region),
                np.min(oak_region),
                np.median(oak_region),
                np.ptp(oak_region)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0, 0, 0])
        
        # 5. УСИЛЕННЫЕ признаки для клена (ключевое улучшение!)
        valid_maple_channels = [ch for ch in maple_channels if ch < len(spectrum)]
        if valid_maple_channels:
            maple_region = spectrum[valid_maple_channels]
            feature_vector.extend([
                np.mean(maple_region),
                np.std(maple_region),
                np.max(maple_region),
                np.min(maple_region),
                np.median(maple_region),
                np.ptp(maple_region),
                np.percentile(maple_region, 25),
                np.percentile(maple_region, 75),
                np.var(maple_region),
                np.sum(maple_region > np.mean(spectrum)),  # количество точек выше общего среднего
            ])
        else:
            feature_vector.extend([0] * 10)
        
        # 6. Спектральные моменты
        normalized_spectrum = spectrum / np.sum(spectrum) if np.sum(spectrum) > 0 else spectrum
        channels = np.arange(len(spectrum))
        
        # Центроид (средняя частота)
        centroid = np.sum(channels * normalized_spectrum) if np.sum(normalized_spectrum) > 0 else 0
        feature_vector.append(centroid)
        
        # Спектральная ширина
        if np.sum(normalized_spectrum) > 0:
            spread = np.sqrt(np.sum(((channels - centroid) ** 2) * normalized_spectrum))
        else:
            spread = 0
        feature_vector.append(spread)
        
        # Асимметрия и куртозис
        if np.std(spectrum) > 0:
            skewness = np.mean(((spectrum - np.mean(spectrum)) / np.std(spectrum)) ** 3)
            kurtosis = np.mean(((spectrum - np.mean(spectrum)) / np.std(spectrum)) ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        feature_vector.extend([skewness, kurtosis])
        
        # 7. Энергия в разных диапазонах (больше диапазонов)
        n_bands = 8  # было 5
        band_size = len(spectrum) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(spectrum))
            band_energy = np.sum(spectrum[start_idx:end_idx] ** 2)
            feature_vector.append(band_energy)
        
        features.append(feature_vector)
    
    return np.array(features)

def create_maple_focused_model(input_shape, num_classes):
    """Создает модель с особым фокусом на распознавание клена"""
    
    # Входной слой
    inputs = layers.Input(shape=(input_shape,))
    
    # Основная ветка обработки
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Разделение на специализированные пути
    
    # Путь для общей классификации
    general_path = layers.Dense(256, activation='relu', name='general_path')(x)
    general_path = layers.BatchNormalization()(general_path)
    general_path = layers.Dropout(0.3)(general_path)
    general_path = layers.Dense(128, activation='relu')(general_path)
    
    # Специальный путь для клена (критически важно!)
    maple_path = layers.Dense(128, activation='relu', name='maple_path')(x)
    maple_path = layers.BatchNormalization()(maple_path)
    maple_path = layers.Dropout(0.2)(maple_path)
    maple_path = layers.Dense(64, activation='relu')(maple_path)
    
    # Путь для проблемных видов (дуб)
    oak_path = layers.Dense(64, activation='relu', name='oak_path')(x)
    oak_path = layers.Dropout(0.2)(oak_path)
    
    # Объединение всех путей
    combined = layers.Concatenate()([general_path, maple_path, oak_path])
    
    # Финальные слои с attention
    attention = layers.Dense(combined.shape[-1], activation='sigmoid')(combined)
    attended = layers.Multiply()([combined, attention])
    
    # Выходной слой
    output = layers.Dense(128, activation='relu')(attended)
    output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_classes, activation='softmax')(output)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    # Компиляция с фокусированной функцией потерь
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # меньше learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def analyze_final_results(y_test, y_pred, class_names, model_name="Final Model"):
    """Детальный анализ финальных результатов"""
    print(f"\n" + "="*60)
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ - {model_name}")
    print("="*60)
    
    # Общая точность
    accuracy = np.mean(y_test == y_pred)
    print(f"Общая точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Отчет по классам
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("\nОтчет по классам:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("\nМатрица ошибок (нормализованная по строкам):")
    print("Каждая строка показывает распределение предсказаний для истинного класса")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        row_sum = np.sum(cm_normalized[i])
        accuracy_class = cm_normalized[i][i]
        print(f"{class_name:>8}: точность {accuracy_class:.3f} ({accuracy_class*100:.1f}%) | сумма: {row_sum:.3f}")
    
    # Особый фокус на проблемные виды
    print("\n🎯 ФОКУС НА ПРОБЛЕМНЫЕ ВИДЫ:")
    problematic = ['дуб', 'клен']
    for species in problematic:
        if species in class_names:
            idx = list(class_names).index(species)
            correct = cm_normalized[idx][idx]
            total_samples = cm[idx].sum()
            
            if species == 'клен':
                print(f"🍁 {species.upper()}: {correct:.3f} ({correct*100:.1f}%) из {total_samples} образцов")
                if correct > 0.5:
                    print("   ✅ ПРОРЫВ! Клен теперь распознается!")
                elif correct > 0.2:
                    print("   ⚡ ПРОГРЕСС! Значительное улучшение!")
                else:
                    print("   ❌ Все еще проблемы...")
            else:
                print(f"🌳 {species.upper()}: {correct:.3f} ({correct*100:.1f}%) из {total_samples} образцов")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Вероятность'})
    
    plt.title(f'{model_name}\nТочность: {accuracy:.3f}', fontsize=14)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'final_confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return report, cm, cm_normalized

def main():
    """Основная функция"""
    print("ФИНАЛЬНАЯ МОДЕЛЬ С НОВЫМИ ДАННЫМИ КЛЕНА")
    print("="*60)
    print("🍁 Включены новые данные клена из папки 'клен_ам'")
    print("="*60)
    
    # Загрузка данных
    train_data, train_labels = load_spring_data()
    test_data, test_labels = load_summer_data_with_new_maple()
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Ошибка: Не удалось загрузить данные.")
        return
    
    # Статистика по кленам
    maple_count = test_labels.count('клен')
    print(f"\n📊 Статистика по клену в тестовых данных: {maple_count} образцов")
    
    # Предобработка
    all_spectra = train_data + test_data
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    train_data_trimmed = [spectrum[:min_length] for spectrum in train_data]
    test_data_trimmed = [spectrum[:min_length] for spectrum in test_data]
    
    # Извлечение улучшенных признаков
    print("Извлечение улучшенных признаков...")
    X_train_features = extract_enhanced_features_v2(train_data_trimmed)
    X_test_features = extract_enhanced_features_v2(test_data_trimmed)
    
    print(f"Форма признаков: {X_train_features.shape}")
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    print(f"Обучающая выборка: {X_train_scaled.shape}")
    print(f"Тестовая выборка: {X_test_scaled.shape}")
    
    # Random Forest для сравнения
    print("\n1. Random Forest (базовая линия)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    
    analyze_final_results(y_test, rf_pred, label_encoder.classes_, "Random Forest")
    
    # Финальная нейронная сеть
    print("\n2. Финальная нейронная сеть с фокусом на клен...")
    final_model = create_maple_focused_model(X_train_scaled.shape[1], len(label_encoder.classes_))
    
    print("\nАрхитектура модели:")
    final_model.summary()
    
    # Callback'и для обучения
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=15, min_lr=1e-7, verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_final_maple_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Обучение
    history = final_model.fit(
        X_train_scaled, y_train,
        batch_size=32,
        epochs=200,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Финальная оценка
    final_loss, final_accuracy = final_model.evaluate(X_test_scaled, y_test, verbose=0)
    final_pred = np.argmax(final_model.predict(X_test_scaled, verbose=0), axis=1)
    
    analyze_final_results(y_test, final_pred, label_encoder.classes_, "Final Neural Network")
    
    # Сравнение результатов
    rf_accuracy = np.mean(y_test == rf_pred)
    
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*60)
    print(f"Random Forest:     {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    print(f"Final NN Model:    {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    improvement = final_accuracy - rf_accuracy
    if improvement > 0:
        print(f"Улучшение NN:      +{improvement:.4f} (+{improvement*100:.2f}%)")
    else:
        print(f"Разница:           {improvement:.4f} ({improvement*100:.2f}%)")
    
    # Сохранение
    print("\nСохранение моделей...")
    import joblib
    joblib.dump(scaler, 'final_scaler.pkl')
    joblib.dump(label_encoder, 'final_label_encoder.pkl') 
    joblib.dump(rf, 'final_rf_model.pkl')
    final_model.save('final_neural_network.keras')
    
    print("\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("Файлы сохранены:")
    print("- best_final_maple_model.keras")
    print("- final_neural_network.keras")
    print("- final_rf_model.pkl")
    print("- final_scaler.pkl")
    print("- final_label_encoder.pkl")

if __name__ == "__main__":
    main() 