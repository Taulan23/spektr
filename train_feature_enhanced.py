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
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
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
                    continue
    
    print(f"Загружено {len(all_data)} летних спектров")
    return all_data, all_labels

def extract_enhanced_features(spectra):
    """Извлекает расширенные признаки из спектров"""
    features = []
    
    # Ключевые каналы для дуба и клена (из анализа)
    oak_channels = list(range(151, 161))  # 151-160
    maple_channels = list(range(179, 186)) + [258, 276, 286]  # 179-185 + дополнительные
    
    for spectrum in spectra:
        spectrum = np.array(spectrum)
        feature_vector = []
        
        # 1. Исходные спектральные значения (подвыборка)
        feature_vector.extend(spectrum[::10])  # каждый 10-й канал
        
        # 2. Статистические признаки
        feature_vector.extend([
            np.mean(spectrum),
            np.std(spectrum),
            np.median(spectrum),
            np.min(spectrum),
            np.max(spectrum),
            np.percentile(spectrum, 25),
            np.percentile(spectrum, 75),
            np.ptp(spectrum)  # размах
        ])
        
        # 3. Производные (изменения)
        derivative = np.diff(spectrum)
        feature_vector.extend([
            np.mean(derivative),
            np.std(derivative),
            np.max(np.abs(derivative))
        ])
        
        # 4. Специфические признаки для дуба
        if len(spectrum) > max(oak_channels):
            oak_region = spectrum[oak_channels]
            feature_vector.extend([
                np.mean(oak_region),
                np.std(oak_region),
                np.max(oak_region),
                np.min(oak_region)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0])
        
        # 5. Специфические признаки для клена
        valid_maple_channels = [ch for ch in maple_channels if ch < len(spectrum)]
        if valid_maple_channels:
            maple_region = spectrum[valid_maple_channels]
            feature_vector.extend([
                np.mean(maple_region),
                np.std(maple_region),
                np.max(maple_region),
                np.min(maple_region)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0])
        
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
        
        # 7. Энергия в разных диапазонах
        n_bands = 5
        band_size = len(spectrum) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(spectrum))
            band_energy = np.sum(spectrum[start_idx:end_idx] ** 2)
            feature_vector.append(band_energy)
        
        features.append(feature_vector)
    
    return np.array(features)

def create_enhanced_model(input_shape, num_classes):
    """Создает улучшенную модель с вниманием к важным признакам"""
    
    # Входной слой
    inputs = layers.Input(shape=(input_shape,))
    
    # Основная ветка
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Разделение на специализированные ветки
    
    # Ветка для общей классификации
    general_branch = layers.Dense(64, activation='relu', name='general')(x)
    general_branch = layers.Dropout(0.2)(general_branch)
    
    # Ветка для проблемных видов (дуб, клен)
    problematic_branch = layers.Dense(32, activation='relu', name='problematic')(x)
    problematic_branch = layers.Dropout(0.2)(problematic_branch)
    
    # Объединяем ветки
    combined = layers.Concatenate()([general_branch, problematic_branch])
    
    # Финальные слои
    output = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(num_classes, activation='softmax')(output)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    # Компиляция с пользовательской функцией потерь
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compare_models(X_train, X_test, y_train, y_test, class_names):
    """Сравнивает разные модели машинного обучения"""
    results = {}
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РАЗЛИЧНЫХ МОДЕЛЕЙ")
    print("="*60)
    
    # 1. Random Forest
    print("\n1. Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = np.mean(rf_pred == y_test)
    results['Random Forest'] = rf_accuracy
    
    print(f"Random Forest точность: {rf_accuracy:.4f}")
    print("\nОтчет Random Forest:")
    print(classification_report(y_test, rf_pred, target_names=class_names, digits=3))
    
    # 2. SVM
    print("\n2. SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = np.mean(svm_pred == y_test)
    results['SVM'] = svm_accuracy
    
    print(f"SVM точность: {svm_accuracy:.4f}")
    print("\nОтчет SVM:")
    print(classification_report(y_test, svm_pred, target_names=class_names, digits=3))
    
    return results, rf, svm, rf_pred, svm_pred

def plot_feature_importance(rf_model, feature_names):
    """Строит график важности признаков"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Важность признаков (Random Forest)')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]])
    plt.xticks(range(min(20, len(importances))), 
               [f'F{indices[i]}' for i in range(min(20, len(importances)))], 
               rotation=45)
    plt.xlabel('Признаки')
    plt.ylabel('Важность')
    plt.tight_layout()
    plt.savefig('feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_confusion_matrices(y_test, tf_pred, rf_pred, svm_pred, class_names):
    """Анализирует матрицы ошибок разных моделей"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = [
        ('TensorFlow', tf_pred),
        ('Random Forest', rf_pred), 
        ('SVM', svm_pred)
    ]
    
    for idx, (model_name, pred) in enumerate(models):
        cm = confusion_matrix(y_test, pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx])
        axes[idx].set_title(f'{model_name}\nТочность: {np.mean(pred == y_test):.3f}')
        axes[idx].set_xlabel('Предсказанный класс')
        if idx == 0:
            axes[idx].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('models_comparison_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("УЛУЧШЕННАЯ КЛАССИФИКАЦИЯ С ВЫДЕЛЕНИЕМ ПРИЗНАКОВ")
    print("="*60)
    
    # Загрузка данных
    train_data, train_labels = load_spring_data()
    test_data, test_labels = load_summer_data()
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Ошибка: Не удалось загрузить данные.")
        return
    
    # Предобработка - приведение к одинаковой длине
    all_spectra = train_data + test_data
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    train_data_trimmed = [spectrum[:min_length] for spectrum in train_data]
    test_data_trimmed = [spectrum[:min_length] for spectrum in test_data]
    
    # Извлечение расширенных признаков
    print("Извлечение расширенных признаков...")
    X_train_features = extract_enhanced_features(train_data_trimmed)
    X_test_features = extract_enhanced_features(test_data_trimmed)
    
    print(f"Форма признаков: {X_train_features.shape}")
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    # Нормализация признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    print(f"Обучающая выборка: {X_train_scaled.shape}")
    print(f"Тестовая выборка: {X_test_scaled.shape}")
    
    # Сравнение моделей
    model_results, rf_model, svm_model, rf_pred, svm_pred = compare_models(
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder.classes_
    )
    
    # Обучение улучшенной нейронной сети
    print("\n3. Улучшенная нейронная сеть...")
    enhanced_model = create_enhanced_model(X_train_scaled.shape[1], len(label_encoder.classes_))
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20, restore_best_weights=True
    )
    
    history = enhanced_model.fit(
        X_train_scaled, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Оценка улучшенной модели
    tf_loss, tf_accuracy = enhanced_model.evaluate(X_test_scaled, y_test, verbose=0)
    tf_pred = np.argmax(enhanced_model.predict(X_test_scaled, verbose=0), axis=1)
    
    print(f"Улучшенная нейронная сеть точность: {tf_accuracy:.4f}")
    print("\nОтчет улучшенной нейронной сети:")
    print(classification_report(y_test, tf_pred, target_names=label_encoder.classes_, digits=3))
    
    model_results['Enhanced Neural Network'] = tf_accuracy
    
    # Анализ важности признаков
    print("\nАнализ важности признаков...")
    plot_feature_importance(rf_model, [f'Feature_{i}' for i in range(X_train_scaled.shape[1])])
    
    # Сравнение матриц ошибок
    analyze_confusion_matrices(y_test, tf_pred, rf_pred, svm_pred, label_encoder.classes_)
    
    # Итоговые результаты
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    for model_name, accuracy in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:>25}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Сохранение лучшей модели
    best_model_name = max(model_results.items(), key=lambda x: x[1])[0]
    print(f"\nЛучшая модель: {best_model_name}")
    
    if best_model_name == 'Enhanced Neural Network':
        enhanced_model.save('best_enhanced_model.keras')
    
    import joblib
    joblib.dump(scaler, 'enhanced_scaler.pkl')
    joblib.dump(label_encoder, 'enhanced_label_encoder.pkl')
    joblib.dump(rf_model, 'enhanced_rf_model.pkl')
    joblib.dump(svm_model, 'enhanced_svm_model.pkl')
    
    print("\nМодели сохранены!")

if __name__ == "__main__":
    main() 