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
from sklearn.model_selection import train_test_split
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
    
    for tree_type in tree_types:
        folder_path = os.path.join(base_path, tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            
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
                except Exception:
                    continue
    
    return all_data, all_labels

def load_summer_data_original_only():
    """Загружает только оригинальные летние данные (БЕЗ клен_ам)"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            
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
                except Exception:
                    continue
    
    return all_data, all_labels

def extract_enhanced_maple_features(spectra):
    """Извлекает улучшенные признаки с фокусом на клен"""
    features = []
    
    # Расширенные ключевые каналы для клена (из детального анализа)
    maple_channels_1 = list(range(172, 186))  # Основная область
    maple_channels_2 = list(range(258, 287))  # Вторичная область  
    maple_channels_3 = [179, 180, 181, 258, 276, 286]  # Ключевые пики
    oak_channels = list(range(151, 161))      # Для дуба
    
    for spectrum in spectra:
        spectrum = np.array(spectrum)
        feature_vector = []
        
        # 1. Базовые статистические признаки
        feature_vector.extend([
            np.mean(spectrum),
            np.std(spectrum),
            np.median(spectrum),
            np.max(spectrum),
            np.min(spectrum),
            np.ptp(spectrum),  # размах
            np.var(spectrum),  # дисперсия
        ])
        
        # 2. Расширенные квантили
        feature_vector.extend([
            np.percentile(spectrum, 10),
            np.percentile(spectrum, 25),
            np.percentile(spectrum, 75),
            np.percentile(spectrum, 90),
        ])
        
        # 3. Производные (1-я и 2-я)
        derivative1 = np.diff(spectrum)
        derivative2 = np.diff(derivative1)
        feature_vector.extend([
            np.mean(derivative1),
            np.std(derivative1),
            np.max(np.abs(derivative1)),
            np.mean(derivative2),
            np.std(derivative2),
        ])
        
        # 4. УСИЛЕННЫЕ признаки для клена - область 1
        if len(spectrum) > max(maple_channels_1):
            maple_region1 = spectrum[maple_channels_1]
            avg_spectrum = np.mean(spectrum)
            feature_vector.extend([
                np.mean(maple_region1),
                np.std(maple_region1),
                np.max(maple_region1),
                np.min(maple_region1),
                np.ptp(maple_region1),
                np.mean(maple_region1) / avg_spectrum if avg_spectrum > 0 else 0,
                np.std(maple_region1) / avg_spectrum if avg_spectrum > 0 else 0,
                np.sum(maple_region1 > np.mean(maple_region1)),  # Количество пиков
            ])
        else:
            feature_vector.extend([0] * 8)
        
        # 5. УСИЛЕННЫЕ признаки для клена - область 2
        valid_maple_2 = [ch for ch in maple_channels_2 if ch < len(spectrum)]
        if valid_maple_2:
            maple_region2 = spectrum[valid_maple_2]
            avg_spectrum = np.mean(spectrum)
            feature_vector.extend([
                np.mean(maple_region2),
                np.std(maple_region2),
                np.max(maple_region2),
                np.mean(maple_region2) / avg_spectrum if avg_spectrum > 0 else 0,
                len(maple_region2[maple_region2 > np.percentile(spectrum, 75)]),  # Высокие значения
            ])
        else:
            feature_vector.extend([0] * 5)
        
        # 6. КЛЮЧЕВЫЕ пики для клена
        valid_maple_3 = [ch for ch in maple_channels_3 if ch < len(spectrum)]
        if valid_maple_3:
            maple_peaks = spectrum[valid_maple_3]
            feature_vector.extend([
                np.mean(maple_peaks),
                np.max(maple_peaks),
                np.sum(maple_peaks),
                np.std(maple_peaks),
            ])
        else:
            feature_vector.extend([0] * 4)
        
        # 7. Признаки для дуба
        if len(spectrum) > max(oak_channels):
            oak_region = spectrum[oak_channels]
            feature_vector.extend([
                np.mean(oak_region),
                np.std(oak_region),
                np.max(oak_region) - np.min(oak_region),
                np.mean(oak_region) / np.mean(spectrum) if np.mean(spectrum) > 0 else 0,
            ])
        else:
            feature_vector.extend([0] * 4)
        
        # 8. Энергия в расширенных диапазонах
        n_bands = 6  # Больше диапазонов
        band_size = len(spectrum) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(spectrum))
            if start_idx < len(spectrum):
                band_energy = np.sum(spectrum[start_idx:end_idx] ** 2)
                feature_vector.append(band_energy)
            else:
                feature_vector.append(0)
        
        # 9. Спектральные моменты
        normalized_spectrum = spectrum / np.sum(spectrum) if np.sum(spectrum) > 0 else spectrum
        channels = np.arange(len(spectrum))
        if np.sum(normalized_spectrum) > 0:
            centroid = np.sum(channels * normalized_spectrum)
            spread = np.sqrt(np.sum(((channels - centroid) ** 2) * normalized_spectrum))
            feature_vector.extend([centroid, spread])
        else:
            feature_vector.extend([0, 0])
        
        # 10. Отношения между областями (важно для клена)
        mid_point = len(spectrum) // 2
        first_half = np.mean(spectrum[:mid_point])
        second_half = np.mean(spectrum[mid_point:])
        ratio = first_half / second_half if second_half > 0 else 0
        feature_vector.append(ratio)
        
        features.append(feature_vector)
    
    return np.array(features)

def create_hierarchical_classifier(X_train, y_train, X_test, y_test, class_names):
    """Создает иерархический классификатор"""
    print("\n🔸 СТРАТЕГИЯ 1: Иерархический классификатор")
    print("1-й уровень: Клен vs Не-клен")
    print("2-й уровень: Различение остальных видов")
    
    # Уровень 1: Клен vs остальные
    y_binary_train = (y_train == list(class_names).index('клен')).astype(int)
    y_binary_test = (y_test == list(class_names).index('клен')).astype(int)
    
    # Обучаем классификатор клен/не-клен
    maple_classifier = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, 
        class_weight='balanced'  # балансировка классов
    )
    maple_classifier.fit(X_train, y_binary_train)
    
    # Оценка на тестовых данных
    maple_pred_binary = maple_classifier.predict(X_test)
    maple_pred_proba = maple_classifier.predict_proba(X_test)[:, 1]
    
    maple_accuracy = np.mean(y_binary_test == maple_pred_binary)
    print(f"Точность классификации Клен vs Не-клен: {maple_accuracy:.3f}")
    
    # Подробная статистика для клена
    maple_true_positives = np.sum((y_binary_test == 1) & (maple_pred_binary == 1))
    maple_total_positives = np.sum(y_binary_test == 1)
    maple_recall = maple_true_positives / maple_total_positives if maple_total_positives > 0 else 0
    
    maple_predicted_positives = np.sum(maple_pred_binary == 1)
    maple_precision = maple_true_positives / maple_predicted_positives if maple_predicted_positives > 0 else 0
    
    print(f"Клен - Recall: {maple_recall:.3f}, Precision: {maple_precision:.3f}")
    
    # Уровень 2: Классификация остальных видов
    non_maple_mask_train = y_train != list(class_names).index('клен')
    non_maple_mask_test = y_test != list(class_names).index('клен')
    
    if np.sum(non_maple_mask_train) > 0 and np.sum(non_maple_mask_test) > 0:
        X_train_non_maple = X_train[non_maple_mask_train]
        y_train_non_maple = y_train[non_maple_mask_train]
        X_test_non_maple = X_test[non_maple_mask_test]
        y_test_non_maple = y_test[non_maple_mask_test]
        
        # Обучаем классификатор для остальных видов
        other_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42
        )
        other_classifier.fit(X_train_non_maple, y_train_non_maple)
        
        other_pred = other_classifier.predict(X_test_non_maple)
        other_accuracy = np.mean(y_test_non_maple == other_pred)
        print(f"Точность классификации остальных видов: {other_accuracy:.3f}")
    
    return maple_classifier, maple_recall, maple_precision

def analyze_maple_data_distribution(X_train, y_train, X_test, y_test, class_names):
    """Анализирует распределение данных клена"""
    print("\n🔍 АНАЛИЗ ДАННЫХ КЛЕНА:")
    maple_idx = list(class_names).index('клен')
    
    # Анализ тренировочных данных
    maple_train_mask = y_train == maple_idx
    maple_train_data = X_train[maple_train_mask]
    
    print(f"Тренировочные образцы клена: {np.sum(maple_train_mask)}")
    print(f"Среднее значение признаков клена (трен): {np.mean(maple_train_data, axis=0)[:5]}")
    print(f"Стандартное отклонение (трен): {np.std(maple_train_data, axis=0)[:5]}")
    
    # Анализ тестовых данных
    maple_test_mask = y_test == maple_idx
    maple_test_data = X_test[maple_test_mask]
    
    print(f"Тестовые образцы клена: {np.sum(maple_test_mask)}")
    print(f"Среднее значение признаков клена (тест): {np.mean(maple_test_data, axis=0)[:5]}")
    print(f"Стандартное отклонение (тест): {np.std(maple_test_data, axis=0)[:5]}")
    
    # Сравнение с другими классами
    non_maple_train = X_train[~maple_train_mask]
    non_maple_test = X_test[~maple_test_mask]
    
    print(f"Среднее значение остальных классов (трен): {np.mean(non_maple_train, axis=0)[:5]}")
    print(f"Среднее значение остальных классов (тест): {np.mean(non_maple_test, axis=0)[:5]}")
    
    # Проверка разделимости
    maple_train_mean = np.mean(maple_train_data, axis=0)
    maple_test_mean = np.mean(maple_test_data, axis=0)
    correlation = np.corrcoef(maple_train_mean, maple_test_mean)[0, 1]
    
    print(f"Корреляция между тренировочными и тестовыми данными клена: {correlation:.3f}")
    
    if correlation < 0.5:
        print("⚠️  ПРОБЛЕМА: Низкая корреляция между весенними и летними данными клена!")
    
    return maple_train_data, maple_test_data

def create_enhanced_ensemble_solution(X_train, y_train, X_test, y_test, class_names):
    """Создает улучшенное ансамблевое решение с фокусом на клен"""
    print("\n🔸 СТРАТЕГИЯ 2: Усиленный ансамбль для клена")
    
    # Анализ данных клена
    maple_train_data, maple_test_data = analyze_maple_data_distribution(
        X_train, y_train, X_test, y_test, class_names
    )
    
    models = []
    predictions = []
    maple_idx = list(class_names).index('клен')
    
    # Модель 1: Random Forest с экстремальным фокусом на клен
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == maple_idx] = 20.0  # ОЧЕНЬ высокий вес для клена
    
    rf_extreme = RandomForestClassifier(
        n_estimators=500, 
        max_depth=30, 
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight={maple_idx: 20.0}  # Дополнительное взвешивание
    )
    rf_extreme.fit(X_train, y_train, sample_weight=sample_weights)
    pred1 = rf_extreme.predict(X_test)
    predictions.append(pred1)
    models.append(('RF Extreme Maple', rf_extreme))
    
    # Модель 2: Специальная модель только для клена
    # Создаем бинарную задачу: клен vs все остальные
    y_binary_train = (y_train == maple_idx).astype(int)
    y_binary_test = (y_test == maple_idx).astype(int)
    
    maple_detector = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        random_state=43,
        class_weight={1: 30.0, 0: 1.0}  # Сильный фокус на клен
    )
    maple_detector.fit(X_train, y_binary_train)
    
    # Получаем вероятности для клена
    maple_proba = maple_detector.predict_proba(X_test)[:, 1]
    
    # Модель 3: Балансированная модель для остальных классов
    rf_balanced = RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=44,
        class_weight='balanced'
    )
    rf_balanced.fit(X_train, y_train)
    pred3 = rf_balanced.predict(X_test)
    predictions.append(pred3)
    models.append(('RF Balanced', rf_balanced))
    
    # Модель 4: Нейронная сеть с фокусом на клен
    nn_model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Экстремальные веса для клена
    class_weights = {i: 1.0 for i in range(len(class_names))}
    class_weights[maple_idx] = 15.0
    
    nn_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        class_weight=class_weights,
        verbose=0
    )
    
    pred4 = np.argmax(nn_model.predict(X_test, verbose=0), axis=1)
    predictions.append(pred4)
    models.append(('NN Enhanced', nn_model))
    
    # Гибридное предсказание с приоритетом для клена
    ensemble_pred = []
    for i in range(len(X_test)):
        # Если детектор клена уверен (>0.3), используем его
        if maple_proba[i] > 0.3:
            ensemble_pred.append(maple_idx)
        else:
            # Иначе используем голосование остальных моделей
            votes = [predictions[j][i] for j in range(len(predictions))]
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_pred.append(unique[np.argmax(counts)])
    
    ensemble_pred = np.array(ensemble_pred)
    
    # Оценка моделей
    print("Результаты отдельных моделей:")
    for name, _ in models:
        idx = [i for i, (n, _) in enumerate(models) if n == name][0]
        acc = np.mean(y_test == predictions[idx])
        maple_acc = np.mean(y_test[y_test == maple_idx] == predictions[idx][y_test == maple_idx]) if np.sum(y_test == maple_idx) > 0 else 0
        print(f"  {name}: общая {acc:.3f}, клен {maple_acc:.3f}")
    
    # Оценка детектора клена
    maple_detector_acc = np.mean(y_binary_test == (maple_proba > 0.5))
    maple_recall = np.mean((y_binary_test == 1) & (maple_proba > 0.3)) / np.sum(y_binary_test == 1) if np.sum(y_binary_test == 1) > 0 else 0
    print(f"  Детектор клена: точность {maple_detector_acc:.3f}, recall клена {maple_recall:.3f}")
    
    # Финальная оценка ансамбля
    ensemble_acc = np.mean(y_test == ensemble_pred)
    maple_ensemble_acc = np.mean(y_test[y_test == maple_idx] == ensemble_pred[y_test == maple_idx]) if np.sum(y_test == maple_idx) > 0 else 0
    print(f"Гибридный ансамбль: общая {ensemble_acc:.3f}, клен {maple_ensemble_acc:.3f}")
    
    return ensemble_pred, maple_ensemble_acc

def analyze_final_solution(y_test, y_pred, class_names, title):
    """Анализирует финальное решение"""
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ РЕШЕНИЯ: {title}")
    print("="*60)
    
    accuracy = np.mean(y_test == y_pred)
    print(f"Общая точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Детальный отчет
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\nОтчет по классам:")
    print(report)
    
    # Фокус на проблемные виды
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("\n🎯 РЕЗУЛЬТАТЫ ПО ПРОБЛЕМНЫМ ВИДАМ:")
    for problem_species in ['клен', 'дуб']:
        if problem_species in class_names:
            idx = list(class_names).index(problem_species)
            correct = cm_normalized[idx][idx]
            total = cm[idx].sum()
            
            print(f"{problem_species.upper()}: {correct:.3f} ({correct*100:.1f}%) из {total} образцов")
            
            if problem_species == 'клен':
                if correct > 0.5:
                    print("   🎉 УСПЕХ! Клен теперь хорошо распознается!")
                elif correct > 0.3:
                    print("   ⚡ ПРОГРЕСС! Значительное улучшение!")
                elif correct > 0.1:
                    print("   📈 Улучшение есть, но нужно больше")
                else:
                    print("   ❌ Все еще проблемы...")
    
    return accuracy, cm_normalized

def main():
    """Основная функция"""
    print("🔥 РЕШЕНИЕ ПРОБЛЕМЫ РАСПОЗНАВАНИЯ КЛЕНА")
    print("="*60)
    print("Стратегии:")
    print("1. Возврат к оригинальным данным клена (без клен_ам)")
    print("2. Иерархическая классификация")
    print("3. Ансамблевый подход")
    print("="*60)
    
    # Загрузка данных (БЕЗ новых данных клена)
    print("Загрузка данных (только оригинальные)...")
    train_data, train_labels = load_spring_data()
    test_data, test_labels = load_summer_data_original_only()
    
    print(f"Весенние спектры: {len(train_data)}")
    print(f"Летние спектры: {len(test_data)}")
    print(f"Летние спектры клена: {test_labels.count('клен')}")
    
    # Предобработка
    all_spectra = train_data + test_data
    min_length = min(len(spectrum) for spectrum in all_spectra)
    
    train_data_trimmed = [spectrum[:min_length] for spectrum in train_data]
    test_data_trimmed = [spectrum[:min_length] for spectrum in test_data]
    
    # Извлечение признаков
    print("Извлечение улучшенных признаков с фокусом на клен...")
    X_train = extract_enhanced_maple_features(train_data_trimmed)
    X_test = extract_enhanced_maple_features(test_data_trimmed)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Форма признаков: {X_train_scaled.shape}")
    
    # Базовая модель для сравнения
    print("\n📊 БАЗОВАЯ МОДЕЛЬ (для сравнения):")
    rf_baseline = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_baseline.fit(X_train_scaled, y_train)
    baseline_pred = rf_baseline.predict(X_test_scaled)
    
    baseline_acc, _ = analyze_final_solution(y_test, baseline_pred, label_encoder.classes_, "Базовая модель")
    
    # Стратегия 1: Иерархический классификатор
    maple_classifier, maple_recall, maple_precision = create_hierarchical_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test, label_encoder.classes_
    )
    
    # Стратегия 2: Усиленный ансамблевый подход
    ensemble_pred, maple_ensemble_acc = create_enhanced_ensemble_solution(
        X_train_scaled, y_train, X_test_scaled, y_test, label_encoder.classes_
    )
    
    ensemble_acc, _ = analyze_final_solution(y_test, ensemble_pred, label_encoder.classes_, "Ансамблевая модель")
    
    # Итоговое сравнение
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
    print("="*60)
    print(f"Базовая модель (улучшенные признаки): {baseline_acc:.3f}")
    print(f"Усиленная ансамблевая модель:         {ensemble_acc:.3f}")
    print(f"Иерархический подход (recall клена):  {maple_recall:.3f}")
    print(f"Гибридный ансамбль (точность клена):  {maple_ensemble_acc:.3f}")
    
    print("\n🔧 ИСПРАВЛЕНИЯ В СКРИПТЕ:")
    print("✅ Расширены признаки для клена (46 вместо 24)")
    print("✅ Добавлены специфические каналы: 172-186, 258-287")
    print("✅ Добавлены ключевые пики: 179, 180, 181, 258, 276, 286")
    print("✅ Экстремальные веса для клена (20x вместо 5x)")
    print("✅ Специальный детектор клена с порогом 0.3")
    print("✅ Гибридная стратегия предсказания")
    print("✅ Отладочная информация для анализа данных")
    
    # Определяем лучший результат
    best_approach = "Усиленная ансамблевая модель" if ensemble_acc > baseline_acc else "Базовая модель"
    print(f"\n🏆 Лучший общий результат: {best_approach}")
    
    improvement = ensemble_acc - baseline_acc
    if improvement > 0:
        print(f"📈 Улучшение общей точности: +{improvement:.3f} ({improvement*100:.1f}%)")
    
    if maple_ensemble_acc > 0.5:
        print("🎉 ПРОРЫВ в распознавании клена достигнут!")
    elif maple_ensemble_acc > 0.3:
        print("⚡ Значительный ПРОГРЕСС в распознавании клена!")
    elif maple_ensemble_acc > 0.1:
        print("📈 Есть улучшения в распознавании клена!")
    else:
        print("❌ Клен остается сложным для распознавания")
        print("💡 РЕКОМЕНДАЦИЯ: Необходим сбор дополнительных данных клена")
    
    # Сохранение лучших моделей
    print("\nСохранение улучшенных моделей...")
    import joblib
    joblib.dump(scaler, 'enhanced_solution_scaler.pkl')
    joblib.dump(label_encoder, 'enhanced_solution_label_encoder.pkl')
    joblib.dump(rf_baseline, 'enhanced_baseline_model.pkl')
    joblib.dump(maple_classifier, 'enhanced_maple_detector.pkl')
    
    print("\n📁 СОХРАНЕНЫ ФАЙЛЫ:")
    print("- enhanced_solution_scaler.pkl")
    print("- enhanced_solution_label_encoder.pkl") 
    print("- enhanced_baseline_model.pkl")
    print("- enhanced_maple_detector.pkl")
    
    print("\n🎯 ИСПРАВЛЕННОЕ РЕШЕНИЕ ЗАВЕРШЕНО!")
    print("📊 Используйте улучшенные признаки и гибридный подход для лучших результатов")

if __name__ == "__main__":
    main() 