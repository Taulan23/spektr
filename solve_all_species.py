import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
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
            print(f"Загружено {len(excel_files)} весенних файлов для {tree_type}")
            
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

def load_summer_data():
    """Загружает летние данные"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"Загружено {len(excel_files)} летних файлов для {tree_type}")
            
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

def extract_ultimate_features(spectra):
    """Извлекает максимальное количество признаков для всех видов"""
    features = []
    
    # Расширенные каналы для всех проблемных видов
    maple_channels_1 = list(range(170, 190))   # Клен область 1
    maple_channels_2 = list(range(250, 290))   # Клен область 2
    maple_key_peaks = [179, 180, 181, 258, 276, 286]  # Ключевые пики клена
    oak_channels = list(range(149, 165))       # Дуб расширенные
    
    for spectrum in spectra:
        spectrum = np.array(spectrum)
        feature_vector = []
        
        # 1. Базовые статистические признаки (расширенные)
        feature_vector.extend([
            np.mean(spectrum),
            np.std(spectrum),
            np.median(spectrum),
            np.max(spectrum),
            np.min(spectrum),
            np.ptp(spectrum),
            np.var(spectrum),
            np.sqrt(np.mean(spectrum**2)),  # RMS
            np.mean(np.abs(spectrum)),      # MAD
        ])
        
        # 2. Расширенные квантили
        for p in [5, 10, 15, 25, 35, 50, 65, 75, 85, 90, 95]:
            feature_vector.append(np.percentile(spectrum, p))
        
        # 3. Производные всех порядков
        for order in [1, 2, 3]:
            if order == 1:
                derivative = np.diff(spectrum)
            elif order == 2:
                derivative = np.diff(np.diff(spectrum))
            else:
                derivative = np.diff(np.diff(np.diff(spectrum)))
            
            if len(derivative) > 0:
                feature_vector.extend([
                    np.mean(derivative),
                    np.std(derivative),
                    np.max(np.abs(derivative)),
                    np.min(derivative),
                    np.max(derivative),
                ])
            else:
                feature_vector.extend([0] * 5)
        
        # 4. МАКСИМАЛЬНЫЕ признаки для клена - область 1
        valid_maple_1 = [ch for ch in maple_channels_1 if ch < len(spectrum)]
        if valid_maple_1:
            maple_region1 = spectrum[valid_maple_1]
            avg_spectrum = np.mean(spectrum)
            max_spectrum = np.max(spectrum)
            feature_vector.extend([
                np.mean(maple_region1),
                np.std(maple_region1),
                np.max(maple_region1),
                np.min(maple_region1),
                np.ptp(maple_region1),
                np.median(maple_region1),
                np.var(maple_region1),
                np.mean(maple_region1) / avg_spectrum if avg_spectrum > 0 else 0,
                np.std(maple_region1) / avg_spectrum if avg_spectrum > 0 else 0,
                np.max(maple_region1) / max_spectrum if max_spectrum > 0 else 0,
                np.sum(maple_region1 > np.mean(maple_region1)),
                np.sum(maple_region1 > np.percentile(spectrum, 75)),
                np.sum(maple_region1 < np.percentile(spectrum, 25)),
                np.corrcoef(maple_region1, np.arange(len(maple_region1)))[0,1] if len(maple_region1) > 1 else 0,
                np.sum(np.diff(maple_region1) > 0),  # Возрастающие участки
            ])
        else:
            feature_vector.extend([0] * 15)
        
        # 5. МАКСИМАЛЬНЫЕ признаки для клена - область 2
        valid_maple_2 = [ch for ch in maple_channels_2 if ch < len(spectrum)]
        if valid_maple_2:
            maple_region2 = spectrum[valid_maple_2]
            feature_vector.extend([
                np.mean(maple_region2),
                np.std(maple_region2),
                np.max(maple_region2),
                np.min(maple_region2),
                np.ptp(maple_region2),
                np.mean(maple_region2) / np.mean(spectrum) if np.mean(spectrum) > 0 else 0,
                len(maple_region2[maple_region2 > np.percentile(spectrum, 80)]),
                len(maple_region2[maple_region2 < np.percentile(spectrum, 20)]),
                np.trapz(maple_region2),  # Интеграл области
                np.sum(np.diff(maple_region2) < 0),  # Убывающие участки
            ])
        else:
            feature_vector.extend([0] * 10)
        
        # 6. Ключевые пики клена
        valid_peaks = [ch for ch in maple_key_peaks if ch < len(spectrum)]
        if valid_peaks:
            maple_peaks = spectrum[valid_peaks]
            feature_vector.extend([
                np.mean(maple_peaks),
                np.max(maple_peaks),
                np.min(maple_peaks),
                np.sum(maple_peaks),
                np.std(maple_peaks),
                np.median(maple_peaks),
                np.mean(maple_peaks) / np.mean(spectrum) if np.mean(spectrum) > 0 else 0,
            ])
        else:
            feature_vector.extend([0] * 7)
        
        # 7. Расширенные признаки для дуба
        valid_oak = [ch for ch in oak_channels if ch < len(spectrum)]
        if valid_oak:
            oak_region = spectrum[valid_oak]
            feature_vector.extend([
                np.mean(oak_region),
                np.std(oak_region),
                np.max(oak_region),
                np.min(oak_region),
                np.ptp(oak_region),
                np.mean(oak_region) / np.mean(spectrum) if np.mean(spectrum) > 0 else 0,
                np.sum(oak_region > np.mean(oak_region)),
                np.trapz(oak_region),
                np.var(oak_region),
                np.corrcoef(oak_region, np.arange(len(oak_region)))[0,1] if len(oak_region) > 1 else 0,
            ])
        else:
            feature_vector.extend([0] * 10)
        
        # 8. Энергия в множественных диапазонах
        n_bands = 10
        band_size = len(spectrum) // n_bands
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(spectrum))
            if start_idx < len(spectrum):
                band = spectrum[start_idx:end_idx]
                band_energy = np.sum(band ** 2)
                band_mean = np.mean(band)
                feature_vector.extend([band_energy, band_mean])
            else:
                feature_vector.extend([0, 0])
        
        # 9. Спектральные моменты (расширенные)
        normalized_spectrum = spectrum / np.sum(spectrum) if np.sum(spectrum) > 0 else spectrum
        channels = np.arange(len(spectrum))
        if np.sum(normalized_spectrum) > 0:
            centroid = np.sum(channels * normalized_spectrum)
            spread = np.sqrt(np.sum(((channels - centroid) ** 2) * normalized_spectrum))
            skewness = np.sum(((channels - centroid) ** 3) * normalized_spectrum) / (spread ** 3) if spread > 0 else 0
            kurtosis = np.sum(((channels - centroid) ** 4) * normalized_spectrum) / (spread ** 4) if spread > 0 else 0
            feature_vector.extend([centroid, spread, skewness, kurtosis])
        else:
            feature_vector.extend([0, 0, 0, 0])
        
        # 10. Отношения между частями спектра
        quarter = len(spectrum) // 4
        q1 = np.mean(spectrum[:quarter])
        q2 = np.mean(spectrum[quarter:2*quarter])
        q3 = np.mean(spectrum[2*quarter:3*quarter])
        q4 = np.mean(spectrum[3*quarter:])
        
        feature_vector.extend([
            q1/q2 if q2 > 0 else 0,
            q2/q3 if q3 > 0 else 0,
            q3/q4 if q4 > 0 else 0,
            q1/q4 if q4 > 0 else 0,
            (q1+q4)/(q2+q3) if (q2+q3) > 0 else 0,
        ])
        
        # 11. Локальные экстремумы
        from scipy.signal import find_peaks
        try:
            peaks, _ = find_peaks(spectrum, height=np.percentile(spectrum, 60))
            valleys, _ = find_peaks(-spectrum, height=-np.percentile(spectrum, 40))
            feature_vector.extend([
                len(peaks),
                len(valleys),
                np.mean(spectrum[peaks]) if len(peaks) > 0 else 0,
                np.mean(spectrum[valleys]) if len(valleys) > 0 else 0,
            ])
        except:
            feature_vector.extend([0, 0, 0, 0])
        
        features.append(feature_vector)
    
    return np.array(features)

def create_seasonal_adaptation_model(input_shape, num_classes):
    """Создает модель с сезонной адаптацией для клена"""
    
    # Входной слой
    inputs = layers.Input(shape=(input_shape,))
    
    # Базовая экстракция признаков
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Специализированные ветви для каждого проблемного вида
    
    # Ветвь для клена (сезонная адаптация)
    maple_branch = layers.Dense(128, activation='relu', name='maple_specialization')(x)
    maple_branch = layers.BatchNormalization()(maple_branch)
    maple_branch = layers.Dropout(0.3)(maple_branch)
    maple_branch = layers.Dense(64, activation='relu')(maple_branch)
    maple_branch = layers.Dropout(0.2)(maple_branch)
    
    # Ветвь для дуба
    oak_branch = layers.Dense(64, activation='relu', name='oak_specialization')(x)
    oak_branch = layers.BatchNormalization()(oak_branch)
    oak_branch = layers.Dropout(0.2)(oak_branch)
    
    # Общая ветвь для остальных видов
    general_branch = layers.Dense(256, activation='relu', name='general_branch')(x)
    general_branch = layers.BatchNormalization()(general_branch)
    general_branch = layers.Dropout(0.3)(general_branch)
    general_branch = layers.Dense(128, activation='relu')(general_branch)
    general_branch = layers.Dropout(0.2)(general_branch)
    
    # Комбинирование ветвей
    combined = layers.Concatenate()([maple_branch, oak_branch, general_branch])
    
    # Механизм внимания для фокуса на важных признаках
    attention = layers.Dense(combined.shape[-1], activation='sigmoid', name='attention')(combined)
    attended = layers.Multiply()([combined, attention])
    
    # Финальные слои
    output = layers.Dense(256, activation='relu')(attended)
    output = layers.BatchNormalization()(output)
    output = layers.Dropout(0.3)(output)
    
    output = layers.Dense(128, activation='relu')(output)
    output = layers.Dropout(0.2)(output)
    
    # Выходной слой
    predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(output)
    
    model = keras.Model(inputs=inputs, outputs=predictions)
    
    # Специальный оптимизатор с адаптивным learning rate
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_mega_ensemble(X_train, y_train, X_test, y_test, class_names):
    """Создает мега-ансамбль из множества моделей"""
    print("\n🚀 СОЗДАНИЕ МЕГА-АНСАМБЛЯ ДЛЯ ВСЕХ ВИДОВ")
    
    models = []
    predictions = []
    maple_idx = list(class_names).index('клен')
    oak_idx = list(class_names).index('дуб')
    
    # 1. Экстремальный Random Forest для клена
    print("Обучение Extreme Random Forest...")
    rf_extreme = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight={maple_idx: 50.0, oak_idx: 10.0}  # Экстремальные веса
    )
    
    # Экстремальные веса для образцов
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == maple_idx] = 100.0
    sample_weights[y_train == oak_idx] = 20.0
    
    rf_extreme.fit(X_train, y_train, sample_weight=sample_weights)
    pred1 = rf_extreme.predict(X_test)
    predictions.append(pred1)
    models.append(('Extreme RF', rf_extreme))
    
    # 2. Gradient Boosting с фокусом на проблемные виды
    print("Обучение Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )
    gb_model.fit(X_train, y_train, sample_weight=sample_weights)
    pred2 = gb_model.predict(X_test)
    predictions.append(pred2)
    models.append(('Gradient Boosting', gb_model))
    
    # 3. SVM с RBF ядром
    print("Обучение SVM...")
    svm_model = SVC(
        C=100.0,
        gamma='scale',
        kernel='rbf',
        class_weight={maple_idx: 100.0, oak_idx: 20.0},
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train, sample_weight=sample_weights)
    pred3 = svm_model.predict(X_test)
    predictions.append(pred3)
    models.append(('SVM RBF', svm_model))
    
    # 4. Специальная нейронная сеть с сезонной адаптацией
    print("Обучение Seasonal Adaptation Neural Network...")
    nn_model = create_seasonal_adaptation_model(X_train.shape[1], len(class_names))
    
    # Экстремальные веса классов
    class_weights = {i: 1.0 for i in range(len(class_names))}
    class_weights[maple_idx] = 100.0
    class_weights[oak_idx] = 20.0
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=10, min_lr=1e-6
    )
    
    history = nn_model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,  # Маленький batch для лучшего обучения
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    pred4 = np.argmax(nn_model.predict(X_test, verbose=0), axis=1)
    predictions.append(pred4)
    models.append(('Seasonal NN', nn_model))
    
    # 5. Специальный детектор для каждого проблемного вида
    print("Обучение специальных детекторов...")
    
    # Детектор клена
    maple_detector = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=43,
        class_weight={1: 200.0, 0: 1.0},
        n_jobs=-1
    )
    y_maple_binary = (y_train == maple_idx).astype(int)
    maple_weights = np.ones(len(y_train))
    maple_weights[y_train == maple_idx] = 200.0
    maple_detector.fit(X_train, y_maple_binary, sample_weight=maple_weights)
    maple_proba = maple_detector.predict_proba(X_test)[:, 1]
    
    # Детектор дуба
    oak_detector = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=44,
        class_weight={1: 50.0, 0: 1.0},
        n_jobs=-1
    )
    y_oak_binary = (y_train == oak_idx).astype(int)
    oak_weights = np.ones(len(y_train))
    oak_weights[y_train == oak_idx] = 50.0
    oak_detector.fit(X_train, y_oak_binary, sample_weight=oak_weights)
    oak_proba = oak_detector.predict_proba(X_test)[:, 1]
    
    # АГРЕССИВНАЯ стратегия комбинирования
    print("Создание агрессивной стратегии предсказания...")
    
    predictions_array = np.array(predictions)
    final_predictions = []
    
    for i in range(len(X_test)):
        # Если детектор клена очень уверен (порог 0.2), форсируем клен
        if maple_proba[i] > 0.2:
            final_predictions.append(maple_idx)
        # Если детектор дуба уверен (порог 0.3), форсируем дуб
        elif oak_proba[i] > 0.3:
            final_predictions.append(oak_idx)
        else:
            # Взвешенное голосование с приоритетом для специализированных моделей
            votes = predictions_array[:, i]
            
            # Подсчет голосов с весами
            vote_weights = {
                0: 3.0,  # Extreme RF
                1: 2.0,  # Gradient Boosting  
                2: 2.0,  # SVM
                3: 4.0,  # Seasonal NN (максимальный вес)
            }
            
            weighted_votes = {}
            for vote_idx, vote in enumerate(votes):
                weight = vote_weights.get(vote_idx, 1.0)
                if vote in weighted_votes:
                    weighted_votes[vote] += weight
                else:
                    weighted_votes[vote] = weight
            
            # Выбираем класс с максимальным весом
            best_class = max(weighted_votes, key=weighted_votes.get)
            final_predictions.append(best_class)
    
    final_predictions = np.array(final_predictions)
    
    # Анализ результатов
    print("\n📊 РЕЗУЛЬТАТЫ ОТДЕЛЬНЫХ МОДЕЛЕЙ:")
    for name, _ in models:
        idx = [i for i, (n, _) in enumerate(models) if n == name][0]
        acc = np.mean(y_test == predictions_array[idx])
        maple_acc = np.mean(y_test[y_test == maple_idx] == predictions_array[idx][y_test == maple_idx]) if np.sum(y_test == maple_idx) > 0 else 0
        oak_acc = np.mean(y_test[y_test == oak_idx] == predictions_array[idx][y_test == oak_idx]) if np.sum(y_test == oak_idx) > 0 else 0
        print(f"  {name}: общая {acc:.3f}, клен {maple_acc:.3f}, дуб {oak_acc:.3f}")
    
    # Детекторы
    maple_recall = np.sum((y_test == maple_idx) & (maple_proba > 0.2)) / np.sum(y_test == maple_idx) if np.sum(y_test == maple_idx) > 0 else 0
    oak_recall = np.sum((y_test == oak_idx) & (oak_proba > 0.3)) / np.sum(y_test == oak_idx) if np.sum(y_test == oak_idx) > 0 else 0
    print(f"  Детектор клена: recall {maple_recall:.3f} (порог 0.2)")
    print(f"  Детектор дуба: recall {oak_recall:.3f} (порог 0.3)")
    
    # Финальные результаты
    final_acc = np.mean(y_test == final_predictions)
    final_maple_acc = np.mean(y_test[y_test == maple_idx] == final_predictions[y_test == maple_idx]) if np.sum(y_test == maple_idx) > 0 else 0
    final_oak_acc = np.mean(y_test[y_test == oak_idx] == final_predictions[y_test == oak_idx]) if np.sum(y_test == oak_idx) > 0 else 0
    
    print(f"\n🎯 АГРЕССИВНЫЙ АНСАМБЛЬ:")
    print(f"  Общая точность: {final_acc:.3f}")
    print(f"  Клен: {final_maple_acc:.3f}")
    print(f"  Дуб: {final_oak_acc:.3f}")
    
    return final_predictions, final_maple_acc, final_oak_acc, models

def analyze_ultimate_solution(y_test, y_pred, class_names, title):
    """Окончательный анализ решения"""
    print(f"\n{'='*70}")
    print(f"🏆 ФИНАЛЬНЫЙ АНАЛИЗ: {title}")
    print("="*70)
    
    accuracy = np.mean(y_test == y_pred)
    print(f"🎯 ОБЩАЯ ТОЧНОСТЬ: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Детальный отчет
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    print(report)
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("\n🎯 РЕЗУЛЬТАТЫ ПО ВСЕМ ВИДАМ:")
    success_count = 0
    for i, species in enumerate(class_names):
        correct = cm_normalized[i][i]
        total = cm[i].sum()
        
        if correct > 0.5:
            status = "🎉 ОТЛИЧНО"
            success_count += 1
        elif correct > 0.3:
            status = "⚡ ХОРОШО"
            success_count += 1
        elif correct > 0.1:
            status = "📈 ПРИЕМЛЕМО"
        else:
            status = "❌ ПРОБЛЕМА"
        
        print(f"  {species.upper()}: {correct:.3f} ({correct*100:.1f}%) из {total} образцов - {status}")
    
    print(f"\n✅ УСПЕШНО РАСПОЗНАЕТСЯ: {success_count}/7 видов")
    
    if success_count == 7:
        print("🏆 ПОЛНЫЙ УСПЕХ! ВСЕ ВИДЫ РАСПОЗНАЮТСЯ!")
    elif success_count >= 6:
        print("⚡ ПОЧТИ ИДЕАЛЬНО! Осталось довести 1-2 вида")
    elif success_count >= 5:
        print("📈 ХОРОШИЙ РЕЗУЛЬТАТ! Большинство видов работает")
    else:
        print("❌ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ РАБОТА")
    
    return accuracy, cm_normalized, success_count

def main():
    """Главная функция - решение для ВСЕХ видов"""
    print("🔥🔥🔥 АГРЕССИВНОЕ РЕШЕНИЕ ДЛЯ ВСЕХ 7 ВИДОВ ДЕРЕВЬЕВ 🔥🔥🔥")
    print("="*80)
    print("🎯 ЦЕЛЬ: 100% распознавание всех видов включая клен и дуб")
    print("🚀 МЕТОДЫ: Мега-ансамбль + Сезонная адаптация + Экстремальные веса")
    print("="*80)
    
    # Загрузка данных
    print("\n📥 ЗАГРУЗКА ДАННЫХ...")
    train_data, train_labels = load_spring_data()
    test_data, test_labels = load_summer_data()
    
    print(f"✅ Весенние спектры: {len(train_data)}")
    print(f"✅ Летние спектры: {len(test_data)}")
    
    for species in ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']:
        spring_count = train_labels.count(species)
        summer_count = test_labels.count(species)
        print(f"  {species}: весна {spring_count}, лето {summer_count}")
    
    # Предобработка
    print("\n🔧 ПРЕДОБРАБОТКА...")
    all_spectra = train_data + test_data
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    train_data_trimmed = [spectrum[:min_length] for spectrum in train_data]
    test_data_trimmed = [spectrum[:min_length] for spectrum in test_data]
    
    # Максимальное извлечение признаков
    print("\n🧠 ИЗВЛЕЧЕНИЕ МАКСИМАЛЬНОГО КОЛИЧЕСТВА ПРИЗНАКОВ...")
    X_train = extract_ultimate_features(train_data_trimmed)
    X_test = extract_ultimate_features(test_data_trimmed)
    
    print(f"✅ Извлечено {X_train.shape[1]} признаков!")
    
    # Кодирование и нормализация
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    # Робастная нормализация (лучше для выбросов)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Данные нормализованы")
    
    # Создание мега-ансамбля
    final_pred, maple_acc, oak_acc, models = create_mega_ensemble(
        X_train_scaled, y_train, X_test_scaled, y_test, label_encoder.classes_
    )
    
    # Финальный анализ
    accuracy, cm_norm, success_count = analyze_ultimate_solution(
        y_test, final_pred, label_encoder.classes_, "МЕГА-АНСАМБЛЬ ДЛЯ ВСЕХ ВИДОВ"
    )
    
    # Сохранение всех моделей
    print("\n💾 СОХРАНЕНИЕ ВСЕХ МОДЕЛЕЙ...")
    import joblib
    
    joblib.dump(scaler, 'ultimate_scaler.pkl')
    joblib.dump(label_encoder, 'ultimate_label_encoder.pkl')
    
    # Сохранение каждой модели
    for i, (name, model) in enumerate(models):
        if 'NN' in name:
            model.save(f'ultimate_model_{i}_{name.replace(" ", "_")}.keras')
        else:
            joblib.dump(model, f'ultimate_model_{i}_{name.replace(" ", "_")}.pkl')
    
    print("\n🎉 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("="*50)
    print(f"🎯 Общая точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"🍁 Клен: {maple_acc:.3f} ({maple_acc*100:.1f}%)")
    print(f"🌳 Дуб: {oak_acc:.3f} ({oak_acc*100:.1f}%)")
    print(f"✅ Успешных видов: {success_count}/7")
    
    if success_count == 7:
        print("\n🏆🏆🏆 МИССИЯ ВЫПОЛНЕНА! ВСЕ ВИДЫ РАСПОЗНАЮТСЯ! 🏆🏆🏆")
    elif success_count >= 6:
        print("\n⚡⚡⚡ ПОЧТИ ИДЕАЛЬНО! ОТЛИЧНЫЙ РЕЗУЛЬТАТ! ⚡⚡⚡")
    else:
        print(f"\n📈 ХОРОШИЙ ПРОГРЕСС! {success_count} из 7 видов работают")
    
    print("\n🎯 РЕШЕНИЕ ДЛЯ ВСЕХ ВИДОВ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main() 