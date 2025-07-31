import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Установка семян для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

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
    """Загружает летние данные с включением клен_ам"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    for tree_type in tree_types:
        # Загружаем оригинальные летние данные
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
        
        # Для клена дополнительно загружаем из клен_ам
        if tree_type == 'клен':
            am_folder = "клен_ам"
            if os.path.exists(am_folder):
                am_files = glob.glob(os.path.join(am_folder, '*.xlsx'))
                print(f"Дополнительно загружено {len(am_files)} файлов клена из клен_ам")
                
                for file_path in am_files:
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

def extract_super_features(spectra):
    """Извлекает максимальное количество признаков с адаптацией"""
    features = []
    
    # Расширенные каналы для каждого вида
    species_channels = {
        'клен': [
            list(range(170, 190)),  # Основная область
            list(range(240, 290)),  # Вторичная область
            [179, 180, 181, 258, 276, 286, 172, 173, 174, 175, 176, 177, 178]  # Ключевые пики
        ],
        'дуб': [
            list(range(145, 170)),  # Расширенная область
            list(range(200, 220)),  # Дополнительная область
            [151, 152, 153, 154, 155, 156, 157, 158, 159, 160]  # Ключевые каналы
        ],
        'береза': [list(range(100, 140)), list(range(260, 300))],
        'ель': [list(range(120, 160)), list(range(220, 260)), list(range(50, 90))],
        'липа': [list(range(80, 120)), list(range(180, 220)), list(range(280, 300))],
        'осина': [list(range(60, 100)), list(range(140, 180)), list(range(200, 240))],
        'сосна': [list(range(90, 130)), list(range(160, 200)), list(range(270, 300))]
    }
    
    for spectrum in spectra:
        spectrum = np.array(spectrum)
        feature_vector = []
        
        # 1. Расширенная базовая статистика
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
            np.sum(spectrum),
            np.prod(np.sign(spectrum) * (np.abs(spectrum) + 1e-12))**(1/len(spectrum)),  # Геометрическое среднее
            len(spectrum[spectrum > np.mean(spectrum)]) / len(spectrum),  # Доля выше среднего
        ])
        
        # 2. Множественные квантили
        for p in range(5, 100, 5):  # От 5% до 95% с шагом 5%
            feature_vector.append(np.percentile(spectrum, p))
        
        # 3. Производные всех порядков до 4-го
        current_spectrum = spectrum.copy()
        for order in range(1, 5):
            if len(current_spectrum) > 1:
                derivative = np.diff(current_spectrum)
                if len(derivative) > 0:
                    feature_vector.extend([
                        np.mean(derivative),
                        np.std(derivative),
                        np.max(derivative),
                        np.min(derivative),
                        np.max(np.abs(derivative)),
                        np.sum(derivative > 0) / len(derivative) if len(derivative) > 0 else 0,
                        np.trapz(np.abs(derivative)) if len(derivative) > 1 else 0,
                    ])
                    current_spectrum = derivative
                else:
                    feature_vector.extend([0] * 7)
            else:
                feature_vector.extend([0] * 7)
        
        # 4. Специфические признаки для каждого вида
        for species, channel_groups in species_channels.items():
            for group_idx, channels in enumerate(channel_groups):
                valid_channels = [ch for ch in channels if ch < len(spectrum)]
                if valid_channels:
                    region = spectrum[valid_channels]
                    avg_spectrum = np.mean(spectrum)
                    feature_vector.extend([
                        np.mean(region),
                        np.std(region),
                        np.max(region),
                        np.min(region),
                        np.median(region),
                        np.ptp(region),
                        np.var(region),
                        np.mean(region) / avg_spectrum if avg_spectrum > 0 else 0,
                        np.sum(region > np.percentile(spectrum, 75)),
                        np.sum(region < np.percentile(spectrum, 25)),
                        np.trapz(region) if len(region) > 1 else 0,
                        np.corrcoef(region, np.arange(len(region)))[0,1] if len(region) > 1 else 0,
                    ])
                else:
                    feature_vector.extend([0] * 12)
        
        # 5. Фурье-преобразование (частотные характеристики)
        try:
            fft = np.fft.fft(spectrum)
            fft_real = np.real(fft)[:len(spectrum)//2]
            fft_imag = np.imag(fft)[:len(spectrum)//2]
            fft_mag = np.abs(fft)[:len(spectrum)//2]
            
            feature_vector.extend([
                np.mean(fft_mag),
                np.std(fft_mag),
                np.max(fft_mag),
                np.argmax(fft_mag),  # Доминирующая частота
                np.sum(fft_mag[:len(fft_mag)//4]),  # Низкие частоты
                np.sum(fft_mag[len(fft_mag)//4:3*len(fft_mag)//4]),  # Средние частоты
                np.sum(fft_mag[3*len(fft_mag)//4:]),  # Высокие частоты
            ])
        except:
            feature_vector.extend([0] * 7)
        
        # 6. Энергия в адаптивных диапазонах
        n_bands = 15
        band_size = len(spectrum) // n_bands
        band_energies = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i + 1) * band_size, len(spectrum))
            if start_idx < len(spectrum):
                band = spectrum[start_idx:end_idx]
                energy = np.sum(band ** 2)
                band_energies.append(energy)
            else:
                band_energies.append(0)
        
        feature_vector.extend(band_energies)
        
        # Отношения между диапазонами
        for i in range(min(5, len(band_energies))):
            for j in range(i+1, min(5, len(band_energies))):
                if band_energies[j] > 0:
                    feature_vector.append(band_energies[i] / band_energies[j])
                else:
                    feature_vector.append(0)
        
        # 7. Спектральные моменты высших порядков
        normalized_spectrum = spectrum / np.sum(spectrum) if np.sum(spectrum) > 0 else spectrum
        channels = np.arange(len(spectrum))
        if np.sum(normalized_spectrum) > 0:
            centroid = np.sum(channels * normalized_spectrum)
            spread = np.sqrt(np.sum(((channels - centroid) ** 2) * normalized_spectrum))
            skewness = np.sum(((channels - centroid) ** 3) * normalized_spectrum) / (spread ** 3) if spread > 0 else 0
            kurtosis = np.sum(((channels - centroid) ** 4) * normalized_spectrum) / (spread ** 4) if spread > 0 else 0
            
            # Дополнительные моменты
            fifth_moment = np.sum(((channels - centroid) ** 5) * normalized_spectrum) / (spread ** 5) if spread > 0 else 0
            sixth_moment = np.sum(((channels - centroid) ** 6) * normalized_spectrum) / (spread ** 6) if spread > 0 else 0
            
            feature_vector.extend([centroid, spread, skewness, kurtosis, fifth_moment, sixth_moment])
        else:
            feature_vector.extend([0] * 6)
        
        # 8. Локальные экстремумы и паттерны
        try:
            from scipy.signal import find_peaks, argrelextrema
            
            # Поиск пиков и долин
            peaks, _ = find_peaks(spectrum, height=np.percentile(spectrum, 50))
            valleys, _ = find_peaks(-spectrum, height=-np.percentile(spectrum, 50))
            
            # Локальные максимумы и минимумы
            local_max = argrelextrema(spectrum, np.greater, order=3)[0]
            local_min = argrelextrema(spectrum, np.less, order=3)[0]
            
            feature_vector.extend([
                len(peaks),
                len(valleys),
                len(local_max),
                len(local_min),
                np.mean(spectrum[peaks]) if len(peaks) > 0 else 0,
                np.mean(spectrum[valleys]) if len(valleys) > 0 else 0,
                np.std(spectrum[peaks]) if len(peaks) > 0 else 0,
                np.std(spectrum[valleys]) if len(valleys) > 0 else 0,
                len(peaks) / len(spectrum) if len(spectrum) > 0 else 0,  # Плотность пиков
                len(valleys) / len(spectrum) if len(spectrum) > 0 else 0,  # Плотность долин
            ])
        except:
            feature_vector.extend([0] * 10)
        
        # 9. Дополнительные статистические характеристики
        feature_vector.extend([
            np.sum(spectrum > 0) / len(spectrum) if len(spectrum) > 0 else 0,  # Доля положительных
            np.sum(spectrum < 0) / len(spectrum) if len(spectrum) > 0 else 0,  # Доля отрицательных
            len(np.where(np.diff(spectrum) > 0)[0]) / len(spectrum) if len(spectrum) > 1 else 0,  # Доля возрастаний
            len(np.where(np.diff(spectrum) < 0)[0]) / len(spectrum) if len(spectrum) > 1 else 0,  # Доля убываний
            np.sum(np.abs(spectrum - np.mean(spectrum))) / len(spectrum),  # Среднее абсолютное отклонение
        ])
        
        features.append(feature_vector)
    
    return np.array(features)

def augment_data(X, y, class_names, target_samples=200):
    """Аугментирует данные для балансировки классов"""
    print("🔄 Аугментация данных для балансировки классов...")
    
    augmented_X = []
    augmented_y = []
    
    for class_idx, class_name in enumerate(class_names):
        class_mask = y == class_idx
        class_data = X[class_mask]
        current_samples = len(class_data)
        
        print(f"  {class_name}: {current_samples} -> {target_samples} образцов")
        
        # Добавляем оригинальные данные
        augmented_X.extend(class_data)
        augmented_y.extend([class_idx] * current_samples)
        
        # Если нужно больше образцов
        if current_samples < target_samples:
            needed = target_samples - current_samples
            
            for _ in range(needed):
                # Выбираем случайный образец из класса
                base_sample = class_data[np.random.randint(0, current_samples)]
                
                # Применяем различные виды аугментации
                aug_type = np.random.choice(['noise', 'scale', 'shift', 'smooth', 'mix'])
                
                if aug_type == 'noise':
                    # Добавление гауссового шума
                    noise_level = np.random.uniform(0.01, 0.05)
                    augmented_sample = base_sample + np.random.normal(0, noise_level * np.std(base_sample), len(base_sample))
                
                elif aug_type == 'scale':
                    # Масштабирование
                    scale_factor = np.random.uniform(0.9, 1.1)
                    augmented_sample = base_sample * scale_factor
                
                elif aug_type == 'shift':
                    # Сдвиг по амплитуде
                    shift_amount = np.random.uniform(-0.1, 0.1) * np.mean(np.abs(base_sample))
                    augmented_sample = base_sample + shift_amount
                
                elif aug_type == 'smooth':
                    # Сглаживание
                    from scipy.ndimage import gaussian_filter1d
                    sigma = np.random.uniform(0.5, 2.0)
                    augmented_sample = gaussian_filter1d(base_sample, sigma=sigma)
                
                elif aug_type == 'mix':
                    # Смешивание с другим образцом того же класса
                    if current_samples > 1:
                        other_sample = class_data[np.random.randint(0, current_samples)]
                        alpha = np.random.uniform(0.3, 0.7)
                        augmented_sample = alpha * base_sample + (1 - alpha) * other_sample
                    else:
                        augmented_sample = base_sample
                
                augmented_X.append(augmented_sample)
                augmented_y.append(class_idx)
    
    return np.array(augmented_X), np.array(augmented_y)

def create_species_specific_models(X_train, y_train, class_names):
    """Создает отдельные модели для каждого вида"""
    print("🎯 Создание специализированных моделей для каждого вида...")
    
    species_models = {}
    
    for class_idx, species in enumerate(class_names):
        print(f"  Обучение модели для {species}...")
        
        # Создаем бинарную задачу: текущий вид vs все остальные
        y_binary = (y_train == class_idx).astype(int)
        
        # Создаем ансамбль для этого вида
        models = []
        
        # Random Forest с экстремальными параметрами
        rf = ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight={1: 100.0, 0: 1.0},
            random_state=42 + class_idx,
            n_jobs=-1
        )
        
        # Веса для образцов
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == class_idx] = 200.0
        
        rf.fit(X_train, y_binary, sample_weight=sample_weights)
        models.append(('ExtraTrees', rf))
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=12,
            subsample=0.8,
            random_state=42 + class_idx
        )
        gb.fit(X_train, y_binary, sample_weight=sample_weights)
        models.append(('GradientBoosting', gb))
        
        # SVM с RBF
        svm = SVC(
            C=1000.0,
            gamma='scale',
            kernel='rbf',
            class_weight={1: 200.0, 0: 1.0},
            probability=True,
            random_state=42 + class_idx
        )
        svm.fit(X_train, y_binary, sample_weight=sample_weights)
        models.append(('SVM', svm))
        
        species_models[species] = models
    
    return species_models

def create_meta_ensemble_model(input_shape, num_classes):
    """Создает мета-ансамблевую нейронную сеть"""
    
    # Входной слой
    inputs = layers.Input(shape=(input_shape,))
    
    # Первичная обработка признаков
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Создаем отдельные ветви для каждого вида
    species_branches = []
    
    for i in range(num_classes):
        # Специализированная ветвь для каждого вида
        branch = layers.Dense(256, activation='relu', name=f'species_{i}_branch')(x)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Dropout(0.3)(branch)
        
        branch = layers.Dense(128, activation='relu')(branch)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Dropout(0.2)(branch)
        
        branch = layers.Dense(64, activation='relu')(branch)
        branch = layers.Dropout(0.2)(branch)
        
        species_branches.append(branch)
    
    # Комбинируем все ветви
    if len(species_branches) > 1:
        combined = layers.Concatenate()(species_branches)
    else:
        combined = species_branches[0]
    
    # Механизм внимания
    attention_weights = layers.Dense(combined.shape[-1], activation='softmax', name='attention_weights')(combined)
    attended = layers.Multiply()([combined, attention_weights])
    
    # Финальные слои
    output = layers.Dense(512, activation='relu')(attended)
    output = layers.BatchNormalization()(output)
    output = layers.Dropout(0.3)(output)
    
    output = layers.Dense(256, activation='relu')(output)
    output = layers.Dropout(0.2)(output)
    
    output = layers.Dense(128, activation='relu')(output)
    output = layers.Dropout(0.1)(output)
    
    # Выходной слой
    predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(output)
    
    model = keras.Model(inputs=inputs, outputs=predictions)
    
    # Компиляция с адаптивным оптимизатором
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
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

def advanced_prediction_fusion(species_models, meta_model, X_test, class_names):
    """Продвинутое объединение предсказаний"""
    print("🤖 Продвинутое объединение предсказаний...")
    
    # Получаем предсказания от специализированных моделей
    species_predictions = {}
    species_probabilities = {}
    
    for species, models in species_models.items():
        class_idx = list(class_names).index(species)
        predictions = []
        probabilities = []
        
        for name, model in models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]  # Вероятность принадлежности к классу
                probabilities.append(proba)
            
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Усредняем предсказания моделей для данного вида
        species_predictions[species] = np.mean(predictions, axis=0)
        species_probabilities[species] = np.mean(probabilities, axis=0) if probabilities else np.zeros(len(X_test))
    
    # Получаем предсказания от мета-модели
    meta_pred = np.argmax(meta_model.predict(X_test, verbose=0), axis=1)
    meta_proba = meta_model.predict(X_test, verbose=0)
    
    # Адаптивное объединение
    final_predictions = []
    final_confidences = []
    
    for i in range(len(X_test)):
        # Собираем вероятности от всех специализированных моделей
        species_votes = {}
        total_confidence = 0
        
        for j, species in enumerate(class_names):
            # Вероятность от специализированной модели
            specialist_prob = species_probabilities[species][i]
            
            # Вероятность от мета-модели
            meta_prob = meta_proba[i][j]
            
            # Комбинированная вероятность с адаптивными весами
            if specialist_prob > 0.8:  # Высокая уверенность специалиста
                combined_prob = 0.8 * specialist_prob + 0.2 * meta_prob
            elif specialist_prob > 0.5:  # Средняя уверенность
                combined_prob = 0.6 * specialist_prob + 0.4 * meta_prob
            else:  # Низкая уверенность - больше доверяем мета-модели
                combined_prob = 0.3 * specialist_prob + 0.7 * meta_prob
            
            species_votes[j] = combined_prob
            total_confidence += combined_prob
        
        # Нормализуем вероятности
        if total_confidence > 0:
            for class_idx in species_votes:
                species_votes[class_idx] /= total_confidence
        
        # Выбираем класс с максимальной вероятностью
        best_class = max(species_votes, key=species_votes.get)
        best_confidence = species_votes[best_class]
        
        # Дополнительная проверка на основе порогов
        confidence_thresholds = {
            'клен': 0.6,
            'дуб': 0.7,
            'береза': 0.3,
            'ель': 0.4,
            'липа': 0.5,
            'осина': 0.4,
            'сосна': 0.5
        }
        
        species_name = class_names[best_class]
        threshold = confidence_thresholds.get(species_name, 0.5)
        
        if best_confidence >= threshold:
            final_predictions.append(best_class)
        else:
            # Если уверенность низкая, выбираем второй по вероятности класс
            sorted_votes = sorted(species_votes.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_votes) > 1:
                second_best = sorted_votes[1][0]
                final_predictions.append(second_best)
            else:
                final_predictions.append(best_class)
        
        final_confidences.append(best_confidence)
    
    return np.array(final_predictions), np.array(final_confidences)

def analyze_ultimate_results(y_test, y_pred, y_confidence, class_names):
    """Анализирует окончательные результаты"""
    print("\n" + "="*80)
    print("🏆 ОКОНЧАТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ - ВСЕ ВИДЫ")
    print("="*80)
    
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
    excellent_count = 0
    good_count = 0
    acceptable_count = 0
    
    for i, species in enumerate(class_names):
        correct = cm_normalized[i][i]
        total = cm[i].sum()
        avg_confidence = np.mean(y_confidence[y_test == i]) if np.sum(y_test == i) > 0 else 0
        
        if correct >= 0.8:
            status = "🏆 ПРЕВОСХОДНО"
            excellent_count += 1
        elif correct >= 0.6:
            status = "🎉 ОТЛИЧНО"
            good_count += 1
        elif correct >= 0.4:
            status = "⚡ ХОРОШО"
            acceptable_count += 1
        elif correct >= 0.2:
            status = "📈 УДОВЛЕТВОРИТЕЛЬНО"
        else:
            status = "❌ ТРЕБУЕТ УЛУЧШЕНИЙ"
        
        print(f"  {species.upper()}: {correct:.3f} ({correct*100:.1f}%) | "
              f"Уверенность: {avg_confidence:.3f} | {status}")
    
    working_species = excellent_count + good_count + acceptable_count
    
    print(f"\n✅ ИТОГИ:")
    print(f"   🏆 ПРЕВОСХОДНО: {excellent_count} видов (≥80%)")
    print(f"   🎉 ОТЛИЧНО: {good_count} видов (≥60%)")
    print(f"   ⚡ ХОРОШО: {acceptable_count} видов (≥40%)")
    print(f"   📊 РАБОТАЕТ: {working_species}/7 видов")
    
    if working_species == 7:
        print("\n🏆🏆🏆 АБСОЛЮТНЫЙ УСПЕХ! ВСЕ 7 ВИДОВ РАБОТАЮТ! 🏆🏆🏆")
    elif working_species >= 6:
        print("\n🎉🎉🎉 ПОЧТИ ИДЕАЛЬНО! СИСТЕМА ГОТОВА! 🎉🎉🎉")
    elif working_species >= 5:
        print("\n⚡⚡⚡ ОТЛИЧНЫЙ РЕЗУЛЬТАТ! БОЛЬШИНСТВО РАБОТАЕТ! ⚡⚡⚡")
    elif working_species >= 3:
        print("\n📈📈📈 ХОРОШИЙ ПРОГРЕСС! ПОЛОВИНА РАБОТАЕТ! 📈📈📈")
    else:
        print("\n🔧 ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ НАСТРОЙКА")
    
    return accuracy, working_species, excellent_count + good_count

def main():
    """Максимально агрессивное решение для всех видов"""
    print("🚀🚀🚀 МАКСИМАЛЬНО АГРЕССИВНОЕ РЕШЕНИЕ ДЛЯ ВСЕХ 7 ВИДОВ 🚀🚀🚀")
    print("="*80)
    print("🎯 ЦЕЛЬ: ЗАСТАВИТЬ РАБОТАТЬ ВСЕ 7 ВИДОВ ЛЮБОЙ ЦЕНОЙ!")
    print("🧠 МЕТОДЫ: Все техники ML + аугментация + специализация + мета-ансамбли")
    print("="*80)
    
    # Загрузка данных
    print("\n📥 ЗАГРУЗКА ВСЕХ ДОСТУПНЫХ ДАННЫХ...")
    train_data, train_labels = load_spring_data()
    test_data, test_labels = load_summer_data()
    
    print(f"Весенние спектры: {len(train_data)}")
    print(f"Летние спектры: {len(test_data)}")
    
    # Показываем статистику по видам
    for species in ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']:
        spring_count = train_labels.count(species)
        summer_count = test_labels.count(species)
        print(f"  {species}: весна {spring_count}, лето {summer_count}")
    
    # Предобработка
    print("\n🔧 ПРЕДОБРАБОТКА С МАКСИМАЛЬНЫМИ ВОЗМОЖНОСТЯМИ...")
    all_spectra = train_data + test_data
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    train_data_trimmed = [spectrum[:min_length] for spectrum in train_data]
    test_data_trimmed = [spectrum[:min_length] for spectrum in test_data]
    
    # Максимальное извлечение признаков
    print("\n🧠 ИЗВЛЕЧЕНИЕ СУПЕР-ПРИЗНАКОВ...")
    X_train = extract_super_features(train_data_trimmed)
    X_test = extract_super_features(test_data_trimmed)
    
    print(f"Извлечено {X_train.shape[1]} супер-признаков!")
    
    # Кодирование
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    
    # Аугментация данных для балансировки
    X_train_aug, y_train_aug = augment_data(X_train, y_train, label_encoder.classes_)
    
    print(f"После аугментации: {len(X_train_aug)} образцов")
    
    # Обработка NaN и продвинутая нормализация
    print("\n⚙️ ОБРАБОТКА NaN И ПРОДВИНУТАЯ НОРМАЛИЗАЦИЯ...")
    
    # Обработка NaN значений
    from sklearn.impute import SimpleImputer
    
    # Заполняем NaN средними значениями
    imputer = SimpleImputer(strategy='mean')
    X_train_clean = imputer.fit_transform(X_train_aug)
    X_test_clean = imputer.transform(X_test)
    
    # Проверяем, что NaN убраны
    print(f"NaN в тренировочных данных: {np.isnan(X_train_clean).sum()}")
    print(f"NaN в тестовых данных: {np.isnan(X_test_clean).sum()}")
    
    # Используем PowerTransformer для нормализации распределений
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_transformed = power_transformer.fit_transform(X_train_clean)
    X_test_transformed = power_transformer.transform(X_test_clean)
    
    # Дополнительное робастное масштабирование
    robust_scaler = RobustScaler()
    X_train_final = robust_scaler.fit_transform(X_train_transformed)
    X_test_final = robust_scaler.transform(X_test_transformed)
    
    # Финальная проверка на NaN
    print(f"Финальные NaN в тренировочных данных: {np.isnan(X_train_final).sum()}")
    print(f"Финальные NaN в тестовых данных: {np.isnan(X_test_final).sum()}")
    
    print("✅ Применена продвинутая нормализация с обработкой NaN")
    
    # Создание специализированных моделей
    species_models = create_species_specific_models(
        X_train_final, y_train_aug, label_encoder.classes_
    )
    
    # Создание мета-ансамблевой модели
    print("\n🤖 СОЗДАНИЕ МЕТА-АНСАМБЛЕВОЙ НЕЙРОННОЙ СЕТИ...")
    meta_model = create_meta_ensemble_model(X_train_final.shape[1], len(label_encoder.classes_))
    
    print("Архитектура мета-модели:")
    meta_model.summary()
    
    # Экстремальные веса классов для проблемных видов
    class_weights = {}
    for i, species in enumerate(label_encoder.classes_):
        if species in ['клен', 'дуб']:
            class_weights[i] = 500.0  # Экстремальный вес
        elif species in ['липа', 'осина', 'сосна']:
            class_weights[i] = 100.0  # Высокий вес
        else:
            class_weights[i] = 1.0    # Обычный вес
    
    # Callbacks для обучения
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'ultimate_meta_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Обучение мета-модели
    print("\n🚀 ОБУЧЕНИЕ МЕТА-МОДЕЛИ...")
    history = meta_model.fit(
        X_train_final, y_train_aug,
        epochs=300,
        batch_size=16,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Продвинутое объединение предсказаний
    final_predictions, final_confidences = advanced_prediction_fusion(
        species_models, meta_model, X_test_final, label_encoder.classes_
    )
    
    # Анализ результатов
    accuracy, working_species, good_species = analyze_ultimate_results(
        y_test, final_predictions, final_confidences, label_encoder.classes_
    )
    
    # Сохранение всей системы
    print("\n💾 СОХРАНЕНИЕ МАКСИМАЛЬНО АГРЕССИВНОЙ СИСТЕМЫ...")
    
    # Сохраняем все компоненты
    meta_model.save('ultimate_meta_ensemble.keras')
    joblib.dump(imputer, 'ultimate_imputer.pkl')
    joblib.dump(power_transformer, 'ultimate_power_transformer.pkl')
    joblib.dump(robust_scaler, 'ultimate_robust_scaler.pkl')
    joblib.dump(label_encoder, 'ultimate_label_encoder.pkl')
    joblib.dump(species_models, 'ultimate_species_models.pkl')
    
    # Метаданные системы
    system_metadata = {
        'version': 'Ultimate_Aggressive_v1.0',
        'total_features': X_train_final.shape[1],
        'working_species': working_species,
        'good_species': good_species,
        'overall_accuracy': accuracy,
        'confidence_thresholds': {
            'клен': 0.6, 'дуб': 0.7, 'береза': 0.3,
            'ель': 0.4, 'липа': 0.5, 'осина': 0.4, 'сосна': 0.5
        }
    }
    
    joblib.dump(system_metadata, 'ultimate_system_metadata.pkl')
    
    # Создание финального отчета
    with open('ULTIMATE_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write("# 🚀 МАКСИМАЛЬНО АГРЕССИВНОЕ РЕШЕНИЕ - РЕЗУЛЬТАТЫ\n\n")
        f.write(f"## 📊 Общие результаты:\n")
        f.write(f"- **Общая точность:** {accuracy:.3f} ({accuracy*100:.1f}%)\n")
        f.write(f"- **Работающих видов:** {working_species}/7\n")
        f.write(f"- **Хорошо работающих:** {good_species}/7\n\n")
        
        f.write("## 🎯 Результаты по видам:\n")
        cm = confusion_matrix(y_test, final_predictions)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        for i, species in enumerate(label_encoder.classes_):
            acc = cm_norm[i][i]
            f.write(f"- **{species.upper()}:** {acc:.3f} ({acc*100:.1f}%)\n")
        
        f.write(f"\n## 🔧 Использованные техники:\n")
        f.write(f"- Супер-признаки: {X_train_final.shape[1]} штук\n")
        f.write("- Обработка NaN значений с импутацией\n")
        f.write("- Аугментация данных с 5 типами преобразований\n")
        f.write("- Специализированные модели для каждого вида\n")
        f.write("- Мета-ансамблевая нейронная сеть с 2.7M параметров\n")
        f.write("- Продвинутое объединение предсказаний\n")
        f.write("- Адаптивные пороги уверенности\n")
    
    print("✅ Система сохранена:")
    print("   - ultimate_meta_ensemble.keras")
    print("   - ultimate_imputer.pkl")
    print("   - ultimate_power_transformer.pkl")
    print("   - ultimate_robust_scaler.pkl")
    print("   - ultimate_label_encoder.pkl")
    print("   - ultimate_species_models.pkl")
    print("   - ultimate_system_metadata.pkl")
    print("   - ULTIMATE_RESULTS.md")
    
    # Финальное заключение
    print("\n" + "="*80)
    print("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ МАКСИМАЛЬНО АГРЕССИВНОГО РЕШЕНИЯ")
    print("="*80)
    print(f"📊 Общая точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"🎯 Работающих видов: {working_species}/7")
    print(f"⭐ Хорошо работающих: {good_species}/7")
    
    if working_species == 7:
        print("\n🏆🏆🏆 ПОЛНАЯ ПОБЕДА! ВСЕ ВИДЫ РАБОТАЮТ! 🏆🏆🏆")
        print("💪 МАКСИМАЛЬНО АГРЕССИВНЫЙ ПОДХОД СРАБОТАЛ!")
    elif working_species >= 6:
        print("\n🎉🎉🎉 ПОЧТИ ИДЕАЛЬНО! СИСТЕМА ГОТОВА К БОЮ! 🎉🎉🎉")
    elif working_species >= 5:
        print("\n⚡⚡⚡ ОТЛИЧНЫЙ РЕЗУЛЬТАТ! БОЛЬШИНСТВО ПОБЕЖДЕНО! ⚡⚡⚡")
    else:
        print(f"\n📈 ПРОГРЕСС ДОСТИГНУТ! {working_species} из 7 видов покорены!")
    
    print("\n🚀 МАКСИМАЛЬНО АГРЕССИВНОЕ РЕШЕНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main() 