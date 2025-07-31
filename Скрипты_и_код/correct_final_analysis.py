import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

def load_20_species_data():
    """Загружает данные для 20 видов деревьев точно как в оригинальном скрипте"""
    base_path = "../Исходные_данные/Спектры, весенний период, 20 видов"
    
    all_data = []
    all_labels = []
    
    # Получаем список всех папок и сортируем их
    all_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    print(f"Найдено папок: {len(all_folders)}")
    print("Папки:", all_folders)
    
    for folder in all_folders:
        folder_path = os.path.join(base_path, folder)
        
        # Обработка клен_ам
        if folder == "клен_ам":
            species_folder = os.path.join(folder_path, "клен_ам")
        else:
            species_folder = folder_path
            
        print(f"Обрабатываем: {folder} -> {species_folder}")
        
        if os.path.exists(species_folder):
            files = [f for f in os.listdir(species_folder) if f.endswith('.xlsx')]
            print(f"  Найдено файлов: {len(files)}")
            
            for file in files:
                file_path = os.path.join(species_folder, file)
                try:
                    df = pd.read_excel(file_path, header=None)
                    spectrum = df.iloc[:, 1].values  # Вторая колонка - спектр
                    spectrum = spectrum[~pd.isna(spectrum)]  # Убираем NaN
                    
                    all_data.append(spectrum)
                    all_labels.append(folder)
                    
                except Exception as e:
                    print(f"Ошибка при чтении {file_path}: {e}")
        else:
            print(f"Папка не найдена: {species_folder}")
    
    print(f"Всего загружено образцов: {len(all_data)}")
    print(f"Уникальные метки: {sorted(set(all_labels))}")
    
    return all_data, all_labels

def preprocess_spectra(spectra_list):
    """Предобработка спектров"""
    print("Предобработка спектров...")
    
    # Находим минимальную длину
    min_length = min(len(spectrum) for spectrum in spectra_list)
    print(f"Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до минимальной длины
    processed_spectra = []
    for spectrum in spectra_list:
        truncated = spectrum[:min_length]
        processed_spectra.append(truncated)
    
    # Преобразуем в numpy array
    X = np.array(processed_spectra)
    print(f"Финальная форма данных: {X.shape}")
    
    return X

def extract_enhanced_features(X):
    """Извлекает расширенные признаки из спектров точно как в оригинальном скрипте"""
    print("Извлечение признаков...")
    
    features_list = []
    
    for spectrum in X:
        features = []
        
        # Базовые статистики
        features.extend([
            np.mean(spectrum),
            np.std(spectrum),
            np.median(spectrum),
            np.min(spectrum),
            np.max(spectrum),
            np.ptp(spectrum),  # peak-to-peak
            np.var(spectrum)
        ])
        
        # Квантили
        quantiles = np.percentile(spectrum, [10, 25, 75, 90])
        features.extend(quantiles)
        
        # Моменты распределения
        features.extend([
            np.sum(spectrum),
            np.sum(spectrum**2),
            np.sum(spectrum**3),
            np.sum(spectrum**4)
        ])
        
        # Спектральные признаки
        if len(spectrum) > 1:
            diff1 = np.diff(spectrum)
            diff2 = np.diff(diff1) if len(diff1) > 1 else [0]
            
            features.extend([
                np.mean(diff1),
                np.std(diff1),
                np.mean(diff2) if len(diff2) > 0 else 0,
                np.std(diff2) if len(diff2) > 0 else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Энергетические признаки
        if len(spectrum) > 10:
            # Разделяем спектр на части
            n_parts = 5
            part_size = len(spectrum) // n_parts
            
            for i in range(n_parts):
                start_idx = i * part_size
                end_idx = start_idx + part_size if i < n_parts - 1 else len(spectrum)
                part = spectrum[start_idx:end_idx]
                
                features.extend([
                    np.mean(part),
                    np.std(part),
                    np.max(part),
                    np.min(part)
                ])
        else:
            # Заполняем нулями если спектр слишком короткий
            features.extend([0] * (5 * 4))
        
        # Дополнительные признаки
        features.extend([
            np.sum(spectrum > np.mean(spectrum)),  # количество точек выше среднего
            np.sum(spectrum < np.mean(spectrum)),  # количество точек ниже среднего
            len(spectrum),  # длина спектра
            np.argmax(spectrum),  # индекс максимума
            np.argmin(spectrum),  # индекс минимума
        ])
        
        features_list.append(features)
    
    return np.array(features_list)

def create_correct_excel_analysis(model, X_test, y_test, scaler, label_encoder, noise_level=0.10):
    """Создает правильный Excel анализ для осины и сирени"""
    
    # Получаем правильный порядок классов из label_encoder
    tree_types = label_encoder.classes_
    print(f"Порядок классов в модели: {tree_types}")
    
    # Находим индексы осины и сирени
    aspen_idx = np.where(tree_types == 'осина')[0][0]
    lilac_idx = np.where(tree_types == 'сирень')[0][0]
    
    print(f"Индекс осины: {aspen_idx}, индекс сирени: {lilac_idx}")
    
    # Фильтруем тестовые данные только для осины и сирени
    # y_test содержит закодированные метки, поэтому используем индексы
    aspen_mask = y_test == aspen_idx
    lilac_mask = y_test == lilac_idx
    
    aspen_samples = X_test[aspen_mask]
    lilac_samples_data = X_test[lilac_mask]
    
    print(f"Найдено образцов осины: {len(aspen_samples)}")
    print(f"Найдено образцов сирени: {len(lilac_samples_data)}")
    
    if len(aspen_samples) == 0 and len(lilac_samples_data) == 0:
        print("ОШИБКА: Не найдено образцов осины и сирени в тестовой выборке!")
        return None
    
    # Объединяем данные
    combined_samples = np.vstack([aspen_samples, lilac_samples_data])
    combined_labels = np.concatenate([['осина'] * len(aspen_samples), ['сирень'] * len(lilac_samples_data)])
    
    # Применяем шум к данным
    noisy_samples = combined_samples + np.random.normal(0, noise_level, combined_samples.shape)
    
    # Нормализуем данные
    noisy_samples_scaled = scaler.transform(noisy_samples)
    
    # Получаем вероятности
    probabilities = model.predict_proba(noisy_samples_scaled)
    
    # Создаем DataFrame для детального анализа
    detailed_data = []
    
    for i, (sample, label, probs) in enumerate(zip(noisy_samples, combined_labels, probabilities)):
        # Находим максимальную вероятность
        max_prob_idx = np.argmax(probs)
        
        # Создаем строку данных
        row_data = {
            'Образец': f"{label} {i+1:02d}",
            'Истинный_класс': label
        }
        
        # Добавляем исходные вероятности
        for j, species in enumerate(tree_types):
            row_data[f'вероятность_{species}'] = probs[j]
        
        # Добавляем максимальные вероятности (1/0)
        for j, species in enumerate(tree_types):
            row_data[f'макс_вероятность_{species}'] = 1.0 if j == max_prob_idx else 0.0
        
        detailed_data.append(row_data)
    
    # Создаем DataFrame
    df_detailed = pd.DataFrame(detailed_data)
    
    # Создаем Excel файл с несколькими листами
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../Результаты_Extra_Trees_20_видов/correct_dissertation_analysis_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Лист 1: Детальный анализ
        df_detailed.to_excel(writer, sheet_name='Детальный_анализ', index=False)
        
        # Лист 2: Только вероятности
        prob_cols = ['Образец', 'Истинный_класс'] + [col for col in df_detailed.columns if col.startswith('вероятность_')]
        df_detailed[prob_cols].to_excel(writer, sheet_name='Вероятности', index=False)
        
        # Лист 3: Только максимальные вероятности
        max_prob_cols = ['Образец', 'Истинный_класс'] + [col for col in df_detailed.columns if col.startswith('макс_вероятность_')]
        df_detailed[max_prob_cols].to_excel(writer, sheet_name='Максимальные_вероятности', index=False)
        
        # Лист 4: Статистика
        stats_data = []
        
        # Статистика для осины
        aspen_data = df_detailed[df_detailed['Истинный_класс'] == 'осина']
        if len(aspen_data) > 0:
            aspen_correct = aspen_data[f'макс_вероятность_осина'].sum()
            aspen_accuracy = aspen_correct / len(aspen_data)
            stats_data.append({
                'Вид': 'Осина',
                'Количество_образцов': len(aspen_data),
                'Правильно_классифицировано': aspen_correct,
                'Точность': aspen_accuracy,
                'Средняя_вероятность_осины': aspen_data[f'вероятность_осина'].mean(),
                'Ожидаемая_точность_из_CM': 0.9355
            })
        
        # Статистика для сирени
        lilac_data = df_detailed[df_detailed['Истинный_класс'] == 'сирень']
        if len(lilac_data) > 0:
            lilac_correct = lilac_data[f'макс_вероятность_сирень'].sum()
            lilac_accuracy = lilac_correct / len(lilac_data)
            stats_data.append({
                'Вид': 'Сирень',
                'Количество_образцов': len(lilac_data),
                'Правильно_классифицировано': lilac_correct,
                'Точность': lilac_accuracy,
                'Средняя_вероятность_сирени': lilac_data[f'вероятность_сирень'].mean(),
                'Ожидаемая_точность_из_CM': 1.0
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Статистика', index=False)
    
    print(f"Excel файл сохранен: {filename}")
    print("\nСТАТИСТИКА:")
    for _, row in df_stats.iterrows():
        print(f"{row['Вид']}: {row['Точность']:.4f} ({row['Правильно_классифицировано']}/{row['Количество_образцов']})")
        print(f"  Ожидаемая из CM: {row['Ожидаемая_точность_из_CM']:.4f}")
    
    return filename

def main():
    print("=== ЗАГРУЗКА ПРЕДОБУЧЕННОЙ МОДЕЛИ ===")
    
    # Загружаем предобученную модель
    model_path = "../Результаты_Extra_Trees_20_видов/extra_trees_1712_model_20250731_094732.pkl"
    
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл модели не найден: {model_path}")
        return
    
    print(f"Загружаем модель из: {model_path}")
    model = joblib.load(model_path)
    
    print("Модель загружена успешно!")
    print(f"Параметры модели: {model.get_params()}")
    
    print("\n=== ЗАГРУЗКА ДАННЫХ ===")
    all_data, all_labels = load_20_species_data()
    
    print(f"Загружено данных: {len(all_data)}")
    print(f"Уникальные метки: {sorted(set(all_labels))}")
    
    # Предобработка спектров
    X_spectra = preprocess_spectra(all_data)
    
    # Извлечение признаков
    X_features = extract_enhanced_features(X_spectra)
    
    # Создаем label_encoder и scaler заново (как в оригинальном скрипте)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(all_labels)
    
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Создаем scaler заново
    scaler = StandardScaler()
    scaler.fit(X_train)  # Обучаем на train данных
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    print("\n=== СОЗДАНИЕ ПРАВИЛЬНОГО АНАЛИЗА ===")
    excel_file = create_correct_excel_analysis(model, X_test, y_test, scaler, label_encoder, noise_level=0.10)
    
    print(f"\nАнализ завершен! Файл: {excel_file}")

if __name__ == "__main__":
    main() 