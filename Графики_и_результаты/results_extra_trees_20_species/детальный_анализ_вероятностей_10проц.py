import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_extra_trees_model():
    """Загружает обученную модель Extra Trees"""
    model_path = "./extra_trees_20_species_model_20250724_110036.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
        return model
    else:
        print(f"❌ Файл модели не найден: {model_path}")
        return None

def load_data_and_preprocess():
    """Загружает и предобрабатывает данные для 20 видов"""
    # Используем тот же порядок, что и в оригинальном скрипте
    tree_types = [
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 
        'клен_ам', 'липа', 'лиственница', 'орех', 'осина', 'рябина', 
        'сирень', 'сосна', 'тополь_бальзамический', 'тополь_черный', 
        'туя', 'черемуха', 'ясень'
    ]
    
    all_data = []
    all_labels = []
    
    print("🌿 Загрузка данных для 20 видов...")
    
    spring_folder = "../Спектры, весенний период, 20 видов"
    
    for tree_type in tree_types:
        folder_path = os.path.join(spring_folder, tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path, header=None)
                    spectrum = df.iloc[:, 1].values  # Вторая колонка - спектр
                    spectrum = spectrum[~pd.isna(spectrum)]  # Убираем NaN
                    
                    if len(spectrum) > 0:
                        all_data.append(spectrum)
                        all_labels.append(tree_type)
                except Exception as e:
                    print(f"Ошибка при загрузке {file_path}: {e}")
    
    if len(all_data) == 0:
        print("❌ Не удалось загрузить данные")
        return None, None, None
    
    print(f"✅ Загружено {len(all_data)} образцов")
    
    # Предобработка спектров
    X = preprocess_spectra(all_data)
    y = np.array(all_labels)
    
    # Извлечение признаков
    X_features = extract_enhanced_features(X)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test, label_encoder.classes_

def preprocess_spectra(spectra_list):
    """Предобработка спектров"""
    # Находим минимальную длину
    min_length = min(len(spectrum) for spectrum in spectra_list)
    print(f"   📏 Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до минимальной длины
    processed_spectra = []
    for spectrum in spectra_list:
        truncated = spectrum[:min_length]
        processed_spectra.append(truncated)
    
    # Преобразуем в numpy array
    X = np.array(processed_spectra)
    print(f"   📊 Финальная форма данных: {X.shape}")
    
    return X

def extract_enhanced_features(X):
    """Извлекает расширенные признаки из спектров"""
    print("⚙️ ИЗВЛЕЧЕНИЕ РАСШИРЕННЫХ ПРИЗНАКОВ...")
    
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

def analyze_detailed_probabilities(model, X_test, y_test, tree_types, target_species=['осина', 'сирень'], noise_level=0.1):
    """Анализирует детальные вероятности для целевых видов"""
    
    print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ВЕРОЯТНОСТЕЙ ПРИ {noise_level*100}% ШУМА")
    print("="*70)
    
    for target_species_name in target_species:
        print(f"\n📊 АНАЛИЗ ДЛЯ ВИДА: {target_species_name.upper()}")
        print("-" * 50)
        
        # Находим индекс целевого вида
        target_idx = np.where(tree_types == target_species_name)[0]
        if len(target_idx) == 0:
            print(f"❌ Вид {target_species_name} не найден в списке")
            continue
        
        target_idx = target_idx[0]
        
        # Находим образцы целевого вида
        target_samples_mask = (y_test == target_idx)
        target_samples = X_test[target_samples_mask]
        
        if len(target_samples) == 0:
            print(f"❌ Нет образцов для вида {target_species_name}")
            continue
        
        print(f"📈 Найдено {len(target_samples)} образцов {target_species_name}")
        
        # Применяем шум
        noise = np.random.normal(0, noise_level, target_samples.shape).astype(np.float32)
        target_samples_noisy = target_samples + noise
        
        # Получаем вероятности для всех образцов
        probabilities = model.predict_proba(target_samples_noisy)
        
        # Создаем DataFrame с вероятностями
        # Используем правильные названия классов в том же порядке, что и модель
        prob_df = pd.DataFrame(probabilities, columns=tree_types)
        prob_df.insert(0, 'Образец', range(1, len(prob_df) + 1))
        
        print(f"\n📋 ДЕТАЛЬНЫЕ ВЕРОЯТНОСТИ ДЛЯ {target_species_name.upper()}:")
        print("="*70)
        
        # Показываем первые 10 образцов
        print(prob_df.head(10).to_string(index=False, float_format='%.4f'))
        
        # Вычисляем средние вероятности
        mean_probs = prob_df.iloc[:, 1:].mean()
        
        print(f"\n📊 СРЕДНИЕ ВЕРОЯТНОСТИ ДЛЯ {target_species_name.upper()}:")
        print("-" * 50)
        for species, prob in mean_probs.items():
            print(f"{species:15}: {prob:.4f}")
        
        # Создаем матрицу с максимальными вероятностями (1 для максимума, 0 для остальных)
        max_prob_matrix = np.zeros_like(probabilities)
        max_indices = np.argmax(probabilities, axis=1)
        max_prob_matrix[np.arange(len(probabilities)), max_indices] = 1
        
        # Создаем DataFrame для максимальных вероятностей
        max_prob_df = pd.DataFrame(max_prob_matrix, columns=tree_types)
        max_prob_df.insert(0, 'Образец', range(1, len(max_prob_df) + 1))
        
        print(f"\n🎯 МАТРИЦА МАКСИМАЛЬНЫХ ВЕРОЯТНОСТЕЙ (1/0) ДЛЯ {target_species_name.upper()}:")
        print("="*70)
        print(max_prob_df.to_string(index=False))
        
        # Вычисляем средние для максимальных вероятностей
        mean_max_probs = max_prob_df.iloc[:, 1:].mean()
        
        print(f"\n📈 СРЕДНИЕ МАКСИМАЛЬНЫХ ВЕРОЯТНОСТЕЙ ДЛЯ {target_species_name.upper()}:")
        print("-" * 50)
        for species, prob in mean_max_probs.items():
            if prob > 0:
                print(f"{species:15}: {prob:.4f}")
        
        # Сохраняем результаты в файл
        output_filename = f"детальные_вероятности_{target_species_name}_10проц.csv"
        prob_df.to_csv(output_filename, index=False, float_format='%.4f')
        print(f"\n💾 Детальные вероятности сохранены в: {output_filename}")
        
        # Сохраняем матрицу максимальных вероятностей
        max_output_filename = f"максимальные_вероятности_{target_species_name}_10проц.csv"
        max_prob_df.to_csv(max_output_filename, index=False)
        print(f"💾 Матрица максимальных вероятностей сохранена в: {max_output_filename}")
        
        # Создаем тепловую карту
        plt.figure(figsize=(16, 8))
        
        # Подграфик 1: Детальные вероятности
        plt.subplot(1, 2, 1)
        sns.heatmap(prob_df.iloc[:, 1:].T, 
                   cmap='Blues', 
                   cbar_kws={'label': 'Вероятность'},
                   xticklabels=range(1, len(prob_df) + 1),
                   yticklabels=tree_types)
        plt.title(f'Детальные вероятности - {target_species_name} (10% шум)')
        plt.xlabel('Номер образца')
        plt.ylabel('Вид дерева')
        
        # Подграфик 2: Максимальные вероятности
        plt.subplot(1, 2, 2)
        sns.heatmap(max_prob_df.iloc[:, 1:].T, 
                   cmap='Reds', 
                   cbar_kws={'label': 'Максимальная вероятность'},
                   xticklabels=range(1, len(max_prob_df) + 1),
                   yticklabels=tree_types)
        plt.title(f'Максимальные вероятности - {target_species_name} (10% шум)')
        plt.xlabel('Номер образца')
        plt.ylabel('Вид дерева')
        
        plt.tight_layout()
        plt.savefig(f'тепловая_карта_вероятностей_{target_species_name}_10проц.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Тепловая карта сохранена: тепловая_карта_вероятностей_{target_species_name}_10проц.png")

def main():
    """Основная функция"""
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ВЕРОЯТНОСТЕЙ КЛАССИФИКАЦИИ")
    print("="*70)
    
    # Загружаем модель
    model = load_extra_trees_model()
    if model is None:
        return
    
    # Загружаем данные
    X_test, y_test, tree_types = load_data_and_preprocess()
    if X_test is None:
        return
    
    # Анализируем детальные вероятности для осины и сирени при 10% шуме
    analyze_detailed_probabilities(model, X_test, y_test, tree_types, 
                                 target_species=['осина', 'сирень'], 
                                 noise_level=0.1)
    
    print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    import glob
    main() 