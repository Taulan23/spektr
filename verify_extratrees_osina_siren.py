import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob
from datetime import datetime

def load_spectral_data_20_species():
    """Загружает данные для 20 видов деревьев (все доступные файлы)"""
    data = []
    labels = []
    
    # Список 20 видов из папки "Спектры, весенний период, 20 видов"
    species = [
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 
        'клен_ам', 'липа', 'лиственница', 'орех', 'осина', 'рябина', 
        'сирень', 'сосна', 'тополь_бальзамический', 'тополь_черный', 
        'туя', 'черемуха', 'ясень'
    ]
    
    for species_name in species:
        print(f"Загрузка данных для {species_name}...")
        
        folder_path = f'Спектры, весенний период, 20 видов/{species_name}'
        
        # Обработка вложенной папки для клен_ам
        if species_name == 'клен_ам':
            folder_path = f'Спектры, весенний период, 20 видов/клен_ам/клен_ам'
        
        files = glob.glob(f'{folder_path}/*.xlsx')
        
        print(f"  Найдено {len(files)} файлов")
        
        # Загружаем ВСЕ файлы для каждого вида
        for file in files:
            try:
                df = pd.read_excel(file)
                spectral_data = df.iloc[:, 1:].values.flatten()
                
                if len(spectral_data) > 0 and not np.any(np.isnan(spectral_data)):
                    spectral_data = (spectral_data - np.min(spectral_data)) / (np.max(spectral_data) - np.min(spectral_data))
                    data.append(spectral_data)
                    labels.append(species_name)
            except Exception as e:
                print(f"  Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def analyze_spectral_similarity(X, y, label_encoder, target_classes=['осина', 'сирень']):
    """Анализирует сходство спектров между классами"""
    print(f"\nАНАЛИЗ СХОДСТВА СПЕКТРОВ МЕЖДУ КЛАССАМИ:")
    print("=" * 60)
    
    # Находим индексы целевых классов
    target_indices = []
    for class_name in target_classes:
        if class_name in label_encoder.classes_:
            idx = np.where(label_encoder.classes_ == class_name)[0][0]
            target_indices.append(idx)
            print(f"Найден класс '{class_name}' с индексом {idx}")
    
    if len(target_indices) != 2:
        print("Ошибка: не найдены оба целевых класса")
        return
    
    # Получаем данные для целевых классов
    class1_idx, class2_idx = target_indices
    class1_name, class2_name = target_classes
    
    class1_mask = (y == class1_idx)
    class2_mask = (y == class2_idx)
    
    class1_data = X[class1_mask]
    class2_data = X[class2_mask]
    
    print(f"\nДанные для анализа:")
    print(f"  {class1_name}: {len(class1_data)} спектров")
    print(f"  {class2_name}: {len(class2_data)} спектров")
    
    # Анализ средних спектров
    class1_mean = np.mean(class1_data, axis=0)
    class2_mean = np.mean(class2_data, axis=0)
    
    # Корреляция между средними спектрами
    correlation = np.corrcoef(class1_mean, class2_mean)[0,1]
    print(f"\nКорреляция между средними спектрами: {correlation:.6f}")
    
    # Евклидово расстояние между средними спектрами
    euclidean_dist = np.linalg.norm(class1_mean - class2_mean)
    print(f"Евклидово расстояние между средними спектрами: {euclidean_dist:.6f}")
    
    # Анализ всех попарных корреляций
    all_correlations = []
    for i in range(len(class1_data)):
        for j in range(len(class2_data)):
            corr = np.corrcoef(class1_data[i], class2_data[j])[0,1]
            all_correlations.append(corr)
    
    print(f"\nСтатистика попарных корреляций:")
    print(f"  Средняя корреляция: {np.mean(all_correlations):.6f}")
    print(f"  Медианная корреляция: {np.median(all_correlations):.6f}")
    print(f"  Минимальная корреляция: {np.min(all_correlations):.6f}")
    print(f"  Максимальная корреляция: {np.max(all_correlations):.6f}")
    print(f"  Стандартное отклонение: {np.std(all_correlations):.6f}")
    
    # Проверяем, насколько спектры похожи
    high_correlation_count = sum(1 for corr in all_correlations if corr > 0.95)
    print(f"\nСпектры с корреляцией > 0.95: {high_correlation_count} из {len(all_correlations)} ({high_correlation_count/len(all_correlations)*100:.1f}%)")
    
    # Визуализация
    plt.figure(figsize=(15, 10))
    
    # График 1: Средние спектры
    plt.subplot(2, 2, 1)
    plt.plot(class1_mean, label=class1_name, linewidth=2)
    plt.plot(class2_mean, label=class2_name, linewidth=2)
    plt.title(f'Средние спектры {class1_name} и {class2_name}')
    plt.xlabel('Длина волны')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Гистограмма корреляций
    plt.subplot(2, 2, 2)
    plt.hist(all_correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(all_correlations), color='red', linestyle='--', label=f'Среднее: {np.mean(all_correlations):.4f}')
    plt.title(f'Распределение корреляций между {class1_name} и {class2_name}')
    plt.xlabel('Корреляция')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Разность средних спектров
    plt.subplot(2, 2, 3)
    diff = class1_mean - class2_mean
    plt.plot(diff, color='green', linewidth=2)
    plt.title(f'Разность средних спектров ({class1_name} - {class2_name})')
    plt.xlabel('Длина волны')
    plt.ylabel('Разность')
    plt.grid(True, alpha=0.3)
    
    # График 4: Относительная разность
    plt.subplot(2, 2, 4)
    relative_diff = np.abs(diff) / (np.abs(class1_mean) + np.abs(class2_mean) + 1e-8)
    plt.plot(relative_diff, color='orange', linewidth=2)
    plt.title(f'Относительная разность спектров')
    plt.xlabel('Длина волны')
    plt.ylabel('Относительная разность')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'spectral_similarity_analysis_{class1_name}_{class2_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'correlation': correlation,
        'euclidean_distance': euclidean_dist,
        'mean_pairwise_correlation': np.mean(all_correlations),
        'high_correlation_percentage': high_correlation_count/len(all_correlations)*100
    }

def train_extratrees_and_analyze(X, y, label_encoder, target_classes=['осина', 'сирень']):
    """Обучает ExtraTrees и анализирует результаты для целевых классов"""
    print(f"\nОБУЧЕНИЕ EXTRATREES И АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    
    # Разделяем данные 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Разделение данных:")
    print(f"  Обучающая выборка: {len(X_train)} (80%)")
    print(f"  Тестовая выборка: {len(X_test)} (20%)")
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем ExtraTrees
    print("\nОбучение ExtraTrees...")
    et_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    et_model.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred = et_model.predict(X_test_scaled)
    y_pred_proba = et_model.predict_proba(X_test_scaled)
    
    # Общая точность
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nОбщая точность: {accuracy:.4f}")
    
    # Анализ для целевых классов
    target_indices = []
    for class_name in target_classes:
        if class_name in label_encoder.classes_:
            idx = np.where(label_encoder.classes_ == class_name)[0][0]
            target_indices.append(idx)
    
    if len(target_indices) == 2:
        class1_idx, class2_idx = target_indices
        class1_name, class2_name = target_classes
        
        print(f"\nАнализ для {class1_name} и {class2_name}:")
        
        # Точность для каждого класса
        for i, class_idx in enumerate([class1_idx, class2_idx]):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_test == class_idx) & (y_pred == class_idx)) / np.sum(class_mask)
                class_name = label_encoder.classes_[class_idx]
                print(f"  {class_name}: {class_accuracy:.4f} ({np.sum((y_test == class_idx) & (y_pred == class_idx))}/{np.sum(class_mask)})")
        
        # Анализ путаницы между классами
        confusion = confusion_matrix(y_test, y_pred)
        confusion_between_targets = confusion[class1_idx, class2_idx] + confusion[class2_idx, class1_idx]
        total_targets = np.sum(y_test == class1_idx) + np.sum(y_test == class2_idx)
        
        print(f"\nПутаница между {class1_name} и {class2_name}:")
        print(f"  Ошибок классификации: {confusion_between_targets}")
        print(f"  Всего образцов: {total_targets}")
        print(f"  Процент ошибок: {confusion_between_targets/total_targets*100:.2f}%")
        
        # Анализ уверенности модели
        target_mask = (y_test == class1_idx) | (y_test == class2_idx)
        target_proba = y_pred_proba[target_mask]
        target_max_proba = np.max(target_proba, axis=1)
        
        print(f"\nУверенность модели для {class1_name} и {class2_name}:")
        print(f"  Средняя уверенность: {np.mean(target_max_proba):.4f}")
        print(f"  Медианная уверенность: {np.median(target_max_proba):.4f}")
        print(f"  Минимальная уверенность: {np.min(target_max_proba):.4f}")
        print(f"  Максимальная уверенность: {np.max(target_max_proba):.4f}")
    
    return accuracy, y_pred, y_pred_proba

def main():
    """Основная функция для проверки ExtraTrees для осины и сирени"""
    print("ПРОВЕРКА EXTRATREES ДЛЯ ОСИНЫ И СИРЕНИ")
    print("=" * 50)
    
    # Загружаем данные
    X, y = load_spectral_data_20_species()
    
    print(f"\nЗагружено {len(X)} спектров")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Анализируем сходство спектров
    similarity_results = analyze_spectral_similarity(X, y_encoded, label_encoder, ['осина', 'сирень'])
    
    # Обучаем ExtraTrees и анализируем результаты
    accuracy, y_pred, y_pred_proba = train_extratrees_and_analyze(X, y_encoded, label_encoder, ['осина', 'сирень'])
    
    # Выводы
    print(f"\nВЫВОДЫ:")
    print("=" * 50)
    
    if similarity_results['mean_pairwise_correlation'] > 0.9:
        print("⚠️  ВНИМАНИЕ: Очень высокая корреляция между спектрами осины и сирени")
        print("   Это может объяснить, почему модель показывает 100% точность")
        print("   Спектры практически идентичны!")
    elif similarity_results['mean_pairwise_correlation'] > 0.8:
        print("⚠️  Высокая корреляция между спектрами осины и сирени")
        print("   Модель может переобучаться на очень похожих данных")
    else:
        print("✅ Спектры осины и сирени достаточно различны")
    
    if accuracy > 0.95:
        print("⚠️  Очень высокая общая точность может указывать на проблемы с данными")
    else:
        print("✅ Общая точность выглядит реалистично")

if __name__ == "__main__":
    main() 