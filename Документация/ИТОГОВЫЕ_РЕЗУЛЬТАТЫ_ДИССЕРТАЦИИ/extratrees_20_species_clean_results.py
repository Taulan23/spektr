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
    """Загружает данные для 20 видов деревьев"""
    data = []
    labels = []
    
    # Список 20 видов
    species = [
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 
        'клен_ам', 'липа', 'лиственница', 'орех', 'осина', 'рябина', 
        'сирень', 'сосна', 'тополь_бальзамический', 'тополь_черный', 
        'туя', 'черемуха', 'ясень'
    ]
    
    for species_name in species:
        print(f"Загрузка данных для {species_name}...")
        
        folder_path = f'Спектры, весенний период, 20 видов/{species_name}'
        files = glob.glob(f'{folder_path}/*.xlsx')
        
        for file in files:
            try:
                df = pd.read_excel(file)
                # Предполагаем, что спектральные данные начинаются со второй колонки
                spectral_data = df.iloc[:, 1:].values.flatten()
                if len(spectral_data) > 0:
                    data.append(spectral_data)
                    labels.append(species_name)
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def add_noise(X, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return X
    
    # Шум как процент от стандартного отклонения данных
    noise_std = noise_level / 100.0 * np.std(X)
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Создает матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def save_parameters_to_file(filename):
    """Сохраняет параметры модели в текстовый файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ПАРАМЕТРЫ МОДЕЛИ EXTRA TREES ДЛЯ 20 ВИДОВ\n")
        f.write("=" * 50 + "\n\n")
        f.write("Модель: ExtraTreesClassifier\n")
        f.write("n_estimators: 1000\n")
        f.write("max_depth: None\n")
        f.write("min_samples_split: 2\n")
        f.write("min_samples_leaf: 1\n")
        f.write("max_features: 'sqrt'\n")
        f.write("bootstrap: False\n")
        f.write("random_state: 42\n\n")
        
        f.write("ПАРАМЕТРЫ ДАННЫХ:\n")
        f.write("-" * 20 + "\n")
        f.write("Количество видов: 20\n")
        f.write("Разделение данных: 80% обучение, 20% тест\n")
        f.write("Стратификация: Да\n")
        f.write("Предобработка: StandardScaler\n\n")
        
        f.write("ПАРАМЕТРЫ ШУМА:\n")
        f.write("-" * 20 + "\n")
        f.write("Тип шума: Аддитивный гауссовский\n")
        f.write("Среднее: 0\n")
        f.write("Стандартное отклонение: процент от std данных\n")
        f.write("Уровни шума: 0%, 1%, 5%, 10%\n\n")
        
        f.write("ВОСПРОИЗВОДИМОСТЬ:\n")
        f.write("-" * 20 + "\n")
        f.write("np.random.seed(42)\n")
        f.write("random_state=42 в train_test_split\n")
        f.write("random_state=42 в ExtraTreesClassifier\n")

def main():
    """Основная функция"""
    print("КЛАССИФИКАЦИЯ 20 ВИДОВ ДЕРЕВЬЕВ - EXTRA TREES")
    print("=" * 60)
    
    # Установка seed для воспроизводимости
    np.random.seed(42)
    
    # Загрузка данных
    X, y = load_spectral_data_20_species()
    
    if len(X) == 0:
        print("Не удалось загрузить данные!")
        return
    
    print(f"Загружено {len(X)} спектров для {len(np.unique(y))} видов")
    
    # Предобработка
    lengths = [len(s) for s in X]
    target_length = min(lengths)
    X_processed = np.array([spectrum[:target_length] for spectrum in X], dtype=np.float32)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"Форма данных: {X_processed.shape}")
    print(f"Классы: {class_names}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создание и обучение модели
    print("\nОбучение модели Extra Trees...")
    model = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Тестирование на разных уровнях шума
    noise_levels = [0, 1, 5, 10]
    results = {}
    
    for noise_level in noise_levels:
        print(f"\nТестирование с {noise_level}% шумом...")
        
        # Добавление шума к тестовым данным
        X_test_noisy = add_noise(X_test_scaled, noise_level)
        
        # Предсказание
        y_pred = model.predict(X_test_noisy)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[noise_level] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true': y_test
        }
        
        print(f"Точность: {accuracy:.4f}")
        
        # Создание матрицы ошибок
        if noise_level in [1, 5, 10]:
            title = f'Матрица ошибок - {noise_level}% шум (точность: {accuracy:.4f})'
            filename = f'confusion_matrix_20_species_{noise_level}percent.png'
            plot_confusion_matrix(y_test, y_pred, class_names, title, filename)
            print(f"Матрица сохранена: {filename}")
    
    # Сохранение параметров
    save_parameters_to_file('parameters_20_species_extratrees.txt')
    print("\nПараметры сохранены: parameters_20_species_extratrees.txt")
    
    # Итоговый отчет
    print("\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("-" * 30)
    for noise_level in noise_levels:
        acc = results[noise_level]['accuracy']
        print(f"{noise_level}% шум: {acc:.4f}")
    
    print(f"\nФайлы созданы:")
    print("- confusion_matrix_20_species_1percent.png")
    print("- confusion_matrix_20_species_5percent.png") 
    print("- confusion_matrix_20_species_10percent.png")
    print("- parameters_20_species_extratrees.txt")

if __name__ == "__main__":
    main() 