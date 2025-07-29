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
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 'клен_ам',
        'липа', 'лиственница', 'орех', 'осина', 'рябина', 'сирень', 'сосна',
        'тополь_бальзамический', 'тополь_черный', 'туя', 'черемуха', 'ясень'
    ]
    
    for species_name in species:
        print(f"Загрузка данных для {species_name}...")
        
        folder_path = f'Спектры, весенний период, 20 видов/{species_name}'
        files = glob.glob(f'{folder_path}/*.xlsx')
        
        # Берем первые 30 файлов для каждого вида
        for file in files[:30]:
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

def add_gaussian_noise(data, noise_level):
    """Добавляет гауссовский шум к данным"""
    if noise_level == 0:
        return data
    
    # Вычисляем стандартное отклонение на основе уровня шума
    std_dev = noise_level / 100.0 * np.std(data)
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

def create_excel_output_for_osina_siren(model, X_test, y_test, class_names, scaler, noise_level=10):
    """Создает Excel файл с результатами для осины и сирени"""
    
    # Находим индексы осины и сирени
    osina_idx = np.where(class_names == 'осина')[0][0]
    siren_idx = np.where(class_names == 'сирень')[0][0]
    
    # Добавляем шум к тестовым данным
    X_test_noisy = add_gaussian_noise(X_test, noise_level)
    
    # Получаем предсказания
    y_pred_proba = model.predict_proba(X_test_noisy)
    y_pred = model.predict(X_test_noisy)
    
    # Создаем DataFrame для результатов
    results_data = []
    
    # Обрабатываем осину (30 тестовых образцов)
    osina_test_indices = np.where(y_test == osina_idx)[0]
    for i, idx in enumerate(osina_test_indices[:30]):  # Берем первые 30
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = np.max(y_pred_proba[idx])
        
        # Получаем вероятности для всех классов
        probabilities = y_pred_proba[idx]
        
        row = {
            'Номер_образца': f'Осина_{i+1:02d}',
            'Истинный_класс': true_label,
            'Предсказанный_класс': pred_label,
            'Уверенность': f'{confidence:.4f}',
            'Правильно': 'Да' if true_label == pred_label else 'Нет'
        }
        
        # Добавляем вероятности для всех классов
        for j, class_name in enumerate(class_names):
            row[f'Вероятность_{class_name}'] = f'{probabilities[j]:.4f}'
        
        results_data.append(row)
    
    # Обрабатываем сирень (30 тестовых образцов)
    siren_test_indices = np.where(y_test == siren_idx)[0]
    for i, idx in enumerate(siren_test_indices[:30]):  # Берем первые 30
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = np.max(y_pred_proba[idx])
        
        # Получаем вероятности для всех классов
        probabilities = y_pred_proba[idx]
        
        row = {
            'Номер_образца': f'Сирень_{i+1:02d}',
            'Истинный_класс': true_label,
            'Предсказанный_класс': pred_label,
            'Уверенность': f'{confidence:.4f}',
            'Правильно': 'Да' if true_label == pred_label else 'Нет'
        }
        
        # Добавляем вероятности для всех классов
        for j, class_name in enumerate(class_names):
            row[f'Вероятность_{class_name}'] = f'{probabilities[j]:.4f}'
        
        results_data.append(row)
    
    # Создаем DataFrame
    df_results = pd.DataFrame(results_data)
    
    # Добавляем строку со средними вероятностями
    avg_probabilities = {}
    for class_name in class_names:
        prob_col = f'Вероятность_{class_name}'
        avg_prob = df_results[prob_col].astype(float).mean()
        avg_probabilities[prob_col] = f'{avg_prob:.4f}'
    
    avg_row = {
        'Номер_образца': 'СРЕДНЯЯ_ВЕРОЯТНОСТЬ',
        'Истинный_класс': '-',
        'Предсказанный_класс': '-',
        'Уверенность': '-',
        'Правильно': '-'
    }
    avg_row.update(avg_probabilities)
    
    # Добавляем среднюю строку
    df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Сохраняем в Excel
    filename = f'extratrees_20_species_10percent_noise_osina_siren_results.xlsx'
    df_results.to_excel(filename, index=False, sheet_name='Результаты_Осина_Сирень')
    
    print(f"Excel файл сохранен: {filename}")
    print(f"Всего строк: {len(df_results)} (60 образцов + 1 средняя)")
    
    return df_results

def main():
    """Основная функция для ExtraTrees с 10% шумом"""
    print("EXTRA TREES - 20 ВИДОВ ДЕРЕВЬЕВ С 10% ШУМОМ")
    print("=" * 60)
    print("Создание Excel файла для осины и сирени")
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
    
    # Создание и обучение модели ExtraTrees
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
    
    print("\nОбучение модели ExtraTrees...")
    model.fit(X_train_scaled, y_train)
    
    # Тестирование на данных с 10% шумом
    print("\nТестирование с 10% шумом...")
    X_test_noisy = add_gaussian_noise(X_test_scaled, 10)
    
    y_pred = model.predict(X_test_noisy)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Точность при 10% шуме: {accuracy:.4f}")
    
    # Создание Excel файла для осины и сирени
    print("\nСоздание Excel файла для осины и сирени...")
    results_df = create_excel_output_for_osina_siren(
        model, X_test_scaled, y_test, class_names, scaler, noise_level=10
    )
    
    # Создание матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Матрица ошибок ExtraTrees - 10% шум (точность: {accuracy:.4f})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('extratrees_20_species_10percent_noise_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nМатрица ошибок сохранена: extratrees_20_species_10percent_noise_confusion_matrix.png")
    
    # Сохранение параметров модели
    with open('extratrees_20_species_10percent_noise_parameters.txt', 'w', encoding='utf-8') as f:
        f.write("ПАРАМЕТРЫ МОДЕЛИ EXTRA TREES - 20 ВИДОВ С 10% ШУМОМ\n")
        f.write("=" * 60 + "\n\n")
        f.write("Модель: ExtraTreesClassifier\n")
        f.write("n_estimators: 1000\n")
        f.write("max_depth: None\n")
        f.write("min_samples_split: 2\n")
        f.write("min_samples_leaf: 1\n")
        f.write("max_features: 'sqrt'\n")
        f.write("bootstrap: False\n")
        f.write("random_state: 42\n")
        f.write("n_jobs: -1\n\n")
        
        f.write("ДАННЫЕ:\n")
        f.write("-" * 30 + "\n")
        f.write("Количество видов: 20\n")
        f.write("Файлов на вид: 30\n")
        f.write("Разделение данных: 80% обучение, 20% тест\n")
        f.write("Уровень шума при тестировании: 10%\n")
        f.write("Точность при 10% шуме: {:.4f}\n".format(accuracy))
    
    print("Параметры сохранены: extratrees_20_species_10percent_noise_parameters.txt")
    
    print(f"\nВСЕ ФАЙЛЫ СОЗДАНЫ:")
    print(f"- extratrees_20_species_10percent_noise_osina_siren_results.xlsx")
    print(f"- extratrees_20_species_10percent_noise_confusion_matrix.png")
    print(f"- extratrees_20_species_10percent_noise_parameters.txt")

if __name__ == "__main__":
    main() 