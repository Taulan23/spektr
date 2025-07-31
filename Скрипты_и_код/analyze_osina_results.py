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
                spectral_data = df.iloc[:, 1:].values.flatten()
                if len(spectral_data) > 0:
                    data.append(spectral_data)
                    labels.append(species_name)
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def analyze_osina_performance():
    """Анализирует производительность модели для осины"""
    print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ДЛЯ ОСИНЫ")
    print("=" * 50)
    
    # Загрузка данных
    X, y = load_spectral_data_20_species()
    
    if len(X) == 0:
        print("Не удалось загрузить данные!")
        return
    
    # Предобработка
    lengths = [len(s) for s in X]
    target_length = min(lengths)
    X_processed = np.array([spectrum[:target_length] for spectrum in X], dtype=np.float32)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    # Находим индекс осины
    osina_index = np.where(class_names == 'осина')[0][0]
    print(f"Индекс осины в классах: {osina_index}")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создание и обучение модели
    model = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Предсказание
    y_pred = model.predict(X_test_scaled)
    
    # Анализ для осины
    print(f"\nАНАЛИЗ ДЛЯ ОСИНЫ:")
    print("-" * 30)
    
    # Находим тестовые образцы осины
    osina_test_indices = np.where(y_test == osina_index)[0]
    osina_predictions = y_pred[osina_test_indices]
    osina_true = y_test[osina_test_indices]
    
    print(f"Количество тестовых образцов осины: {len(osina_test_indices)}")
    print(f"Правильно классифицировано: {np.sum(osina_predictions == osina_true)}")
    print(f"Точность для осины: {np.sum(osina_predictions == osina_true) / len(osina_test_indices):.4f}")
    
    # Анализ ошибок
    osina_errors = osina_predictions != osina_true
    if np.any(osina_errors):
        print(f"\nОШИБКИ КЛАССИФИКАЦИИ ОСИНЫ:")
        print("-" * 30)
        wrong_predictions = osina_predictions[osina_errors]
        unique_wrong, counts = np.unique(wrong_predictions, return_counts=True)
        
        for wrong_class, count in zip(unique_wrong, counts):
            wrong_class_name = class_names[wrong_class]
            print(f"Осину ошибочно классифицировали как {wrong_class_name}: {count} раз")
    else:
        print("✅ Осина классифицируется без ошибок!")
    
    # Полная матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    
    # Анализ строки осины в матрице ошибок
    osina_row = cm[osina_index, :]
    print(f"\nСТРОКА ОСИНЫ В МАТРИЦЕ ОШИБОК:")
    print("-" * 30)
    print(f"Всего образцов осины: {np.sum(osina_row)}")
    print(f"Правильно классифицировано: {osina_row[osina_index]}")
    print(f"Ошибки классификации: {np.sum(osina_row) - osina_row[osina_index]}")
    
    # Показываем, куда ошибочно классифицировали
    for i, count in enumerate(osina_row):
        if i != osina_index and count > 0:
            print(f"  → {class_names[i]}: {count}")
    
    # Общая точность
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nОБЩАЯ ТОЧНОСТЬ МОДЕЛИ: {overall_accuracy:.4f}")
    
    return cm, class_names, osina_index

if __name__ == "__main__":
    analyze_osina_performance() 