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

def load_spectral_data_osina_siren():
    """Загружает данные только для осины и сирени"""
    data = []
    labels = []
    
    # Загружаем данные осины
    osina_files = glob.glob('осина/*.xlsx')
    for file in osina_files[:30]:  # Берем первые 30 файлов
        try:
            df = pd.read_excel(file)
            # Предполагаем, что спектральные данные начинаются со второй колонки
            spectral_data = df.iloc[:, 1:].values.flatten()
            if len(spectral_data) > 0:
                data.append(spectral_data)
                labels.append('осина')
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")
    
    # Загружаем данные сирени
    siren_files = glob.glob('Спектры, весенний период, 20 видов/сирень/*.xlsx')
    for file in siren_files[:30]:  # Берем первые 30 файлов
        try:
            df = pd.read_excel(file)
            # Предполагаем, что спектральные данные начинаются со второй колонки
            spectral_data = df.iloc[:, 1:].values.flatten()
            if len(spectral_data) > 0:
                data.append(spectral_data)
                labels.append('сирень')
        except Exception as e:
            print(f"Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def train_extra_trees_model(X_train, y_train):
    """Обучает модель Extra Trees"""
    model = ExtraTreesClassifier(
        n_estimators=1712,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    return model

def add_noise(X, noise_level):
    """Добавляет гауссов шум к данным"""
    if noise_level == 0:
        return X
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def create_excel_output(model, X_test, y_test, scaler, label_encoder, noise_levels):
    """Создает Excel файл с результатами в формате et80_10.xlsx"""
    
    # Создаем DataFrame для результатов
    results_data = []
    
    for noise_level in noise_levels:
        # Добавляем шум к тестовым данным
        X_test_noisy = add_noise(X_test, noise_level)
        
        # Получаем вероятности для каждого класса
        probabilities = model.predict_proba(X_test_noisy)
        
        # Находим индекс класса с наибольшей вероятностью
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Создаем строки для каждого спектра
        for i, (true_label, pred_class) in enumerate(zip(y_test, predicted_classes)):
            # Получаем названия классов
            true_class_name = label_encoder.inverse_transform([true_label])[0]
            pred_class_name = label_encoder.inverse_transform([pred_class])[0]
            
            # Создаем строку с единицей для наибольшей вероятности
            row = [f"Спектр_{i+1}_{true_class_name}"]
            
            # Добавляем нули для всех классов
            for j in range(len(label_encoder.classes_)):
                if j == pred_class:
                    row.append(1)  # Единица для наибольшей вероятности
                else:
                    row.append(0)
            
            results_data.append(row)
    
    # Создаем заголовки
    headers = ['Спектр'] + list(label_encoder.classes_)
    
    # Создаем DataFrame
    df = pd.DataFrame(results_data, columns=headers)
    
    # Сохраняем в Excel
    filename = f'extra_trees_osina_siren_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    df.to_excel(filename, index=False)
    
    print(f"Результаты сохранены в файл: {filename}")
    return filename

def analyze_osina_siren_classification():
    """Основная функция анализа"""
    print("Загрузка данных для осины и сирени...")
    X, y = load_spectral_data_osina_siren()
    
    print(f"Загружено {len(X)} спектров")
    print(f"Распределение классов: {np.bincount(y == 'осина')}")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные (80% на обучение)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучаем модель
    print("Обучение модели Extra Trees...")
    model = train_extra_trees_model(X_train_scaled, y_train)
    
    # Тестируем на разных уровнях шума
    noise_levels = [0, 0.01, 0.05, 0.10]  # 0%, 1%, 5%, 10%
    
    print("\nРезультаты классификации:")
    print("-" * 50)
    
    for noise_level in noise_levels:
        X_test_noisy = add_noise(X_test_scaled, noise_level)
        y_pred = model.predict(X_test_noisy)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Шум {noise_level*100:.0f}%: Точность = {accuracy:.4f}")
        
        # Выводим детальный отчет
        print(f"\nОтчет для шума {noise_level*100:.0f}%:")
        print(classification_report(y_test, y_pred, 
                                  target_names=label_encoder.classes_))
        
        # Создаем матрицу ошибок
        cm = confusion_matrix(y_test, y_pred)
        print(f"Матрица ошибок для шума {noise_level*100:.0f}%:")
        print(cm)
        print()
    
    # Создаем Excel файл с результатами
    print("Создание Excel файла с результатами...")
    excel_file = create_excel_output(model, X_test_scaled, y_test, scaler, 
                                   label_encoder, noise_levels)
    
    # Сохраняем модель и препроцессоры
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    import joblib
    joblib.dump(model, f'extra_trees_osina_siren_model_{timestamp}.pkl')
    joblib.dump(scaler, f'extra_trees_osina_siren_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'extra_trees_osina_siren_label_encoder_{timestamp}.pkl')
    
    print(f"\nМодель и препроцессоры сохранены с временной меткой: {timestamp}")
    
    return model, scaler, label_encoder, excel_file

if __name__ == "__main__":
    model, scaler, label_encoder, excel_file = analyze_osina_siren_classification()
    print(f"\nАнализ завершен! Результаты сохранены в: {excel_file}") 