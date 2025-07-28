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
    
    species_folders = [
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 'клен_ам', 
        'липа', 'лиственница', 'орех', 'осина', 'рябина', 'сирень', 'сосна', 
        'тополь_бальзамический', 'тополь_черный', 'туя', 'черемуха', 'ясень'
    ]
    
    for species in species_folders:
        print(f"Загрузка данных для {species}...")
        folder_path = f'Спектры, весенний период, 20 видов/{species}'
        
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена, пропускаем...")
            continue
            
        files = glob.glob(f'{folder_path}/*.xlsx')
        print(f"  Найдено {len(files)} файлов")
        
        # Загружаем ВСЕ файлы для каждого вида
        for file in files:
            try:
                df = pd.read_excel(file)
                spectral_data = df.iloc[:, 1:].values.flatten()
                
                if len(spectral_data) > 0 and not np.any(np.isnan(spectral_data)):
                    # Нормализация к [0,1]
                    spectral_data = (spectral_data - np.min(spectral_data)) / (np.max(spectral_data) - np.min(spectral_data))
                    data.append(spectral_data)
                    labels.append(species)
            except Exception as e:
                print(f"  Ошибка при чтении файла {file}: {e}")
                continue
    
    return np.array(data), np.array(labels)

def add_noise_to_data(X, noise_level):
    """Добавляет шум к данным"""
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    # Ограничиваем значения в диапазоне [0, 1]
    noisy_X = np.clip(noisy_X, 0, 1)
    return noisy_X

def create_excel_output_for_osina_siren(model, X_test, y_test, label_encoder, scaler, filename):
    """Создает Excel файл с результатами для осины и сирени"""
    print(f"\nСоздание Excel вывода для осины и сирени...")
    
    # Получаем индексы осины и сирени
    osina_idx = np.where(label_encoder.classes_ == 'осина')[0][0]
    siren_idx = np.where(label_encoder.classes_ == 'сирень')[0][0]
    
    # Фильтруем тестовые данные только для осины и сирени
    osina_mask = (y_test == osina_idx)
    siren_mask = (y_test == siren_idx)
    
    # Объединяем маски
    target_mask = osina_mask | siren_mask
    
    if not np.any(target_mask):
        print("Ошибка: Не найдены данные для осины и сирени в тестовой выборке")
        return
    
    X_target = X_test[target_mask]
    y_target = y_test[target_mask]
    
    # Масштабируем данные
    X_target_scaled = scaler.transform(X_target)
    
    # Получаем предсказания
    y_pred_proba = model.predict_proba(X_target_scaled)
    y_pred = model.predict(X_target_scaled)
    
    # Создаем DataFrame для Excel
    results_data = []
    
    for i in range(len(X_target)):
        # Определяем истинный класс
        true_class = label_encoder.classes_[y_target[i]]
        
        # Создаем строку для каждого спектра
        row = {
            'Спектр': f'{true_class}_{i+1:03d}',
            'Истинный_класс': true_class
        }
        
        # Добавляем вероятности для всех классов
        for j, class_name in enumerate(label_encoder.classes_):
            row[f'Вероятность_{class_name}'] = y_pred_proba[i][j]
        
        # Добавляем предсказанный класс
        predicted_class = label_encoder.classes_[y_pred[i]]
        row['Предсказанный_класс'] = predicted_class
        
        # Добавляем 1 для наибольшей вероятности, 0 для остальных
        max_prob_idx = np.argmax(y_pred_proba[i])
        for j, class_name in enumerate(label_encoder.classes_):
            if j == max_prob_idx:
                row[f'Результат_{class_name}'] = 1
            else:
                row[f'Результат_{class_name}'] = 0
        
        results_data.append(row)
    
    # Создаем DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Сохраняем в Excel
    results_df.to_excel(filename, index=False)
    print(f"Excel файл сохранен: {filename}")
    print(f"Количество строк: {len(results_df)}")
    
    # Показываем статистику
    print(f"\nСтатистика для осины и сирени:")
    print(f"Всего спектров: {len(results_df)}")
    print(f"Осина: {np.sum(results_df['Истинный_класс'] == 'осина')}")
    print(f"Сирень: {np.sum(results_df['Истинный_класс'] == 'сирень')}")
    
    # Точность классификации
    accuracy = accuracy_score(y_target, y_pred)
    print(f"Точность классификации: {accuracy:.4f}")
    
    return results_df

def test_with_noise(model, X_test, y_test, scaler, label_encoder, noise_levels=[0.0, 0.01, 0.05, 0.10]):
    """Тестирует модель на данных с разными уровнями шума"""
    print(f"\nТЕСТИРОВАНИЕ EXTRATREES С ШУМОМ:")
    print("=" * 50)
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\nТестирование с шумом {noise_level*100}%:")
        
        # Добавляем шум к тестовым данным
        if noise_level > 0:
            X_test_noisy = add_noise_to_data(X_test, noise_level)
        else:
            X_test_noisy = X_test.copy()
        
        # Масштабируем данные
        X_test_scaled = scaler.transform(X_test_noisy)
        
        # Предсказания
        y_pred_proba = model.predict_proba(X_test_scaled)
        y_pred = model.predict(X_test_scaled)
        
        # Точность
        accuracy = accuracy_score(y_test, y_pred)
        
        # Средняя уверенность
        confidence = np.mean(np.max(y_pred_proba, axis=1))
        confidence_std = np.std(np.max(y_pred_proba, axis=1))
        
        print(f"  Точность: {accuracy:.4f}")
        print(f"  Средняя уверенность: {confidence:.4f} ± {confidence_std:.4f}")
        
        # Анализ по классам (только осина и сирень)
        osina_idx = np.where(label_encoder.classes_ == 'осина')[0][0]
        siren_idx = np.where(label_encoder.classes_ == 'сирень')[0][0]
        
        print("  Точность по классам (осина и сирень):")
        for idx, class_name in [(osina_idx, 'осина'), (siren_idx, 'сирень')]:
            class_mask = (y_test == idx)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_confidence = np.mean(np.max(y_pred_proba[class_mask], axis=1))
                print(f"    {class_name}: {class_accuracy:.3f} (уверенность: {class_confidence:.3f})")
        
        results.append({
            'noise_level': noise_level,
            'accuracy': accuracy,
            'confidence': confidence,
            'confidence_std': confidence_std
        })
    
    return results

def plot_confusion_matrix(y_true, y_pred, label_encoder, title, filename):
    """Создает матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax1)
    ax1.set_title(f'{title} (абсолютные значения)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Предсказанный класс', fontsize=12)
    ax1.set_ylabel('Истинный класс', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax2)
    ax2.set_title(f'{title} (нормализованные значения)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Предсказанный класс', fontsize=12)
    ax2.set_ylabel('Истинный класс', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_noise_results(results, filename):
    """Создает график результатов тестирования с шумом"""
    noise_levels = [r['noise_level'] * 100 for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    confidences = [r['confidence'] * 100 for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График точности
    ax1.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Уровень шума (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Точность (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Точность ExtraTrees при разных уровнях шума', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Добавляем значения на точки
    for i, (x, y) in enumerate(zip(noise_levels, accuracies)):
        ax1.annotate(f'{y:.1f}%', (x, y + 2), ha='center', va='bottom', fontweight='bold')
    
    # График уверенности
    ax2.plot(noise_levels, confidences, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Уровень шума (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Средняя уверенность (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Уверенность ExtraTrees при разных уровнях шума', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Добавляем значения на точки
    for i, (x, y) in enumerate(zip(noise_levels, confidences)):
        ax2.annotate(f'{y:.1f}%', (x, y + 2), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Основная функция"""
    print("EXTRATREES - 20 ВИДОВ С EXCEL ВЫВОДОМ ДЛЯ ОСИНЫ И СИРЕНИ")
    print("=" * 60)
    
    # Загружаем данные
    X, y = load_spectral_data_20_species()
    
    print(f"\nЗагружено {len(X)} спектров")
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {len(X_train)} (80%)")
    print(f"  Тестовая выборка: {len(X_test)} (20%)")
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создаем и обучаем ExtraTrees
    print("\nОбучение ExtraTrees...")
    et_model = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    et_model.fit(X_train_scaled, y_train)
    
    # Оцениваем модель на чистых данных
    print("\nРЕЗУЛЬТАТЫ НА ЧИСТЫХ ДАННЫХ:")
    print("-" * 50)
    
    y_pred = et_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность на тестовых данных: {accuracy:.4f}")
    
    # Отчет о классификации
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Создаем Excel вывод для осины и сирени
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f'extratrees_osina_siren_results_{timestamp}.xlsx'
    
    excel_results = create_excel_output_for_osina_siren(
        et_model, X_test, y_test, label_encoder, scaler, excel_filename
    )
    
    # Тестируем с шумом
    noise_results = test_with_noise(et_model, X_test, y_test, scaler, label_encoder)
    
    # Создаем графики
    # Матрица ошибок для чистых данных
    plot_confusion_matrix(y_test, y_pred, label_encoder, 
                         f'ExtraTrees 20 видов (точность: {accuracy:.1%})',
                         f'extratrees_20_species_confusion_matrix_{timestamp}.png')
    
    # Результаты с шумом
    plot_noise_results(noise_results, f'extratrees_20_species_noise_results_{timestamp}.png')
    
    # Сохраняем модель
    import joblib
    joblib.dump(et_model, f'extratrees_20_species_model_{timestamp}.pkl')
    joblib.dump(scaler, f'extratrees_20_species_scaler_{timestamp}.pkl')
    joblib.dump(label_encoder, f'extratrees_20_species_label_encoder_{timestamp}.pkl')
    
    # Сохраняем результаты в CSV
    results_df = pd.DataFrame(noise_results)
    results_df.to_csv(f'extratrees_20_species_noise_results_{timestamp}.csv', index=False)
    
    print(f"\nМодель и результаты сохранены с временной меткой: {timestamp}")
    
    # Открываем результаты
    import subprocess
    subprocess.run(['open', excel_filename])
    subprocess.run(['open', f'extratrees_20_species_confusion_matrix_{timestamp}.png'])
    subprocess.run(['open', f'extratrees_20_species_noise_results_{timestamp}.png'])
    
    return accuracy, et_model, scaler, label_encoder

if __name__ == "__main__":
    main() 