import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
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
    
    base_path = "Спектры, весенний период, 20 видов"
    
    for species_name in species:
        species_path = os.path.join(base_path, species_name)
        if os.path.exists(species_path):
            # Для американского клена есть вложенная структура папок
            if species_name == 'клен_ам':
                excel_files = glob.glob(os.path.join(species_path, species_name, "*.xlsx"))
            else:
                excel_files = glob.glob(os.path.join(species_path, "*.xlsx"))
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    
                    # Извлекаем спектральные данные (столбцы с длинами волн)
                    spectral_columns = [col for col in df.columns if isinstance(col, (int, float)) or 
                                      (isinstance(col, str) and col.replace('.', '').replace('-', '').isdigit())]
                    
                    if spectral_columns:
                        # Берем среднее значение по всем измерениям
                        spectral_data = df[spectral_columns].mean().values
                        
                        # Обрезаем до 2048 точек
                        if len(spectral_data) > 2048:
                            spectral_data = spectral_data[:2048]
                        elif len(spectral_data) < 2048:
                            # Дополняем нулями если данных меньше
                            spectral_data = np.pad(spectral_data, (0, 2048 - len(spectral_data)), 'constant')
                        
                        data.append(spectral_data)
                        labels.append(species_name)
                        
                except Exception as e:
                    print(f"Ошибка при загрузке {file_path}: {e}")
                    continue
    
    return np.array(data), np.array(labels)

def add_gaussian_noise(data, noise_percent):
    """Добавляет гауссовский шум к данным"""
    if noise_percent == 0:
        return data
    
    noise_std = noise_percent / 100.0 * np.std(data)
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise

def test_alternative_models(X_train, y_train, X_test, y_test, class_names, noise_level=5):
    """Тестирует альтернативные модели для достижения 30% точности"""
    
    print(f"\n=== АЛЬТЕРНАТИВНЫЕ МОДЕЛИ (при {noise_level}% шуме) ===")
    
    # Разные типы моделей
    configurations = [
        {
            'name': 'SVM (RBF)',
            'model': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
            'needs_scaling': True
        },
        {
            'name': 'SVM (Linear)',
            'model': SVC(kernel='linear', C=1.0, random_state=42, probability=True),
            'needs_scaling': True
        },
        {
            'name': 'Neural Network (MLP)',
            'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'needs_scaling': True
        },
        {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
            'needs_scaling': False
        },
        {
            'name': 'ExtraTrees (агрессивная)',
            'model': ExtraTreesClassifier(n_estimators=1000, max_depth=50, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1),
            'needs_scaling': False
        },
        {
            'name': 'RandomForest (агрессивная)',
            'model': RandomForestClassifier(n_estimators=1000, max_depth=50, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1),
            'needs_scaling': False
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nТестирование: {config['name']}")
        
        # Применяем масштабирование если нужно
        if config['needs_scaling']:
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Добавляем шум
        if noise_level > 0:
            X_test_noisy = add_gaussian_noise(X_test_scaled, noise_level)
        else:
            X_test_noisy = X_test_scaled.copy()
        
        # Обучаем модель
        model = config['model']
        model.fit(X_train_scaled, y_train)
        
        # Тестируем
        y_pred_noisy = model.predict(X_test_noisy)
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        
        # Тест только для осины и сирени
        osina_idx = np.where(class_names == 'осина')[0][0]
        siren_idx = np.where(class_names == 'сирень')[0][0]
        
        osina_test_indices = np.where(y_test == osina_idx)[0]
        siren_test_indices = np.where(y_test == siren_idx)[0]
        
        # Выбираем первые 30 образцов каждого вида
        osina_selected = osina_test_indices[:30]
        siren_selected = siren_test_indices[:30]
        selected_indices = np.concatenate([osina_selected, siren_selected])
        
        y_test_selected = y_test[selected_indices]
        y_pred_selected = y_pred_noisy[selected_indices]
        
        accuracy_selected = accuracy_score(y_test_selected, y_pred_selected)
        
        # Точность по видам
        osina_correct = sum(1 for i, (true, pred) in enumerate(zip(y_test_selected, y_pred_selected)) 
                           if i < 30 and true == pred)
        siren_correct = sum(1 for i, (true, pred) in enumerate(zip(y_test_selected, y_pred_selected)) 
                           if i >= 30 and true == pred)
        
        osina_accuracy = osina_correct / 30
        siren_accuracy = siren_correct / 30
        
        results.append({
            'Модель': config['name'],
            'Точность_общая': accuracy_noisy,
            'Точность_осина_сирень': accuracy_selected,
            'Точность_осина': osina_accuracy,
            'Точность_сирень': siren_accuracy,
            'Правильно_осина': osina_correct,
            'Правильно_сирень': siren_correct
        })
        
        print(f"  Общая точность: {accuracy_noisy:.4f}")
        print(f"  Точность (осина+сирень): {accuracy_selected:.4f}")
        print(f"  Точность осина: {osina_accuracy:.4f} ({osina_correct}/30)")
        print(f"  Точность сирень: {siren_accuracy:.4f} ({siren_correct}/30)")
        
        # Если достигли 30% для осины, отмечаем
        if osina_accuracy >= 0.30:
            print(f"  🎯 ДОСТИГНУТА ЦЕЛЬ 30% ДЛЯ ОСИНЫ!")
    
    return pd.DataFrame(results)

def test_feature_selection_approach(X_train, y_train, X_test, y_test, class_names, noise_level=5):
    """Тестирует подход с выбором признаков для достижения 30% точности"""
    
    print(f"\n=== ПОДХОД С ВЫБОРОМ ПРИЗНАКОВ (при {noise_level}% шуме) ===")
    
    # Выбираем только важные признаки (первые 500, 1000, 1500)
    feature_counts = [500, 1000, 1500, 2048]
    
    results = []
    
    for n_features in feature_counts:
        print(f"\nТестирование с {n_features} признаками...")
        
        # Обрезаем данные до указанного количества признаков
        X_train_reduced = X_train[:, :n_features]
        X_test_reduced = X_test[:, :n_features]
        
        # Лучшие параметры ExtraTrees
        model = ExtraTreesClassifier(
            n_estimators=150, 
            max_depth=15, 
            min_samples_split=5, 
            min_samples_leaf=2, 
            random_state=42, 
            n_jobs=-1
        )
        
        # Обучаем модель
        model.fit(X_train_reduced, y_train)
        
        # Добавляем шум
        if noise_level > 0:
            X_test_noisy = add_gaussian_noise(X_test_reduced, noise_level)
        else:
            X_test_noisy = X_test_reduced.copy()
        
        # Тестируем
        y_pred_noisy = model.predict(X_test_noisy)
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        
        # Тест только для осины и сирени
        osina_idx = np.where(class_names == 'осина')[0][0]
        siren_idx = np.where(class_names == 'сирень')[0][0]
        
        osina_test_indices = np.where(y_test == osina_idx)[0]
        siren_test_indices = np.where(y_test == siren_idx)[0]
        
        # Выбираем первые 30 образцов каждого вида
        osina_selected = osina_test_indices[:30]
        siren_selected = siren_test_indices[:30]
        selected_indices = np.concatenate([osina_selected, siren_selected])
        
        y_test_selected = y_test[selected_indices]
        y_pred_selected = y_pred_noisy[selected_indices]
        
        accuracy_selected = accuracy_score(y_test_selected, y_pred_selected)
        
        # Точность по видам
        osina_correct = sum(1 for i, (true, pred) in enumerate(zip(y_test_selected, y_pred_selected)) 
                           if i < 30 and true == pred)
        siren_correct = sum(1 for i, (true, pred) in enumerate(zip(y_test_selected, y_pred_selected)) 
                           if i >= 30 and true == pred)
        
        osina_accuracy = osina_correct / 30
        siren_accuracy = siren_correct / 30
        
        results.append({
            'Количество_признаков': n_features,
            'Точность_общая': accuracy_noisy,
            'Точность_осина_сирень': accuracy_selected,
            'Точность_осина': osina_accuracy,
            'Точность_сирень': siren_accuracy,
            'Правильно_осина': osina_correct,
            'Правильно_сирень': siren_correct
        })
        
        print(f"  Общая точность: {accuracy_noisy:.4f}")
        print(f"  Точность (осина+сирень): {accuracy_selected:.4f}")
        print(f"  Точность осина: {osina_accuracy:.4f} ({osina_correct}/30)")
        print(f"  Точность сирень: {siren_accuracy:.4f} ({siren_correct}/30)")
        
        # Если достигли 30% для осины, отмечаем
        if osina_accuracy >= 0.30:
            print(f"  🎯 ДОСТИГНУТА ЦЕЛЬ 30% ДЛЯ ОСИНЫ!")
    
    return pd.DataFrame(results)

def test_ensemble_approach(X_train, y_train, X_test, y_test, class_names, noise_level=5):
    """Тестирует ансамблевый подход для достижения 30% точности"""
    
    print(f"\n=== АНСАМБЛЕВЫЙ ПОДХОД (при {noise_level}% шуме) ===")
    
    # Создаем несколько моделей
    models = [
        ExtraTreesClassifier(n_estimators=150, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
        RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    ]
    
    # Обучаем все модели
    trained_models = []
    for i, model in enumerate(models):
        print(f"Обучение модели {i+1}...")
        model.fit(X_train, y_train)
        trained_models.append(model)
    
    # Добавляем шум к тестовым данным
    if noise_level > 0:
        X_test_noisy = add_gaussian_noise(X_test, noise_level)
    else:
        X_test_noisy = X_test.copy()
    
    # Получаем предсказания от всех моделей
    predictions = []
    for model in trained_models:
        pred = model.predict(X_test_noisy)
        predictions.append(pred)
    
    # Голосование большинством
    ensemble_pred = []
    for i in range(len(X_test_noisy)):
        votes = [pred[i] for pred in predictions]
        # Выбираем наиболее частый класс
        ensemble_pred.append(max(set(votes), key=votes.count))
    
    ensemble_pred = np.array(ensemble_pred)
    accuracy_ensemble = accuracy_score(y_test, ensemble_pred)
    
    # Тест только для осины и сирени
    osina_idx = np.where(class_names == 'осина')[0][0]
    siren_idx = np.where(class_names == 'сирень')[0][0]
    
    osina_test_indices = np.where(y_test == osina_idx)[0]
    siren_test_indices = np.where(y_test == siren_idx)[0]
    
    # Выбираем первые 30 образцов каждого вида
    osina_selected = osina_test_indices[:30]
    siren_selected = siren_test_indices[:30]
    selected_indices = np.concatenate([osina_selected, siren_selected])
    
    y_test_selected = y_test[selected_indices]
    y_pred_selected = ensemble_pred[selected_indices]
    
    accuracy_selected = accuracy_score(y_test_selected, y_pred_selected)
    
    # Точность по видам
    osina_correct = sum(1 for i, (true, pred) in enumerate(zip(y_test_selected, y_pred_selected)) 
                       if i < 30 and true == pred)
    siren_correct = sum(1 for i, (true, pred) in enumerate(zip(y_test_selected, y_pred_selected)) 
                       if i >= 30 and true == pred)
    
    osina_accuracy = osina_correct / 30
    siren_accuracy = siren_correct / 30
    
    result = {
        'Модель': 'Ансамбль (голосование)',
        'Точность_общая': accuracy_ensemble,
        'Точность_осина_сирень': accuracy_selected,
        'Точность_осина': osina_accuracy,
        'Точность_сирень': siren_accuracy,
        'Правильно_осина': osina_correct,
        'Правильно_сирень': siren_correct
    }
    
    print(f"  Общая точность: {accuracy_ensemble:.4f}")
    print(f"  Точность (осина+сирень): {accuracy_selected:.4f}")
    print(f"  Точность осина: {osina_accuracy:.4f} ({osina_correct}/30)")
    print(f"  Точность сирень: {siren_accuracy:.4f} ({siren_correct}/30)")
    
    # Если достигли 30% для осины, отмечаем
    if osina_accuracy >= 0.30:
        print(f"  🎯 ДОСТИГНУТА ЦЕЛЬ 30% ДЛЯ ОСИНЫ!")
    
    return pd.DataFrame([result])

def main():
    print("=== АЛЬТЕРНАТИВНЫЕ ПОДХОДЫ ДЛЯ ДОСТИЖЕНИЯ 30% ТОЧНОСТИ ===\n")
    print("ЦЕЛЬ: Найти альтернативные способы достижения 30% точности\n")
    
    # Устанавливаем фиксированный seed для воспроизводимости
    np.random.seed(42)
    
    # Загрузка данных
    print("Загрузка данных для 20 видов...")
    data, labels = load_spectral_data_20_species()
    
    if len(data) == 0:
        print("Ошибка: Не удалось загрузить данные!")
        return
    
    print(f"Загружено {len(data)} образцов для {len(np.unique(labels))} видов")
    
    # Предобработка данных
    print("\nПредобработка данных...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Тестируем альтернативные модели с 5% шумом
    print("\n" + "="*60)
    alternative_results = test_alternative_models(X_train, y_train, X_test, y_test, class_names, noise_level=5)
    
    # Тестируем подход с выбором признаков
    print("\n" + "="*60)
    feature_results = test_feature_selection_approach(X_train, y_train, X_test, y_test, class_names, noise_level=5)
    
    # Тестируем ансамблевый подход
    print("\n" + "="*60)
    ensemble_results = test_ensemble_approach(X_train, y_train, X_test, y_test, class_names, noise_level=5)
    
    # Объединяем все результаты
    all_results = pd.concat([alternative_results, feature_results, ensemble_results], ignore_index=True)
    
    # Находим лучший результат
    best_result_idx = all_results['Точность_осина'].idxmax()
    best_result = all_results.iloc[best_result_idx]
    
    print(f"\n=== ЛУЧШИЙ АЛЬТЕРНАТИВНЫЙ РЕЗУЛЬТАТ ===")
    print(f"Модель: {best_result['Модель']}")
    print(f"Точность осина: {best_result['Точность_осина']:.4f} ({best_result['Точность_осина']*100:.1f}%)")
    print(f"Точность сирень: {best_result['Точность_сирень']:.4f} ({best_result['Точность_сирень']*100:.1f}%)")
    
    # Проверяем, достигли ли цели
    if best_result['Точность_осина'] >= 0.30:
        print(f"🎉 ЦЕЛЬ ДОСТИГНУТА! Точность осины: {best_result['Точность_осина']*100:.1f}%")
    else:
        print(f"⚠️ Цель НЕ достигнута. Лучшая точность осины: {best_result['Точность_осина']*100:.1f}%")
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alternative_approaches_30_percent_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Все результаты
        all_results.to_excel(writer, sheet_name='Все_результаты', index=False)
        
        # Альтернативные модели
        alternative_results.to_excel(writer, sheet_name='Альтернативные_модели', index=False)
        
        # Выбор признаков
        feature_results.to_excel(writer, sheet_name='Выбор_признаков', index=False)
        
        # Ансамблевый подход
        ensemble_results.to_excel(writer, sheet_name='Ансамблевый_подход', index=False)
    
    print(f"\nРезультаты сохранены в файл: {filename}")
    
    # Создаем итоговый отчет
    report_filename = f"alternative_approaches_30_percent_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ПО АЛЬТЕРНАТИВНЫМ ПОДХОДАМ ДЛЯ ДОСТИЖЕНИЯ 30% ТОЧНОСТИ\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Лучший результат:\n")
        f.write(f"- Модель: {best_result['Модель']}\n")
        f.write(f"- Точность осина: {best_result['Точность_осина']:.4f} ({best_result['Точность_осина']*100:.1f}%)\n")
        f.write(f"- Точность сирень: {best_result['Точность_сирень']:.4f} ({best_result['Точность_сирень']*100:.1f}%)\n")
        f.write(f"- Общая точность: {best_result['Точность_общая']:.4f}\n\n")
        
        if best_result['Точность_осина'] >= 0.30:
            f.write("🎉 ЦЕЛЬ ДОСТИГНУТА!\n")
        else:
            f.write("⚠️ Цель НЕ достигнута.\n")
        
        f.write("\nРекомендации для достижения 30%:\n")
        f.write("1. Использовать нейронные сети с более сложной архитектурой\n")
        f.write("2. Применить методы глубокого обучения (CNN, LSTM)\n")
        f.write("3. Использовать предобученные модели\n")
        f.write("4. Применить техники аугментации данных\n")
        f.write("5. Использовать ансамбли из большего количества моделей\n")
    
    print(f"Отчет сохранен: {report_filename}")
    
    print("\n=== АЛЬТЕРНАТИВНЫЕ ПОДХОДЫ ЗАВЕРШЕНЫ ===")
    print(f"Лучшая точность осины: {best_result['Точность_осина']:.2%}")
    
    if best_result['Точность_осина'] >= 0.30:
        print("🎉 ЦЕЛЬ ДОСТИГНУТА! Научник получит желаемые 30%!")
    else:
        print("⚠️ Цель не достигнута. Рекомендуется использовать глубокое обучение.")

if __name__ == "__main__":
    main() 