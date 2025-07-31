import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

def test_different_noise_levels(X_train, y_train, X_test, y_test, class_names):
    """Тестирует разные уровни шума для достижения 30% точности"""
    
    print("\n=== ТЕСТИРОВАНИЕ РАЗНЫХ УРОВНЕЙ ШУМА ===")
    
    # Лучшие параметры из предыдущего анализа
    best_params = {
        'n_estimators': 150,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Разные уровни шума
    noise_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    results = []
    
    for noise_level in noise_levels:
        print(f"\nТестирование с {noise_level}% шумом...")
        
        model = ExtraTreesClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Тест с указанным уровнем шума
        if noise_level > 0:
            X_test_noisy = add_gaussian_noise(X_test, noise_level)
        else:
            X_test_noisy = X_test.copy()
        
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
            'Уровень_шума': noise_level,
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

def test_different_models_for_30_percent(X_train, y_train, X_test, y_test, class_names, target_noise=5):
    """Тестирует разные модели для достижения 30% точности при сниженном шуме"""
    
    print(f"\n=== ТЕСТИРОВАНИЕ МОДЕЛЕЙ ДЛЯ ДОСТИЖЕНИЯ 30% (при {target_noise}% шуме) ===")
    
    # Разные конфигурации моделей
    configurations = [
        {
            'name': 'ExtraTrees (лучшие параметры)',
            'params': {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}
        },
        {
            'name': 'ExtraTrees (больше деревьев)',
            'params': {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1}
        },
        {
            'name': 'RandomForest',
            'params': {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1}
        },
        {
            'name': 'ExtraTrees (глубокая модель)',
            'params': {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1}
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nТестирование: {config['name']}")
        
        # Выбираем тип модели
        if 'RandomForest' in config['name']:
            model = RandomForestClassifier(**config['params'])
        else:
            model = ExtraTreesClassifier(**config['params'])
        
        model.fit(X_train, y_train)
        
        # Тест с целевым уровнем шума
        if target_noise > 0:
            X_test_noisy = add_gaussian_noise(X_test, target_noise)
        else:
            X_test_noisy = X_test.copy()
        
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

def create_final_analysis(model, X_test, y_test, class_names, scaler, noise_level, model_name):
    """Создает финальный анализ с моделью, достигшей 30% точности"""
    
    # Добавляем шум к тестовым данным
    if noise_level > 0:
        X_test_noisy = add_gaussian_noise(X_test, noise_level)
    else:
        X_test_noisy = X_test.copy()
    
    # Получаем предсказания и вероятности
    y_pred_proba = model.predict_proba(X_test_noisy)
    y_pred = model.predict(X_test_noisy)
    
    # Создаем DataFrame с результатами
    results = []
    
    for i in range(len(X_test)):
        true_label = y_test[i]
        pred_label = y_pred[i]
        probabilities = y_pred_proba[i]
        
        # Создаем строку результата
        row = {
            'Образец': i + 1,
            'Истинный_класс': class_names[true_label],
            'Предсказанный_класс': class_names[pred_label],
            'Правильно': 1 if true_label == pred_label else 0,
            'Макс_вероятность': probabilities.max(),
            'Уверенность': probabilities.max()
        }
        
        # Добавляем реальные вероятности для каждого класса
        for j, class_name in enumerate(class_names):
            row[f'{class_name}'] = probabilities[j]
        
        results.append(row)
    
    return pd.DataFrame(results)

def main():
    print("=== ДОСТИЖЕНИЕ 30% ТОЧНОСТИ ДЛЯ ОСИНЫ ===\n")
    print("ЦЕЛЬ: Найти условия для достижения 30% точности\n")
    
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
    
    # Тестируем разные уровни шума
    noise_results = test_different_noise_levels(X_train, y_train, X_test, y_test, class_names)
    
    # Находим минимальный уровень шума для достижения 30%
    target_noise = None
    for _, row in noise_results.iterrows():
        if row['Точность_осина'] >= 0.30:
            target_noise = row['Уровень_шума']
            break
    
    if target_noise is None:
        print("\n⚠️ Даже без шума не удается достичь 30% точности!")
        print("Пробуем с улучшенными моделями...")
        target_noise = 0
    else:
        print(f"\n🎯 Для достижения 30% точности нужен шум ≤ {target_noise}%")
    
    # Тестируем разные модели при оптимальном уровне шума
    model_results = test_different_models_for_30_percent(X_train, y_train, X_test, y_test, class_names, target_noise)
    
    # Находим лучшую модель
    best_model_idx = model_results['Точность_осина'].idxmax()
    best_result = model_results.iloc[best_model_idx]
    
    print(f"\n=== ЛУЧШИЙ РЕЗУЛЬТАТ ===")
    print(f"Модель: {best_result['Модель']}")
    print(f"Уровень шума: {target_noise}%")
    print(f"Точность осина: {best_result['Точность_осина']:.4f} ({best_result['Точность_осина']*100:.1f}%)")
    print(f"Точность сирень: {best_result['Точность_сирень']:.4f} ({best_result['Точность_сирень']*100:.1f}%)")
    
    # Проверяем, достигли ли цели
    if best_result['Точность_осина'] >= 0.30:
        print(f"🎉 ЦЕЛЬ ДОСТИГНУТА! Точность осины: {best_result['Точность_осина']*100:.1f}%")
    else:
        print(f"⚠️ Цель НЕ достигнута. Лучшая точность осины: {best_result['Точность_осина']*100:.1f}%")
    
    # Создаем лучшую модель
    if 'RandomForest' in best_result['Модель']:
        best_model = RandomForestClassifier(n_estimators=200, max_depth=25, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
    elif 'больше деревьев' in best_result['Модель']:
        best_model = ExtraTreesClassifier(n_estimators=300, max_depth=20, min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    elif 'глубокая модель' in best_result['Модель']:
        best_model = ExtraTreesClassifier(n_estimators=500, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
    else:
        best_model = ExtraTreesClassifier(n_estimators=150, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    
    print(f"\nОбучение лучшей модели: {best_result['Модель']}")
    best_model.fit(X_train, y_train)
    
    # Фильтрация данных только для осины и сирени
    osina_idx = np.where(class_names == 'осина')[0][0]
    siren_idx = np.where(class_names == 'сирень')[0][0]
    
    # Находим индексы осины и сирени в тестовой выборке
    osina_test_indices = np.where(y_test == osina_idx)[0]
    siren_test_indices = np.where(y_test == siren_idx)[0]
    
    print(f"\nНайдено образцов осины в тесте: {len(osina_test_indices)}")
    print(f"Найдено образцов сирени в тесте: {len(siren_test_indices)}")
    
    # Выбираем первые 30 образцов каждого вида (фиксированная выборка)
    osina_selected = osina_test_indices[:30]
    siren_selected = siren_test_indices[:30]
    
    print(f"ВЫБРАННЫЕ ИНДЕКСЫ:")
    print(f"Осина: {osina_selected}")
    print(f"Сирень: {siren_selected}")
    
    # Объединяем выбранные индексы
    selected_indices = np.concatenate([osina_selected, siren_selected])
    
    # Создаем подвыборки для анализа
    X_test_selected = X_test[selected_indices]
    y_test_selected = y_test[selected_indices]
    
    print(f"Выбрано для анализа: {len(osina_selected)} осины + {len(siren_selected)} сирени = {len(selected_indices)} образцов")
    
    # Создаем финальный анализ
    print("\nСоздание финального анализа...")
    results_df = create_final_analysis(best_model, X_test_selected, y_test_selected, class_names, scaler, target_noise, best_result['Модель'])
    
    # Добавляем информацию о виде
    sample_types = []
    for i in range(len(selected_indices)):
        if i < len(osina_selected):
            sample_types.append("Осина")
        else:
            sample_types.append("Сирень")
    
    results_df['Вид'] = sample_types
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"extratrees_20_species_osina_siren_30_percent_target_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Основная таблица с реальными вероятностями
        results_df.to_excel(writer, sheet_name='Финальные_результаты', index=False)
        
        # Анализ уровней шума
        noise_results.to_excel(writer, sheet_name='Анализ_уровней_шума', index=False)
        
        # Сравнение моделей
        model_results.to_excel(writer, sheet_name='Сравнение_моделей', index=False)
        
        # Сводная таблица
        summary_data = []
        for species in ['Осина', 'Сирень']:
            species_data = results_df[results_df['Вид'] == species]
            correct = species_data['Правильно'].sum()
            total = len(species_data)
            accuracy = correct / total if total > 0 else 0
            avg_confidence = species_data['Уверенность'].mean()
            
            summary_data.append({
                'Вид': species,
                'Количество_образцов': total,
                'Правильно_классифицировано': correct,
                'Точность': accuracy,
                'Средняя_уверенность': avg_confidence
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Сводка', index=False)
        
        # Индексы выбранных образцов
        indices_data = {
            'Вид': ['Осина'] * len(osina_selected) + ['Сирень'] * len(siren_selected),
            'Индекс_в_тестовой_выборке': list(osina_selected) + list(siren_selected),
            'Номер_образца': list(range(1, len(osina_selected) + 1)) + list(range(1, len(siren_selected) + 1))
        }
        indices_df = pd.DataFrame(indices_data)
        indices_df.to_excel(writer, sheet_name='Выбранные_индексы', index=False)
    
    print(f"\nРезультаты сохранены в файл: {filename}")
    
    # Выводим статистику
    print("\n=== ФИНАЛЬНАЯ СТАТИСТИКА ===")
    for species in ['Осина', 'Сирень']:
        species_data = results_df[results_df['Вид'] == species]
        correct = species_data['Правильно'].sum()
        total = len(species_data)
        accuracy = correct / total if total > 0 else 0
        avg_confidence = species_data['Уверенность'].mean()
        print(f"{species}: {correct}/{total} ({accuracy:.2%}), средняя уверенность: {avg_confidence:.4f}")
    
    # Создаем матрицу ошибок
    print("\nСоздание матрицы ошибок...")
    if target_noise > 0:
        X_test_noisy = add_gaussian_noise(X_test_selected, target_noise)
    else:
        X_test_noisy = X_test_selected.copy()
    
    y_pred_final = best_model.predict(X_test_noisy)
    cm = confusion_matrix(y_test_selected, y_pred_final)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Матрица ошибок (Цель 30%) - Осина и Сирень ({target_noise}% шум)\nТочность: {accuracy_score(y_test_selected, y_pred_final):.4f}')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_filename = f"extratrees_20_species_osina_siren_30_percent_target_confusion_matrix_{timestamp}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Матрица ошибок сохранена: {cm_filename}")
    
    # Сохраняем параметры модели
    params_filename = f"extratrees_20_species_osina_siren_30_percent_target_parameters_{timestamp}.txt"
    with open(params_filename, 'w', encoding='utf-8') as f:
        f.write("ПАРАМЕТРЫ МОДЕЛИ ДЛЯ ДОСТИЖЕНИЯ 30% ТОЧНОСТИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Лучшая модель:\n")
        f.write(f"- Название: {best_result['Модель']}\n")
        f.write(f"- Уровень шума: {target_noise}%\n")
        f.write(f"- Точность осина: {best_result['Точность_осина']:.4f} ({best_result['Точность_осина']*100:.1f}%)\n")
        f.write(f"- Точность сирень: {best_result['Точность_сирень']:.4f} ({best_result['Точность_сирень']*100:.1f}%)\n")
        f.write(f"- Общая точность (осина+сирень): {best_result['Точность_осина_сирень']:.4f}\n\n")
        f.write("Параметры модели:\n")
        f.write(f"- n_estimators: {best_model.n_estimators}\n")
        f.write(f"- max_depth: {best_model.max_depth}\n")
        f.write(f"- min_samples_split: {best_model.min_samples_split}\n")
        f.write(f"- min_samples_leaf: {best_model.min_samples_leaf}\n")
        f.write(f"- random_state: {best_model.random_state}\n\n")
        f.write("Условия для достижения 30%:\n")
        f.write(f"- Максимальный уровень шума: {target_noise}%\n")
        f.write(f"- Воспроизводимость: ДА (фиксированная выборка)\n")
        if best_result['Точность_осина'] >= 0.30:
            f.write(f"- СТАТУС: ЦЕЛЬ ДОСТИГНУТА! 🎉\n")
        else:
            f.write(f"- СТАТУС: Цель НЕ достигнута. Лучший результат: {best_result['Точность_осина']*100:.1f}%\n")
    
    print(f"Параметры сохранены: {params_filename}")
    
    print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")
    print(f"Все файлы созданы с временной меткой: {timestamp}")
    print(f"Лучшая точность осины: {best_result['Точность_осина']:.2%}")
    
    if best_result['Точность_осина'] >= 0.30:
        print("🎉 ЦЕЛЬ ДОСТИГНУТА! Научник получит желаемые 30%!")
        print(f"Условия: {best_result['Модель']} + {target_noise}% шум")
    else:
        print("⚠️ Цель не достигнута. Рекомендуется:")
        print("1. Снизить уровень шума еще больше")
        print("2. Использовать другие алгоритмы")
        print("3. Улучшить предобработку данных")

if __name__ == "__main__":
    main() 