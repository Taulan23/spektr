import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data_from_folders():
    """Загружает данные из папок с Excel файлами"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    print("Загрузка данных...")
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"Найдено {len(excel_files)} файлов для {tree_type}")
            
            for file_path in excel_files:
                try:
                    # Чтение Excel файла
                    df = pd.read_excel(file_path)
                    
                    # Предполагаем, что данные в первых двух столбцах (волновая длина и интенсивность)
                    if df.shape[1] >= 2:
                        # Берем только числовые значения интенсивности
                        spectrum_data = df.iloc[:, 1].values
                        
                        # Проверяем на валидность данных
                        if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                            # Удаляем NaN значения
                            spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                            
                            if len(spectrum_data) > 10:  # Минимум 10 точек для спектра
                                all_data.append(spectrum_data)
                                all_labels.append(tree_type)
                                
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
                    continue
        else:
            print(f"Папка {folder_path} не найдена")
    
    print(f"Загружено {len(all_data)} спектров")
    return all_data, all_labels, tree_types

def preprocess_data(all_data, all_labels):
    """Предобрабатывает данные"""
    print("Предобработка данных...")
    
    # Находим минимальную длину спектра
    min_length = min(len(spectrum) for spectrum in all_data)
    print(f"Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до одинаковой длины
    processed_data = []
    for spectrum in all_data:
        # Берем первые min_length точек
        truncated_spectrum = spectrum[:min_length]
        processed_data.append(truncated_spectrum)
    
    # Преобразуем в numpy массив
    X = np.array(processed_data, dtype=np.float32)
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(all_labels)
    
    print(f"Форма данных: {X.shape}")
    print(f"Количество классов: {len(np.unique(y))}")
    print(f"Классы: {label_encoder.classes_}")
    
    return X, y, label_encoder

def create_models():
    """Создает несколько моделей для сравнения"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    return models

def plot_feature_importance(model, model_name, n_features=20):
    """Строит график важности признаков для Random Forest"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Важность признаков - {model_name}')
        plt.bar(range(n_features), importances[indices])
        plt.xlabel('Индекс спектральной точки')
        plt.ylabel('Важность')
        plt.xticks(range(n_features), [f'Точка {i}' for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_model_with_noise(model, test_data_scaled, test_labels, tree_types, noise_levels, n_realizations=10):
    """Тестирует модель с различными уровнями шума - ваш оригинальный код адаптированный для sklearn"""
    print(f"\nТестирование модели с различными уровнями шума...")
    
    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"Тестирование с уровнем шума: {noise_level * 100}%")
        print(f"{'='*60}")
        
        accuracies = []
        
        for i in range(n_realizations):
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, test_data_scaled.shape).astype(np.float32)
                X_test_noisy = test_data_scaled + noise
            else:
                X_test_noisy = test_data_scaled
            
            # Предсказание
            y_pred = model.predict(X_test_noisy)
            test_accuracy = accuracy_score(test_labels, y_pred)
            accuracies.append(test_accuracy)
            
            if i == 0:
                y_pred_classes = y_pred
                y_test_classes = test_labels

        mean_accuracy = np.mean(accuracies)
        print(f"Средняя точность тестирования (шум {noise_level * 100}%): {mean_accuracy:.4f} ± {np.std(accuracies):.4f}")
        print(f"Отчет о классификации (шум {noise_level * 100}%):")
        print(classification_report(y_test_classes, y_pred_classes, target_names=tree_types, digits=3))
        
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        print("Матрица ошибок:")
        print(cm)
        
        print("Коэффициент ложных срабатываний (FPR) для каждого класса:")
        for i, tree in enumerate(tree_types):
            FP = cm.sum(axis=0)[i] - cm[i, i]
            TN = cm.sum() - cm.sum(axis=0)[i] - cm.sum(axis=1)[i] + cm[i, i]
            FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
            print(f"{tree}: {FPR:.3f}")

def save_model_and_scaler(model, scaler, label_encoder, model_name):
    """Сохраняет модель и предобработчики"""
    model_filename = f'tree_classification_{model_name.replace(" ", "_").lower()}.pkl'
    joblib.dump(model, model_filename)
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    print(f"Модель {model_name} и предобработчики сохранены!")

def plot_confusion_matrix(cm, tree_types, model_name):
    """Строит тепловую карту матрицы ошибок"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Матрица ошибок - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(tree_types))
    plt.xticks(tick_marks, tree_types, rotation=45)
    plt.yticks(tick_marks, tree_types)
    
    # Добавляем числа в ячейки
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("Начало программы классификации спектров деревьев (scikit-learn версия)")
    print("="*70)
    
    # Загрузка данных
    all_data, all_labels, tree_types = load_data_from_folders()
    
    if len(all_data) == 0:
        print("Не удалось загрузить данные. Проверьте наличие Excel файлов в папках.")
        return
    
    # Предобработка данных
    X, y, label_encoder = preprocess_data(all_data, all_labels)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создание и обучение моделей
    models = create_models()
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Обучение модели: {model_name}")
        print(f"{'='*70}")
        
        # Обучение модели
        model.fit(X_train_scaled, y_train)
        
        # Базовая оценка модели
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        
        # Отчет о классификации
        print("\nОтчет о классификации:")
        print(classification_report(y_test, y_pred, target_names=tree_types))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, tree_types, model_name)
        
        # График важности признаков для Random Forest
        if model_name == 'Random Forest':
            plot_feature_importance(model, model_name)
        
        # Сохраняем лучшую модель
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model
            best_model_name = model_name
        
        # Сохранение модели
        save_model_and_scaler(model, scaler, label_encoder, model_name)
    
    # Тестирование лучшей модели с различными уровнями шума
    print(f"\n{'='*70}")
    print(f"ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ ({best_model_name}) С ШУМОМ")
    print(f"{'='*70}")
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
    test_model_with_noise(best_model, X_test_scaled, y_test, tree_types, noise_levels)
    
    print("\n" + "="*70)
    print("ПРОГРАММА ЗАВЕРШЕНА УСПЕШНО!")
    print(f"Лучшая модель: {best_model_name} с точностью {best_accuracy:.4f}")
    print("="*70)

if __name__ == "__main__":
    main() 