import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("TensorFlow доступен!")
except ImportError:
    print("⚠️ TensorFlow недоступен. Используем альтернативную реализацию.")
    TF_AVAILABLE = False
    # Альтернативная реализация через PyTorch или другие библиотеки
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        PYTORCH_AVAILABLE = True
        print("PyTorch доступен для альтернативной реализации!")
    except ImportError:
        PYTORCH_AVAILABLE = False
        print("⚠️ PyTorch также недоступен. Используем симуляцию.")

def load_spectral_data():
    """Загружает спектральные данные растительности для 1D-AlexNet"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_spectra = []
    all_labels = []
    
    print("🌿 Загрузка спектральных данных растительности...")
    print("="*60)
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"📁 {tree_type}: {len(excel_files)} файлов")
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    
                    if df.shape[1] >= 2:
                        # Берем спектральные данные (второй столбец)
                        spectrum = df.iloc[:, 1].values
                        
                        # Очистка от NaN
                        spectrum = spectrum[~np.isnan(spectrum)]
                        
                        if len(spectrum) >= 100:  # Минимум для надежной классификации
                            all_spectra.append(spectrum)
                            all_labels.append(tree_type)
                            
                except Exception as e:
                    print(f"❗️ Ошибка при чтении файла {file_path}: {e}")
                    continue
    
    print(f"✅ Загружено {len(all_spectra)} спектров растительности")
    return all_spectra, all_labels, tree_types

def preprocess_spectra_for_1d_alexnet(spectra, labels, target_length=300):
    """
    Предобрабатывает спектры для 1D-AlexNet с использованием интерполяции.
    """
    print("\n🔧 Предобработка спектров для 1D-AlexNet (улучшенный метод)...")
    print(f"📏 Целевая длина спектра: {target_length} (с интерполяцией)")
    
    # Приводим все спектры к одинаковой длине через интерполяцию
    processed_spectra = []
    for spectrum in spectra:
        # Пропускаем очень короткие спектры
        if len(spectrum) < 50:
            continue
            
        # Интерполируем, если длина не совпадает
        if len(spectrum) != target_length:
            processed_spectrum = np.interp(
                np.linspace(0, len(spectrum) - 1, target_length),
                np.arange(len(spectrum)),
                spectrum
            )
        else:
            processed_spectrum = spectrum
            
        processed_spectra.append(processed_spectrum)
    
    # Преобразуем в numpy массив
    X = np.array(processed_spectra, dtype=np.float32)
    
    # Кодируем метки классов
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"📊 Форма данных: {X.shape}")
    print(f"🎯 Количество классов: {len(np.unique(y))}")
    print(f"🏷️ Классы: {label_encoder.classes_}")
    
    return X, y, label_encoder, target_length

def create_1d_alexnet_tensorflow(input_shape, num_classes):
    """Создает 1D-AlexNet архитектуру в TensorFlow согласно статье"""
    model = keras.Sequential([
        # Первый сверточный блок
        layers.Conv1D(filters=96, kernel_size=11, strides=4, activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.BatchNormalization(),
        
        # Второй сверточный блок
        layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.BatchNormalization(),
        
        # Третий сверточный блок
        layers.Conv1D(filters=384, kernel_size=3, activation='relu', padding='same'),
        
        # Четвертый сверточный блок
        layers.Conv1D(filters=384, kernel_size=3, activation='relu', padding='same'),
        
        # Пятый сверточный блок
        layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=3, strides=2),
        
        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def simulate_1d_alexnet_training(X_train, y_train, X_test, y_test, num_classes):
    """Симуляция обучения 1D-AlexNet без TensorFlow"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    
    print("🤖 Симуляция 1D-AlexNet через глубокую нейронную сеть (оптимизированная)...")
    
    # Создаем глубокую сеть с улучшенными параметрами
    model = MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128), # Оптимизированная архитектура
        activation='relu',
        solver='adam',
        max_iter=500, # Уменьшено, т.к. есть ранняя остановка
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20, # Параметр для ранней остановки
        validation_fraction=0.1,
        learning_rate_init=0.0001, # Уменьшенная скорость обучения
        batch_size=32 # Фиксированный размер батча
    )
    
    print("📚 Обучение симулированной 1D-AlexNet...")
    model.fit(X_train, y_train)
    
    # Базовая оценка
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"📈 Точность на обучающей выборке: {train_accuracy:.4f}")
    print(f"📊 Точность на тестовой выборке: {test_accuracy:.4f}")
    
    return model

def test_with_gaussian_noise_1000_realizations(model, X_test, y_test, tree_types, noise_levels):
    """Тестирование с гауссовским шумом - 1000 реализаций как в статье"""
    print("\n" + "="*70)
    print("🎲 ТЕСТИРОВАНИЕ С ГАУССОВСКИМ ШУМОМ (1000 РЕАЛИЗАЦИЙ)")
    print("="*70)
    
    n_realizations = 1000  # Как в статье
    results = {}
    
    for noise_level in noise_levels:
        print(f"\n🔊 Уровень шума: {noise_level * 100:.1f}%")
        print("-" * 50)
        
        accuracies = []
        all_predictions = []
        all_true_labels = []
        
        # 1000 реализаций шума
        for realization in range(n_realizations):
            if realization % 100 == 0:
                print(f"  Реализация {realization + 1}/1000...")
            
            # Добавляем гауссовский шум с нулевым средним
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, X_test.shape).astype(np.float32)
                X_test_noisy = X_test + noise
            else:
                X_test_noisy = X_test
            
            # Предсказание
            y_pred = model.predict(X_test_noisy)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            # Сохраняем для первой реализации
            if realization == 0:
                all_predictions = y_pred
                all_true_labels = y_test
        
        # Вычисляем статистики
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"📊 Средняя точность: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Отчет о классификации для первой реализации
        print(f"\n📋 Отчет о классификации (шум {noise_level * 100:.1f}%):")
        print(classification_report(all_true_labels, all_predictions, 
                                  target_names=tree_types, digits=3))
        
        # Матрица ошибок
        cm = confusion_matrix(all_true_labels, all_predictions)
        print("\n📊 Матрица ошибок:")
        print(cm)
        
        # Вероятности правильной классификации по классам
        print(f"\n✅ Вероятности правильной классификации по классам:")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, tree in enumerate(tree_types):
            print(f"  {tree}: {class_accuracies[i]:.3f}")
        
        # Вероятность ложной тревоги (False Positive Rate) для каждого класса
        print(f"\n🚨 Вероятность ложной тревоги (FPR) для каждого класса:")
        for i, tree in enumerate(tree_types):
            FP = cm.sum(axis=0)[i] - cm[i, i]  # Ложные срабатывания
            TN = cm.sum() - cm.sum(axis=0)[i] - cm.sum(axis=1)[i] + cm[i, i]  # Истинные отрицательные
            FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
            print(f"  {tree}: {FPR:.3f}")
        
        # Сохраняем результаты
        results[noise_level] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm,
            'all_accuracies': accuracies
        }
    
    return results

def plot_noise_analysis(results, tree_types):
    """Строит графики анализа устойчивости к шуму"""
    noise_levels = list(results.keys())
    mean_accuracies = [results[noise]['mean_accuracy'] for noise in noise_levels]
    std_accuracies = [results[noise]['std_accuracy'] for noise in noise_levels]
    
    # График общей точности
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.errorbar([n*100 for n in noise_levels], mean_accuracies, yerr=std_accuracies, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность классификации')
    plt.title('Устойчивость 1D-AlexNet к гауссовскому шуму')
    plt.grid(True, alpha=0.3)
    
    # График точности по классам
    plt.subplot(2, 2, 2)
    for i, tree in enumerate(tree_types):
        class_accs = [results[noise]['class_accuracies'][i] for noise in noise_levels]
        plt.plot([n*100 for n in noise_levels], class_accs, marker='o', label=tree)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность по классам')
    plt.title('Точность классификации по видам растительности')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Гистограмма точностей для максимального шума
    plt.subplot(2, 2, 3)
    max_noise = max(noise_levels)
    accuracies = results[max_noise]['all_accuracies']
    plt.hist(accuracies, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Точность')
    plt.ylabel('Частота')
    plt.title(f'Распределение точности (шум {max_noise*100}%)')
    plt.grid(True, alpha=0.3)
    
    # Тепловая карта матрицы ошибок
    plt.subplot(2, 2, 4)
    cm = results[0.0]['confusion_matrix']  # Без шума
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (без шума)')
    plt.colorbar()
    tick_marks = np.arange(len(tree_types))
    plt.xticks(tick_marks, tree_types, rotation=45)
    plt.yticks(tick_marks, tree_types)
    
    plt.tight_layout()
    plt.savefig('1d_alexnet_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция для реализации 1D-AlexNet классификации"""
    print("🌲 КЛАССИФИКАЦИЯ РАСТИТЕЛЬНОСТИ С 1D-AlexNet")
    print("=" * 70)
    print("📄 Реализация согласно статье с 1000 реализациями шума")
    print("=" * 70)

    # Загрузка данных
    spectra, labels, tree_types = load_spectral_data()

    if len(spectra) == 0:
        print("❌ Не удалось загрузить данные!")
        return

    # Предобработка с фиксированной длиной
    X, y, label_encoder, input_length = preprocess_spectra_for_1d_alexnet(spectra, labels, target_length=300)

    # Разделение данных на обучающую и тестовую выборки 50/50
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    print(f"\n📏 Размеры данных:")
    print(f"  Обучающая выборка: {X_train.shape}")
    print(f"  Тестовая выборка: {X_test.shape}")

    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создание и обучение модели
    if TF_AVAILABLE:
        print("\n🚀 Создание 1D-AlexNet в TensorFlow...")
        model = create_1d_alexnet_tensorflow((input_length, 1), len(tree_types))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Обучение
        history = model.fit(
            X_train_scaled.reshape(-1, input_length, 1),
            y_train,
            validation_data=(X_test_scaled.reshape(-1, input_length, 1), y_test),
            epochs=100,
            batch_size=32,
            verbose=1
        )
        
        # Сохранение модели
        model.save('1d_alexnet_vegetation_classifier.h5')
        
    else:
        print("\n🤖 Использование симулированной 1D-AlexNet...")
        model = simulate_1d_alexnet_training(X_train_scaled, y_train, X_test_scaled, y_test, len(tree_types))

    # Тестирование с гауссовским шумом
    # Уровни шума 1%, 5%, 10% (в виде стандартных отклонений)
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    results = test_with_gaussian_noise_1000_realizations(
        model, X_test_scaled, y_test, tree_types, noise_levels
    )

    # Построение графиков анализа
    plot_noise_analysis(results, tree_types)

    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("📊 Результаты согласно методологии статьи:")
    print("   - Разделение данных 50/50")
    print("   - 1000 реализаций гауссовского шума на уровнях 1%, 5%, 10%")
    print("   - Вероятности правильной классификации по классам")
    print("   - Вероятности ложной тревоги (FPR)")
    print("="*70)

if __name__ == "__main__":
    main() 