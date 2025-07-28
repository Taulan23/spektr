import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
    print("⚠️ TensorFlow недоступен. Используем оптимизированную симуляцию.")
    TF_AVAILABLE = False

def load_spectral_data_enhanced():
    """Загружает спектральные данные с улучшенной предобработкой"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_spectra = []
    all_labels = []
    
    print("🌿 Загрузка спектральных данных растительности (оптимизированная)...")
    print("="*70)
    
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
                        
                        # Улучшенная очистка от NaN
                        spectrum = spectrum[~np.isnan(spectrum)]
                        
                        # Дополнительная фильтрация выбросов
                        if len(spectrum) >= 100:
                            # Удаляем экстремальные выбросы (за 3 сигмы)
                            mean_val = np.mean(spectrum)
                            std_val = np.std(spectrum)
                            mask = np.abs(spectrum - mean_val) <= 3 * std_val
                            spectrum = spectrum[mask]
                            
                            if len(spectrum) >= 100:
                                all_spectra.append(spectrum)
                                all_labels.append(tree_type)
                            
                except Exception as e:
                    continue
    
    print(f"✅ Загружено {len(all_spectra)} спектров растительности")
    return all_spectra, all_labels, tree_types

def create_optimized_1d_alexnet_tensorflow(input_shape, num_classes):
    """Создает оптимизированную 1D-AlexNet с лучшими параметрами"""
    model = keras.Sequential([
        # Первый сверточный блок - увеличенные фильтры
        layers.Conv1D(filters=128, kernel_size=11, strides=2, activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Dropout(0.2),
        
        # Второй сверточный блок
        layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Dropout(0.3),
        
        # Третий сверточный блок
        layers.Conv1D(filters=384, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Четвертый сверточный блок
        layers.Conv1D(filters=384, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Пятый сверточный блок
        layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=3, strides=2),
        layers.Dropout(0.4),
        
        # Полносвязные слои - увеличенные размеры
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_optimized_simulation_model(X_train, y_train, X_val, y_val):
    """Создает оптимизированную симуляцию 1D-AlexNet с лучшими параметрами"""
    print("🤖 Создание оптимизированной симуляции 1D-AlexNet...")
    
    # Попробуем несколько архитектур и выберем лучшую
    models = {
        'Deep_MLP': MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256, 128, 64),  # Глубже
            activation='relu',
            solver='adam',
            max_iter=2000,  # Больше эпох
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.001,
            batch_size=32,  # Оптимальный батч
            alpha=0.0001,  # L2 регуляризация
            beta_1=0.9,
            beta_2=0.999
        ),
        'Wide_MLP': MLPClassifier(
            hidden_layer_sizes=(2048, 1024, 512, 256),  # Шире
            activation='relu',
            solver='adam',
            max_iter=1500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.0005,
            batch_size=16,  # Меньший батч для более точного обучения
            alpha=0.0001
        ),
        'Gradient_Boost': GradientBoostingClassifier(
            n_estimators=500,  # Больше деревьев
            learning_rate=0.1,
            max_depth=8,
            random_state=42,
            subsample=0.8,
            max_features='sqrt'
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"  🔧 Обучение {name}...")
        
        model.fit(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        
        print(f"    Точность на валидации: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_name = name
    
    print(f"🏆 Лучшая модель: {best_name} с точностью {best_accuracy:.4f}")
    return best_model

def enhanced_data_augmentation(X, y, augment_factor=2):
    """Расширение данных для улучшения обучения"""
    print(f"🔄 Аугментация данных (фактор {augment_factor})...")
    
    augmented_X = [X]
    augmented_y = [y]
    
    for i in range(augment_factor):
        # Добавляем небольшой шум
        noise_level = 0.01 * (i + 1)
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        
        # Небольшой сдвиг
        shift = np.random.randint(-2, 3, X.shape[0])
        X_shifted = np.array([np.roll(spectrum, s) for spectrum, s in zip(X, shift)])
        
        augmented_X.extend([X_noisy, X_shifted])
        augmented_y.extend([y, y])
    
    X_augmented = np.vstack(augmented_X)
    y_augmented = np.hstack(augmented_y)
    
    print(f"  📊 Размер после аугментации: {X_augmented.shape}")
    return X_augmented, y_augmented

def test_optimized_noise_robustness(model, X_test, y_test, tree_types, noise_levels, n_realizations=1000):
    """Оптимизированное тестирование с 1000 реализациями"""
    print("\n" + "="*70)
    print("🎯 ОПТИМИЗИРОВАННОЕ ТЕСТИРОВАНИЕ С ШУМОМ (1000 РЕАЛИЗАЦИЙ)")
    print("="*70)
    
    results = {}
    target_results = {
        'береза': [0.944, 0.939, 0.919],
        'дуб': [0.783, 0.820, 0.827],
        'клен': [0.818, 0.821, 0.830],
        'липа': [0.931, 0.875, 0.791],
        'осина': [0.821, 0.751, 0.640],
        'ель': [0.914, 0.908, 0.881],
        'сосна': [0.854, 0.832, 0.792]
    }
    
    noise_map = {0.01: 0, 0.05: 1, 0.1: 2}  # Маппинг для целевых результатов
    
    for noise_level in noise_levels:
        print(f"\n🔊 Уровень шума: {noise_level * 100:.1f}%")
        print("-" * 50)
        
        accuracies = []
        class_correct = np.zeros(len(tree_types))
        class_total = np.zeros(len(tree_types))
        
        # 1000 реализаций
        for realization in range(n_realizations):
            if realization % 200 == 0:
                print(f"  Реализация {realization + 1}/1000...")
            
            # Гауссовский шум с нулевым средним
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, X_test.shape).astype(np.float32)
                X_test_noisy = X_test + noise
            else:
                X_test_noisy = X_test
            
            # Предсказание
            y_pred = model.predict(X_test_noisy)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            # Подсчет правильных классификаций по классам
            for i in range(len(tree_types)):
                mask = (y_test == i)
                class_total[i] += np.sum(mask)
                class_correct[i] += np.sum((y_pred == i) & mask)
            
            # Сохраняем для первой реализации
            if realization == 0:
                first_pred = y_pred
                first_true = y_test
        
        # Вычисляем статистики
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        class_accuracies = class_correct / class_total
        
        print(f"📊 Средняя точность: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Сравнение с целевыми результатами
        if noise_level in noise_map:
            target_idx = noise_map[noise_level]
            print(f"\n🎯 Сравнение с целевыми результатами (шум {noise_level*100}%):")
            total_diff = 0
            for i, tree in enumerate(tree_types):
                target_val = target_results[tree][target_idx]
                our_val = class_accuracies[i]
                diff = abs(target_val - our_val)
                total_diff += diff
                status = "✅" if diff < 0.05 else "⚠️" if diff < 0.1 else "❌"
                print(f"  {tree}: {our_val:.3f} (цель: {target_val:.3f}) {status}")
            
            avg_diff = total_diff / len(tree_types)
            print(f"  Средняя разница: {avg_diff:.3f}")
        
        # Отчет о классификации
        print(f"\n📋 Отчет о классификации:")
        print(classification_report(first_true, first_pred, target_names=tree_types, digits=3))
        
        # Матрица ошибок и FPR
        cm = confusion_matrix(first_true, first_pred)
        print("\n🚨 Вероятность ложной тревоги (FPR):")
        for i, tree in enumerate(tree_types):
            FP = cm.sum(axis=0)[i] - cm[i, i]
            TN = cm.sum() - cm.sum(axis=0)[i] - cm.sum(axis=1)[i] + cm[i, i]
            FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
            print(f"  {tree}: {FPR:.3f}")
        
        results[noise_level] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm
        }
    
    return results

def plot_comparison_with_target(results, tree_types):
    """Строит сравнительные графики с целевыми результатами"""
    target_results = {
        'береза': [0.944, 0.939, 0.919],
        'дуб': [0.783, 0.820, 0.827],
        'клен': [0.818, 0.821, 0.830],
        'липа': [0.931, 0.875, 0.791],
        'осина': [0.821, 0.751, 0.640],
        'ель': [0.914, 0.908, 0.881],
        'сосна': [0.854, 0.832, 0.792]
    }
    
    noise_levels_target = [0.01, 0.05, 0.1]
    noise_levels_our = [0.01, 0.05, 0.1]
    
    plt.figure(figsize=(20, 15))
    
    # График сравнения по классам
    for i, tree in enumerate(tree_types):
        plt.subplot(3, 3, i + 1)
        
        # Целевые результаты
        target_vals = target_results[tree]
        plt.plot([n*100 for n in noise_levels_target], target_vals, 
                'o-', label='Статья (цель)', linewidth=2, markersize=8)
        
        # Наши результаты
        our_vals = [results[noise]['class_accuracies'][i] for noise in noise_levels_our]
        plt.plot([n*100 for n in noise_levels_our], our_vals, 
                's-', label='Наши результаты', linewidth=2, markersize=8)
        
        plt.xlabel('Уровень шума (%)')
        plt.ylabel('Точность классификации')
        plt.title(f'{tree.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.5, 1.0)
    
    # Общий график
    plt.subplot(3, 3, 8)
    
    # Средние значения
    target_means = [np.mean([target_results[tree][i] for tree in tree_types]) 
                   for i in range(3)]
    our_means = [np.mean([results[noise]['class_accuracies'] for noise in noise_levels_our])]
    
    plt.plot([n*100 for n in noise_levels_target], target_means, 
            'o-', label='Статья (среднее)', linewidth=3, markersize=10)
    
    overall_accuracies = [results[noise]['mean_accuracy'] for noise in noise_levels_our]
    plt.plot([n*100 for n in noise_levels_our], overall_accuracies, 
            's-', label='Наши результаты (общие)', linewidth=3, markersize=10)
    
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Средняя точность')
    plt.title('ОБЩЕЕ СРАВНЕНИЕ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_1d_alexnet_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Оптимизированная главная функция"""
    print("🌲 ОПТИМИЗИРОВАННАЯ КЛАССИФИКАЦИЯ РАСТИТЕЛЬНОСТИ (1D-AlexNet)")
    print("=" * 70)
    print("🎯 Цель: достичь результатов из научной статьи")
    print("=" * 70)
    
    # Загрузка с улучшенной предобработкой
    spectra, labels, tree_types = load_spectral_data_enhanced()
    
    if len(spectra) == 0:
        print("❌ Не удалось загрузить данные!")
        return
    
    # Предобработка
    lengths = [len(s) for s in spectra]
    target_length = min(lengths)
    print(f"📏 Целевая длина спектра: {target_length}")
    
    # Обрезаем до одинаковой длины
    X = np.array([spectrum[:target_length] for spectrum in spectra], dtype=np.float32)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"📊 Форма данных: {X.shape}")
    print(f"🎯 Классы: {label_encoder.classes_}")
    
    # Разделение с стратификацией
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp
    )
    
    print(f"📏 Размеры данных:")
    print(f"  Обучающая: {X_train.shape}")
    print(f"  Валидационная: {X_val.shape}")
    print(f"  Тестовая: {X_test.shape}")
    
    # Аугментация данных
    X_train_aug, y_train_aug = enhanced_data_augmentation(X_train, y_train, augment_factor=1)
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    if TF_AVAILABLE:
        print("\n🚀 Обучение оптимизированной 1D-AlexNet в TensorFlow...")
        model = create_optimized_1d_alexnet_tensorflow((target_length, 1), len(tree_types))
        
        # Оптимизированные параметры компиляции
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Коллбэки для улучшения обучения
        callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
            keras.callbacks.ModelCheckpoint('best_1d_alexnet.h5', save_best_only=True)
        ]
        
        # Обучение с большим количеством эпох
        history = model.fit(
            X_train_scaled.reshape(-1, target_length, 1),
            y_train_aug,
            validation_data=(X_val_scaled.reshape(-1, target_length, 1), y_val),
            epochs=200,  # Больше эпох
            batch_size=16,  # Оптимальный батч
            callbacks=callbacks,
            verbose=1
        )
        
    else:
        print("\n🤖 Обучение оптимизированной симуляции...")
        model = create_optimized_simulation_model(X_train_scaled, y_train_aug, X_val_scaled, y_val)
    
    # Тестирование с оптимизированными параметрами
    noise_levels = [0.01, 0.05, 0.1]  # Соответствуют статье
    
    results = test_optimized_noise_robustness(
        model, X_test_scaled, y_test, tree_types, noise_levels, n_realizations=1000
    )
    
    # Построение сравнительных графиков
    plot_comparison_with_target(results, tree_types)
    
    print("\n" + "="*70)
    print("✅ ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
    print("🎯 Результаты сравнены с целевыми значениями из статьи")
    print("📊 График сравнения: optimized_1d_alexnet_comparison.png")
    print("="*70)

if __name__ == "__main__":
    main() 