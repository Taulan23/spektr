import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

def load_spectral_data_enhanced():
    """Загружает спектральные данные с улучшенной предобработкой"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_spectra = []
    all_labels = []
    
    print("🌿 ОПТИМИЗИРОВАННАЯ загрузка данных для достижения целевых результатов...")
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
                        spectrum = df.iloc[:, 1].values
                        spectrum = spectrum[~np.isnan(spectrum)]
                        
                        # Улучшенная предобработка (менее агрессивная)
                        if len(spectrum) >= 100:
                            # Мягкая фильтрация только экстремальных выбросов (за 5 сигм)
                            if len(spectrum) > 50:  # Только если достаточно данных для статистики
                                mean_val = np.mean(spectrum)
                                std_val = np.std(spectrum)
                                if std_val > 0:  # Проверяем на деление на ноль
                                    mask = np.abs(spectrum - mean_val) <= 5 * std_val
                                    spectrum = spectrum[mask]
                            
                            if len(spectrum) >= 50:  # Понижаем требования
                                all_spectra.append(spectrum)
                                all_labels.append(tree_type)
                            
                except Exception as e:
                    continue
    
    print(f"✅ Загружено {len(all_spectra)} качественных спектров")
    return all_spectra, all_labels, tree_types

def create_optimized_model_ensemble():
    """Создает ансамбль оптимизированных моделей"""
    models = {
        'Deep_Neural_Network': MLPClassifier(
            hidden_layer_sizes=(2048, 1024, 512, 256, 128),  # Более глубокая сеть
            activation='relu',
            solver='adam',
            max_iter=3000,  # Значительно больше эпох
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            learning_rate_init=0.0005,  # Меньший learning rate
            batch_size=16,  # Оптимальный batch size
            alpha=0.0001,  # L2 регуляризация
            beta_1=0.9,
            beta_2=0.999,
            n_iter_no_change=50  # Больше терпения для early stopping
        ),
        'Wide_Neural_Network': MLPClassifier(
            hidden_layer_sizes=(3072, 1536, 768, 384),  # Широкая сеть
            activation='relu',
            solver='adam',
            max_iter=2500,
            random_state=43,
            early_stopping=True,
            validation_fraction=0.2,
            learning_rate_init=0.0003,
            batch_size=8,  # Еще меньший batch для лучшей точности
            alpha=0.0001
        ),
        'Gradient_Boost_Optimized': GradientBoostingClassifier(
            n_estimators=1000,  # Много деревьев
            learning_rate=0.05,  # Медленное обучение
            max_depth=10,
            random_state=42,
            subsample=0.8,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2
        )
    }
    
    return models

def enhanced_data_augmentation(X, y, augment_factor=3):
    """Усиленная аугментация данных"""
    print(f"🔄 Усиленная аугментация данных (фактор {augment_factor})...")
    
    augmented_X = [X]
    augmented_y = [y]
    
    for i in range(augment_factor):
        # Различные типы шума
        noise_level = 0.005 * (i + 1)
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        
        # Сдвиги
        shift_range = min(5, X.shape[1] // 20)
        shifts = np.random.randint(-shift_range, shift_range + 1, X.shape[0])
        X_shifted = np.array([np.roll(spectrum, s) for spectrum, s in zip(X, shifts)])
        
        # Масштабирование
        scale_factors = np.random.uniform(0.95, 1.05, (X.shape[0], 1))
        X_scaled = X * scale_factors
        
        augmented_X.extend([X_noisy, X_shifted, X_scaled])
        augmented_y.extend([y, y, y])
    
    X_augmented = np.vstack(augmented_X)
    y_augmented = np.hstack(augmented_y)
    
    print(f"  📊 Размер после аугментации: {X_augmented.shape}")
    return X_augmented, y_augmented

def train_optimized_ensemble(models, X_train, y_train, X_val, y_val):
    """Обучает ансамбль моделей и выбирает лучшую"""
    print("🤖 Обучение оптимизированного ансамбля моделей...")
    
    trained_models = {}
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n  🔧 Обучение {name}...")
        print(f"    Параметры: {model.get_params()}")
        
        # Обучение
        model.fit(X_train, y_train)
        
        # Оценка
        train_accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        
        print(f"    Точность на обучении: {train_accuracy:.4f}")
        print(f"    Точность на валидации: {val_accuracy:.4f}")
        
        trained_models[name] = model
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Лучшая модель: {best_name} с точностью {best_accuracy:.4f}")
    return best_model, trained_models

def test_with_target_comparison(model, X_test, y_test, tree_types, n_realizations=1000):
    """Тестирование с прямым сравнением с целевыми результатами из статьи"""
    print("\n" + "="*70)
    print("🎯 ТЕСТИРОВАНИЕ С ЦЕЛЕВЫМИ РЕЗУЛЬТАТАМИ (1000 РЕАЛИЗАЦИЙ)")
    print("="*70)
    
    # Целевые результаты из статьи
    target_results = {
        'береза': [0.944, 0.939, 0.919],
        'дуб': [0.783, 0.820, 0.827],
        'клен': [0.818, 0.821, 0.830],
        'липа': [0.931, 0.875, 0.791],
        'осина': [0.821, 0.751, 0.640],
        'ель': [0.914, 0.908, 0.881],
        'сосна': [0.854, 0.832, 0.792]
    }
    
    noise_levels = [0.01, 0.05, 0.1]  # δ = 1%, 5%, 10%
    noise_names = ['δ=1%', 'δ=5%', 'δ=10%']
    
    results = {}
    
    print("📊 СРАВНЕНИЕ С ЦЕЛЕВЫМИ РЕЗУЛЬТАТАМИ ИЗ СТАТЬИ:")
    print("="*70)
    print(f"{'Порода':<8} | {'δ=1%':<12} | {'δ=5%':<12} | {'δ=10%':<12}")
    print(f"{'дерева':<8} | {'Pc   Pe':<12} | {'Pc   Pe':<12} | {'Pc   Pe':<12}")
    print("-" * 70)
    
    for noise_idx, noise_level in enumerate(noise_levels):
        print(f"\n🔊 Тестирование с шумом {noise_level*100}% (1000 реализаций)...")
        
        accuracies = []
        class_correct = np.zeros(len(tree_types))
        class_total = np.zeros(len(tree_types))
        all_fps = np.zeros(len(tree_types))
        
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
                if np.sum(mask) > 0:
                    class_total[i] += np.sum(mask)
                    class_correct[i] += np.sum((y_pred[mask] == i))
            
            # Подсчет ложных срабатываний для первой реализации
            if realization == 0:
                cm = confusion_matrix(y_test, y_pred)
                for i in range(len(tree_types)):
                    FP = cm.sum(axis=0)[i] - cm[i, i]
                    TN = cm.sum() - cm.sum(axis=0)[i] - cm.sum(axis=1)[i] + cm[i, i]
                    all_fps[i] = FP / (FP + TN) if (FP + TN) != 0 else 0
        
        # Вычисляем финальные результаты
        class_accuracies = class_correct / np.maximum(class_total, 1)
        
        # Сохраняем результаты
        results[noise_level] = {
            'class_accuracies': class_accuracies,
            'false_positive_rates': all_fps,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
        
        print(f"    Средняя общая точность: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    # Форматированный вывод таблицы как в статье
    print("\n" + "="*70)
    print("📋 ФИНАЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (формат статьи):")
    print("="*70)
    print(f"{'Порода':<8} | {'δ=1%':<12} | {'δ=5%':<12} | {'δ=10%':<12}")
    print(f"{'дерева':<8} | {'Pc   Pe':<12} | {'Pc   Pe':<12} | {'Pc   Pe':<12}")
    print("-" * 70)
    
    total_diff = 0
    count_comparisons = 0
    
    for i, tree in enumerate(tree_types):
        row = f"{tree:<8} |"
        
        for noise_idx, noise_level in enumerate(noise_levels):
            our_pc = results[noise_level]['class_accuracies'][i]
            our_pe = results[noise_level]['false_positive_rates'][i]
            target_pc = target_results[tree][noise_idx]
            
            # Примерные Pe из статьи (средние значения)
            target_pe_map = {
                'береза': [0.014, 0.020, 0.021],
                'дуб': [0.001, 0.003, 0.006],
                'клен': [0.016, 0.016, 0.023],
                'липа': [0.045, 0.040, 0.039],
                'осина': [0.011, 0.014, 0.022],
                'ель': [0.039, 0.053, 0.075],
                'сосна': [0.030, 0.029, 0.031]
            }
            
            target_pe = target_pe_map[tree][noise_idx]
            
            # Вычисляем разность с целью
            diff_pc = abs(our_pc - target_pc)
            total_diff += diff_pc
            count_comparisons += 1
            
            # Статус достижения цели
            status = "✅" if diff_pc < 0.03 else "⚠️" if diff_pc < 0.06 else "❌"
            
            row += f" {our_pc:.3f} {our_pe:.3f} |"
        
        print(row)
        
        # Показываем целевые значения
        target_row = f"(цель)   |"
        for noise_idx in range(3):
            target_pc = target_results[tree][noise_idx]
            target_pe = target_pe_map[tree][noise_idx]
            target_row += f" {target_pc:.3f} {target_pe:.3f} |"
        print(target_row)
        print("-" * 70)
    
    avg_diff = total_diff / count_comparisons
    print(f"\n📊 ОБЩАЯ ОЦЕНКА:")
    print(f"Средняя разность с целевыми Pc: {avg_diff:.3f}")
    
    if avg_diff < 0.03:
        print("🎉 ОТЛИЧНО! Результаты очень близки к статье!")
    elif avg_diff < 0.06:
        print("✅ ХОРОШО! Результаты приемлемо близки к статье.")
    else:
        print("⚠️ Нужна дополнительная оптимизация.")
    
    return results

def plot_final_comparison(results, tree_types):
    """Финальные графики сравнения"""
    target_results = {
        'береза': [0.944, 0.939, 0.919],
        'дуб': [0.783, 0.820, 0.827],
        'клен': [0.818, 0.821, 0.830],
        'липа': [0.931, 0.875, 0.791],
        'осина': [0.821, 0.751, 0.640],
        'ель': [0.914, 0.908, 0.881],
        'сосна': [0.854, 0.832, 0.792]
    }
    
    noise_levels = [0.01, 0.05, 0.1]
    
    plt.figure(figsize=(20, 15))
    
    # Индивидуальные графики по видам
    for i, tree in enumerate(tree_types):
        plt.subplot(3, 3, i + 1)
        
        # Целевые результаты
        target_vals = target_results[tree]
        plt.plot([n*100 for n in noise_levels], target_vals, 
                'ro-', label='Статья (цель)', linewidth=3, markersize=10)
        
        # Наши результаты
        our_vals = [results[noise]['class_accuracies'][i] for noise in noise_levels]
        plt.plot([n*100 for n in noise_levels], our_vals, 
                'bs-', label='Наши результаты', linewidth=3, markersize=10)
        
        plt.xlabel('Уровень шума (%)', fontsize=12)
        plt.ylabel('Точность (Pc)', fontsize=12)
        plt.title(f'{tree.upper()}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.6, 1.0)
    
    # Общий график производительности
    plt.subplot(3, 3, 8)
    
    # Средние значения по всем видам
    target_means = [np.mean([target_results[tree][i] for tree in tree_types]) 
                   for i in range(3)]
    our_means = [np.mean([results[noise]['class_accuracies']]) for noise in noise_levels]
    
    plt.plot([n*100 for n in noise_levels], target_means, 
            'ro-', label='Статья (среднее)', linewidth=4, markersize=12)
    plt.plot([n*100 for n in noise_levels], our_means, 
            'bs-', label='Наши результаты (среднее)', linewidth=4, markersize=12)
    
    plt.xlabel('Уровень шума (%)', fontsize=12)
    plt.ylabel('Средняя точность', fontsize=12)
    plt.title('ОБЩЕЕ СРАВНЕНИЕ', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Таблица разностей
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Создаем таблицу разностей
    differences = []
    for tree in tree_types:
        row = []
        for i, noise in enumerate(noise_levels):
            our_val = results[noise]['class_accuracies'][tree_types.index(tree)]
            target_val = target_results[tree][i]
            diff = abs(our_val - target_val)
            row.append(f'{diff:.3f}')
        differences.append(row)
    
    table = plt.table(cellText=differences,
                     rowLabels=tree_types,
                     colLabels=['δ=1%', 'δ=5%', 'δ=10%'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Разности с целевыми значениями', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('optimized_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Оптимизированная главная функция для достижения целевых результатов"""
    print("🎯 ОПТИМИЗИРОВАННАЯ КЛАССИФИКАЦИЯ РАСТИТЕЛЬНОСТИ")
    print("=" * 70)
    print("🎯 ЦЕЛЬ: ДОСТИЧЬ РЕЗУЛЬТАТОВ ИЗ НАУЧНОЙ СТАТЬИ")
    print("=" * 70)
    
    # Загрузка с улучшенной предобработкой
    spectra, labels, tree_types = load_spectral_data_enhanced()
    
    if len(spectra) == 0:
        print("❌ Не удалось загрузить данные!")
        return
    
    # Предобработка
    lengths = [len(s) for s in spectra]
    target_length = min(lengths)
    X = np.array([spectrum[:target_length] for spectrum in spectra], dtype=np.float32)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"📊 Финальная форма данных: {X.shape}")
    
    # Стратифицированное разделение
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp
    )
    
    print(f"📏 Обучающая: {X_train.shape}, Валидационная: {X_val.shape}, Тестовая: {X_test.shape}")
    
    # Усиленная аугментация
    X_train_aug, y_train_aug = enhanced_data_augmentation(X_train, y_train, augment_factor=2)
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Создание и обучение оптимизированного ансамбля
    models = create_optimized_model_ensemble()
    best_model, all_models = train_optimized_ensemble(models, X_train_scaled, y_train_aug, X_val_scaled, y_val)
    
    # Финальное тестирование с прямым сравнением с статьей
    results = test_with_target_comparison(best_model, X_test_scaled, y_test, tree_types, n_realizations=1000)
    
    # Построение финальных графиков
    plot_final_comparison(results, tree_types)
    
    print("\n" + "="*70)
    print("✅ ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
    print("🎯 Результаты напрямую сравнены с целевыми из статьи")
    print("📊 Финальные графики: optimized_final_comparison.png")
    print("🚀 При необходимости можно дополнительно увеличить эпохи/батчи")
    print("="*70)

if __name__ == "__main__":
    main() 