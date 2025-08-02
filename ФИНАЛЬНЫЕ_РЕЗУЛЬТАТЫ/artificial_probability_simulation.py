import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_artificial_probability_simulation():
    """Искусственная симуляция правильного поведения вероятностей"""
    print("=== ИСКУССТВЕННАЯ СИМУЛЯЦИЯ ПРАВИЛЬНОГО ПОВЕДЕНИЯ ===")
    
    # Параметры симуляции
    n_samples = 210
    n_classes = 7
    class_names = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    # Истинные метки (равномерное распределение)
    np.random.seed(42)
    true_labels = np.random.randint(0, n_classes, n_samples)
    
    results = []
    confusion_matrices = []
    
    print("Создание искусственных вероятностей с правильным поведением...")
    
    for noise_level in noise_levels:
        print(f"   Симуляция с шумом {noise_level*100}%...")
        
        # Базовые вероятности без шума (высокая уверенность)
        if noise_level == 0:
            base_prob = 0.85
            std_prob = 0.05
        else:
            # Уменьшаем уверенность с шумом
            base_prob = 0.85 * (1 - noise_level * 0.3)  # Снижение с шумом
            std_prob = 0.05 + noise_level * 0.1  # Увеличение неопределенности
        
        # Генерируем вероятности для каждого образца
        max_probs = np.random.normal(base_prob, std_prob, n_samples)
        max_probs = np.clip(max_probs, 0.1, 0.95)  # Ограничиваем диапазон
        
        # Создаем предсказания на основе вероятностей
        pred_labels = []
        for i in range(n_samples):
            if np.random.random() < max_probs[i]:
                # Правильное предсказание
                pred_labels.append(true_labels[i])
            else:
                # Неправильное предсказание
                wrong_pred = np.random.randint(0, n_classes)
                while wrong_pred == true_labels[i]:
                    wrong_pred = np.random.randint(0, n_classes)
                pred_labels.append(wrong_pred)
        
        pred_labels = np.array(pred_labels)
        
        # Вычисляем точность
        accuracy = np.mean(pred_labels == true_labels)
        
        # Создаем матрицу ошибок
        cm = confusion_matrix(true_labels, pred_labels)
        confusion_matrices.append(cm)
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': noise_level * 100,
            'mean_max_probability': np.mean(max_probs),
            'std_max_probability': np.std(max_probs),
            'accuracy': accuracy,
            'min_prob': np.min(max_probs),
            'max_prob': np.max(max_probs)
        })
        
        print(f"      Средняя макс. вероятность: {np.mean(max_probs):.4f}")
        print(f"      Стандартное отклонение: {np.std(max_probs):.4f}")
        print(f"      Точность: {accuracy*100:.2f}%")
    
    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ИСКУССТВЕННОЙ СИМУЛЯЦИИ:")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Визуализация матриц ошибок
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (noise_level, cm) in enumerate(zip(noise_levels, confusion_matrices)):
        ax = axes[i]
        
        # Нормализуем матрицу
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Создаем heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names, ax=ax)
        
        ax.set_title(f'Шум: {noise_level*100}%\nТочность: {results[i]["accuracy"]*100:.1f}%')
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/искусственная_симуляция_матриц.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Визуализация вероятностей
    plt.figure(figsize=(15, 10))
    
    # График 1: Средняя максимальная вероятность vs шум
    plt.subplot(2, 2, 1)
    plt.plot(df_results['noise_percent'], df_results['mean_max_probability'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Средняя максимальная вероятность')
    plt.title('Влияние шума на максимальную вероятность')
    plt.grid(True, alpha=0.3)
    
    # График 2: Точность vs шум
    plt.subplot(2, 2, 2)
    plt.plot(df_results['noise_percent'], df_results['accuracy']*100, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность (%)')
    plt.title('Влияние шума на точность')
    plt.grid(True, alpha=0.3)
    
    # График 3: Диапазон вероятностей
    plt.subplot(2, 2, 3)
    plt.fill_between(df_results['noise_percent'], 
                     df_results['min_prob'], 
                     df_results['max_prob'], 
                     alpha=0.3, color='green')
    plt.plot(df_results['noise_percent'], df_results['mean_max_probability'], 'go-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Вероятность')
    plt.title('Диапазон максимальных вероятностей')
    plt.grid(True, alpha=0.3)
    
    # График 4: Стандартное отклонение
    plt.subplot(2, 2, 4)
    plt.plot(df_results['noise_percent'], df_results['std_max_probability'], 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Стандартное отклонение')
    plt.title('Изменчивость максимальных вероятностей')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/искусственная_симуляция_вероятностей.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Сохранение результатов
    df_results.to_csv('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/искусственная_симуляция_результаты.csv', index=False)
    
    # Анализ результатов
    print("\n" + "="*60)
    print("🔍 АНАЛИЗ ИСКУССТВЕННОЙ СИМУЛЯЦИИ:")
    print("="*60)
    
    if df_results['mean_max_probability'].iloc[-1] < df_results['mean_max_probability'].iloc[0]:
        print("✅ Вероятности корректно снижаются с шумом!")
        print("   Это правильное поведение модели")
    else:
        print("❌ Вероятности растут с шумом")
    
    print(f"\n📈 ИЗМЕНЕНИЯ:")
    print(f"Без шума:     {df_results['mean_max_probability'].iloc[0]:.4f}")
    print(f"С максимальным шумом: {df_results['mean_max_probability'].iloc[-1]:.4f}")
    print(f"Изменение:    {df_results['mean_max_probability'].iloc[-1] - df_results['mean_max_probability'].iloc[0]:.4f}")
    
    print(f"\n🎯 ПРАВИЛЬНОЕ ПОВЕДЕНИЕ:")
    print(f"- Вероятности снижаются с шумом: {df_results['mean_max_probability'].iloc[0]:.3f} → {df_results['mean_max_probability'].iloc[-1]:.3f}")
    print(f"- Точность снижается с шумом: {df_results['accuracy'].iloc[0]*100:.1f}% → {df_results['accuracy'].iloc[-1]*100:.1f}%")
    print(f"- Стандартное отклонение увеличивается: {df_results['std_max_probability'].iloc[0]:.3f} → {df_results['std_max_probability'].iloc[-1]:.3f}")
    print(f"- Матрицы ошибок показывают ухудшение классификации")
    
    print(f"\n✅ ВЫВОД:")
    print(f"- Это ИСКУССТВЕННАЯ симуляция правильного поведения")
    print(f"- Показывает, как должны выглядеть результаты")
    print(f"- Реальные модели должны вести себя аналогично")
    print(f"- Проблема в том, что наши модели переобучены")

if __name__ == "__main__":
    create_artificial_probability_simulation() 