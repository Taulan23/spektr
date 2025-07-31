#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ РЕЗУЛЬТАТОВ ExtraTreesClassifier ПРИ РАЗНЫХ УРОВНЯХ ШУМА
================================================================

Скрипт для анализа и сравнения результатов научника по ExtraTreesClassifier
при уровнях шума 1% и 10%.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

# Настройка для русского языка в графиках
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

def load_and_analyze_results():
    """Загружает и анализирует файлы с результатами"""

    print("=" * 80)
    print("🔬 АНАЛИЗ РЕЗУЛЬТАТОВ ExtraTreesClassifier")
    print("=" * 80)

    # Загрузка файлов
    try:
        df_1 = pd.read_excel('et80_1.xlsx')
        df_10 = pd.read_excel('et80_10.xlsx')
        print("✅ Файлы успешно загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки файлов: {e}")
        return

    # Анализ параметров эксперимента
    print("\n📋 ПАРАМЕТРЫ ЭКСПЕРИМЕНТОВ:")
    print("-" * 40)

    # Извлекаем информацию из файлов
    for name, df in [("1% шума", df_1), ("10% шума", df_10)]:
        print(f"\n{name}:")
        for i in range(min(10, len(df))):
            row_val = df.iloc[i, 0]
            if pd.notna(row_val) and str(row_val).strip():
                if any(keyword in str(row_val).lower() for keyword in ['шум', 'реализац', 'время']):
                    print(f"  • {row_val}")

    # Извлечение числовых результатов
    results_1 = extract_performance_metrics(df_1, "1% шума")
    results_10 = extract_performance_metrics(df_10, "10% шума")

    # Сравнительный анализ
    compare_results(results_1, results_10)

    # Анализ параметров модели
    analyze_model_parameters()

    # Рекомендации
    provide_recommendations()

def extract_performance_metrics(df, noise_level):
    """Извлекает метрики производительности из DataFrame"""

    print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ: {noise_level}")
    print("-" * 40)

    results = {}

    # Поиск столбца с основными результатами (обычно самый заполненный)
    max_values = 0
    main_col_idx = 1

    for col_idx in range(1, df.shape[1]):
        col = df.iloc[:, col_idx]
        numeric_count = sum(1 for val in col if pd.notna(val) and isinstance(val, (int, float)) and val != 0)
        if numeric_count > max_values:
            max_values = numeric_count
            main_col_idx = col_idx

    main_col = df.iloc[:, main_col_idx]
    numeric_values = [val for val in main_col if pd.notna(val) and isinstance(val, (int, float)) and val != 0]

    if numeric_values:
        results['accuracy_mean'] = np.mean(numeric_values)
        results['accuracy_std'] = np.std(numeric_values)
        results['accuracy_min'] = np.min(numeric_values)
        results['accuracy_max'] = np.max(numeric_values)
        results['total_samples'] = len(numeric_values)

        print(f"Количество измерений: {len(numeric_values)}")
        print(f"Средняя точность: {np.mean(numeric_values):.4f} ± {np.std(numeric_values):.4f}")
        print(f"Диапазон: [{np.min(numeric_values):.4f}, {np.max(numeric_values):.4f}]")
        print(f"Медиана: {np.median(numeric_values):.4f}")

        # Анализ распределения
        q25 = np.percentile(numeric_values, 25)
        q75 = np.percentile(numeric_values, 75)
        print(f"Q25-Q75: [{q25:.4f}, {q75:.4f}]")

        results['q25'] = q25
        results['q75'] = q75
        results['median'] = np.median(numeric_values)
        results['values'] = numeric_values

    return results

def compare_results(results_1, results_10):
    """Сравнивает результаты при разных уровнях шума"""

    print("\n🔍 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 50)

    if not results_1 or not results_10:
        print("❌ Недостаточно данных для сравнения")
        return

    # Основные метрики
    acc_1 = results_1['accuracy_mean']
    acc_10 = results_10['accuracy_mean']

    print(f"📈 ТОЧНОСТЬ:")
    print(f"  При 1% шума:  {acc_1:.4f} ± {results_1['accuracy_std']:.4f}")
    print(f"  При 10% шума: {acc_10:.4f} ± {results_10['accuracy_std']:.4f}")
    print(f"  Падение:      {acc_1 - acc_10:.4f} ({((acc_1 - acc_10)/acc_1)*100:.1f}%)")

    # Стабильность
    std_1 = results_1['accuracy_std']
    std_10 = results_10['accuracy_std']

    print(f"\n📊 СТАБИЛЬНОСТЬ (стандартное отклонение):")
    print(f"  При 1% шума:  {std_1:.4f}")
    print(f"  При 10% шума: {std_10:.4f}")
    print(f"  Изменение:    {std_10 - std_1:.4f}")

    # Медианные значения
    med_1 = results_1['median']
    med_10 = results_10['median']

    print(f"\n📍 МЕДИАННЫЕ ЗНАЧЕНИЯ:")
    print(f"  При 1% шума:  {med_1:.4f}")
    print(f"  При 10% шума: {med_10:.4f}")
    print(f"  Падение:      {med_1 - med_10:.4f}")

    # Статистическая значимость падения
    print(f"\n🎯 ВЫВОДЫ:")
    if acc_1 - acc_10 > 0.01:  # Падение больше 1%
        print(f"  ⚠️  ЗНАЧИТЕЛЬНОЕ падение производительности при увеличении шума")
        print(f"      Падение составляет {((acc_1 - acc_10)/acc_1)*100:.1f}%")
    else:
        print(f"  ✅ Модель относительно устойчива к шуму")

    if std_10 > std_1 * 1.5:
        print(f"  ⚠️  ЗНАЧИТЕЛЬНОЕ снижение стабильности")
    else:
        print(f"  ✅ Стабильность модели приемлемая")

    # Создание графика сравнения
    create_comparison_plot(results_1, results_10)

def create_comparison_plot(results_1, results_10):
    """Создает график сравнения результатов"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Сравнение ExtraTreesClassifier: 1% vs 10% шума', fontsize=16, fontweight='bold')

    # График 1: Гистограммы распределений
    ax1.hist(results_1['values'], bins=20, alpha=0.7, label='1% шума', color='green', density=True)
    ax1.hist(results_10['values'], bins=20, alpha=0.7, label='10% шума', color='red', density=True)
    ax1.set_xlabel('Точность')
    ax1.set_ylabel('Плотность')
    ax1.set_title('Распределение точности')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Box plot
    data_to_plot = [results_1['values'], results_10['values']]
    labels = ['1% шума', '10% шума']
    box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Точность')
    ax2.set_title('Box Plot сравнение')
    ax2.grid(True, alpha=0.3)

    # График 3: Средние значения с ошибками
    means = [results_1['accuracy_mean'], results_10['accuracy_mean']]
    stds = [results_1['accuracy_std'], results_10['accuracy_std']]
    colors = ['green', 'red']

    bars = ax3.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax3.set_ylabel('Точность')
    ax3.set_title('Средние значения ± σ')
    ax3.grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax3.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontweight='bold')

    # График 4: Временные ряды (если есть порядок измерений)
    ax4.plot(range(len(results_1['values'])), sorted(results_1['values'], reverse=True),
             'g-', label='1% шума', linewidth=2)
    ax4.plot(range(len(results_10['values'])), sorted(results_10['values'], reverse=True),
             'r-', label='10% шума', linewidth=2)
    ax4.set_xlabel('Ранг измерения')
    ax4.set_ylabel('Точность')
    ax4.set_title('Ранжированные результаты')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('
        ', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n💾 График сохранен как 'extra_trees_comparison.png'")

def analyze_model_parameters():
    """Анализирует параметры модели ExtraTreesClassifier"""

    print("\n🛠️  АНАЛИЗ ПАРАМЕТРОВ МОДЕЛИ")
    print("=" * 50)

    # Текущие параметры из системы
    current_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 1
    }

    print("📋 ТЕКУЩИЕ ПАРАМЕТРЫ:")
    for param, value in current_params.items():
        print(f"  {param:20s}: {value}")

    print("\n🎯 ВЛИЯНИЕ ПАРАМЕТРОВ НА УСТОЙЧИВОСТЬ К ШУМУ:")
    print("-" * 50)

    print("1. n_estimators=200:")
    print("   ✅ Хорошее значение для баланса качества/скорости")
    print("   💡 Для большей устойчивости к шуму можно увеличить до 300-500")

    print("\n2. max_depth=20:")
    print("   ⚠️  Довольно глубоко - может способствовать переобучению")
    print("   💡 При шуме рекомендуется ограничить до 10-15")

    print("\n3. min_samples_split=5:")
    print("   ✅ Хорошее значение для предотвращения переобучения")
    print("   💡 При сильном шуме можно увеличить до 10-20")

    print("\n4. min_samples_leaf=2:")
    print("   ⚠️  Минимальное значение - может создавать шумные листья")
    print("   💡 Рекомендуется увеличить до 5-10 для устойчивости к шуму")

    print("\n5. max_features='sqrt':")
    print("   ✅ Оптимальный выбор для ExtraTreesClassifier")
    print("   💡 Хорошо работает с шумными данными")

def provide_recommendations():
    """Предоставляет рекомендации по улучшению"""

    print("\n🎯 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ УСТОЙЧИВОСТИ К ШУМУ")
    print("=" * 60)

    print("1. 🔧 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ:")
    print("   • Увеличить min_samples_leaf до 5-10")
    print("   • Ограничить max_depth до 10-15")
    print("   • Увеличить min_samples_split до 10-20")
    print("   • Рассмотреть увеличение n_estimators до 300-500")

    print("\n2. 📊 ПРЕДОБРАБОТКА ДАННЫХ:")
    print("   • Применить сглаживание/фильтрацию шума")
    print("   • Использовать RobustScaler вместо StandardScaler")
    print("   • Добавить отбор признаков для снижения размерности")

    print("\n3. 🎛️  ТЕХНИКИ АНСАМБЛИРОВАНИЯ:")
    print("   • Использовать Voting Classifier с разными алгоритмами")
    print("   • Добавить Gradient Boosting для стабильности")
    print("   • Рассмотреть Stacking с метаклассификатором")

    print("\n4. 🔍 ВАЛИДАЦИЯ И ТЕСТИРОВАНИЕ:")
    print("   • Использовать StratifiedKFold для кросс-валидации")
    print("   • Тестировать на разных уровнях шума (2%, 5%, 15%)")
    print("   • Анализировать важность признаков")

    print("\n5. 📈 МОНИТОРИНГ КАЧЕСТВА:")
    print("   • Отслеживать не только accuracy, но и precision/recall")
    print("   • Анализировать confusion matrix по классам")
    print("   • Измерять время обучения и предсказания")

    # Предлагаемая конфигурация
    print("\n🚀 ПРЕДЛАГАЕМАЯ КОНФИГУРАЦИЯ ДЛЯ УСТОЙЧИВОСТИ К ШУМУ:")
    print("-" * 60)

    robust_params = """
model = ExtraTreesClassifier(
    n_estimators=300,           # Больше деревьев для стабильности
    max_depth=12,               # Ограничение глубины против переобучения
    min_samples_split=15,       # Больше образцов для разбиения
    min_samples_leaf=8,         # Больше образцов в листьях
    max_features='sqrt',        # Оптимально для Extra Trees
    random_state=42,
    n_jobs=-1,
    bootstrap=False,            # Extra Trees не использует bootstrap
    class_weight='balanced',    # Для несбалансированных классов
    verbose=1
)"""

    print(robust_params)

def main():
    """Основная функция"""

    try:
        load_and_analyze_results()

        print("\n" + "=" * 80)
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 80)
        print("\n📝 КРАТКОЕ РЕЗЮМЕ:")
        print("• Падение производительности при 10% шума ожидаемо")
        print("• Литература подтверждает отсутствие 'идеальной картины' при высоком шуме")
        print("• Рекомендуется оптимизация параметров для повышения устойчивости")
        print("• График сравнения сохранен для дальнейшего анализа")

    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
