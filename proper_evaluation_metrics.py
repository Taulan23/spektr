#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛЬНЫЕ МЕТРИКИ ОЦЕНКИ ДЛЯ РЕШЕНИЯ ПРОБЛЕМЫ ВЫСОКИХ РЕЗУЛЬТАТОВ
==================================================================

Этот код реализует правильную систему оценки моделей машинного обучения
для решения проблемы подозрительно высоких результатов.

ПРОБЛЕМА: Стандартная accuracy дает misleading результаты при дисбалансе классов
РЕШЕНИЕ: Комплексная система метрик, учитывающая реальные условия применения
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class ProperEvaluationMetrics:
    """Класс для правильной оценки моделей с учетом дисбаланса классов"""

    def __init__(self, model, X, y, class_names=None):
        self.model = model
        self.X = X
        self.y = y
        self.class_names = class_names if class_names else [f"Class_{i}" for i in range(len(np.unique(y)))]
        self.n_classes = len(np.unique(y))

    def calculate_all_metrics(self, y_true, y_pred, y_proba=None):
        """Вычисляет все важные метрики для правильной оценки"""

        metrics = {}

        # 1. ОСНОВНЫЕ МЕТРИКИ КЛАССИФИКАЦИИ
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        # 2. МЕТРИКИ НА ОСНОВЕ PRECISION/RECALL
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)

        metrics['precision_weighted'] = np.average(precision, weights=support)
        metrics['recall_weighted'] = np.average(recall, weights=support)
        metrics['f1_weighted'] = np.average(f1, weights=support)

        # 3. МЕТРИКИ ДЛЯ КАЖДОГО КЛАССА
        metrics['per_class_precision'] = precision
        metrics['per_class_recall'] = recall
        metrics['per_class_f1'] = f1
        metrics['per_class_support'] = support

        # 4. AUC МЕТРИКИ (если есть вероятности)
        if y_proba is not None:
            try:
                if self.n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    # Для мультиклассовой классификации
                    y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
                    if y_true_bin.shape[1] == 1:  # Только один класс в y_true
                        metrics['roc_auc_macro'] = np.nan
                        metrics['roc_auc_weighted'] = np.nan
                    else:
                        metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                        metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')
            except Exception as e:
                print(f"⚠️ Не удалось вычислить AUC метрики: {e}")

        return metrics

    def cross_validate_with_proper_metrics(self, cv_folds=5):
        """Выполняет кросс-валидацию с правильными метриками"""

        print("🔄 КРОСС-ВАЛИДАЦИЯ С ПРАВИЛЬНЫМИ МЕТРИКАМИ")
        print("=" * 60)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Контейнеры для метрик
        cv_metrics = {
            'accuracy': [],
            'balanced_accuracy': [],
            'f1_macro': [],
            'cohen_kappa': [],
            'matthews_corrcoef': []
        }

        per_class_metrics = {
            'precision': [],
            'recall': [],
            'f1': []
        }

        for fold, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
            print(f"   Fold {fold + 1}/{cv_folds}...", end=" ")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Обучение модели
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)

            # Получение вероятностей (если возможно)
            try:
                y_proba = self.model.predict_proba(X_val)
            except:
                y_proba = None

            # Вычисление метрик
            fold_metrics = self.calculate_all_metrics(y_val, y_pred, y_proba)

            # Сохранение основных метрик
            for metric in cv_metrics.keys():
                if metric in fold_metrics:
                    cv_metrics[metric].append(fold_metrics[metric])

            # Сохранение per-class метрик
            per_class_metrics['precision'].append(fold_metrics['per_class_precision'])
            per_class_metrics['recall'].append(fold_metrics['per_class_recall'])
            per_class_metrics['f1'].append(fold_metrics['per_class_f1'])

            print("✓")

        # Усреднение результатов
        print("\n📊 РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ:")
        print("-" * 60)

        results_summary = {}
        for metric, values in cv_metrics.items():
            if values:  # Проверяем, что список не пустой
                mean_val = np.mean(values)
                std_val = np.std(values)
                results_summary[metric] = {'mean': mean_val, 'std': std_val}
                print(f"   {metric:20s}: {mean_val:.3f} ± {std_val:.3f}")

        # Per-class результаты
        print(f"\n📋 РЕЗУЛЬТАТЫ ПО КЛАССАМ:")
        print("-" * 60)

        per_class_summary = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(per_class_metrics['precision'][0]):  # Проверяем наличие класса
                class_precision = [fold_prec[i] for fold_prec in per_class_metrics['precision']]
                class_recall = [fold_rec[i] for fold_rec in per_class_metrics['recall']]
                class_f1 = [fold_f1[i] for fold_f1 in per_class_metrics['f1']]

                per_class_summary[class_name] = {
                    'precision': {'mean': np.mean(class_precision), 'std': np.std(class_precision)},
                    'recall': {'mean': np.mean(class_recall), 'std': np.std(class_recall)},
                    'f1': {'mean': np.mean(class_f1), 'std': np.std(class_f1)}
                }

                print(f"   {class_name:15s}:")
                print(f"     Precision: {np.mean(class_precision):.3f} ± {np.std(class_precision):.3f}")
                print(f"     Recall:    {np.mean(class_recall):.3f} ± {np.std(class_recall):.3f}")
                print(f"     F1-score:  {np.mean(class_f1):.3f} ± {np.std(class_f1):.3f}")

        return results_summary, per_class_summary

    def analyze_class_imbalance_impact(self):
        """Анализирует влияние дисбаланса классов на метрики"""

        print("\n🎯 АНАЛИЗ ВЛИЯНИЯ ДИСБАЛАНСА КЛАССОВ")
        print("=" * 60)

        # Подсчет образцов по классам
        unique, counts = np.unique(self.y, return_counts=True)
        class_distribution = dict(zip(unique, counts))

        print("📊 Распределение классов:")
        total_samples = len(self.y)
        imbalance_ratios = []

        for i, (class_idx, count) in enumerate(class_distribution.items()):
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
            percentage = (count / total_samples) * 100
            print(f"   {class_name:20s}: {count:4d} образцов ({percentage:5.1f}%)")
            imbalance_ratios.append(count)

        # Вычисление коэффициента дисбаланса
        max_count = max(imbalance_ratios)
        min_count = min(imbalance_ratios)
        imbalance_ratio = max_count / min_count

        print(f"\n📈 Коэффициент дисбаланса: {imbalance_ratio:.1f}:1")

        # Интерпретация
        if imbalance_ratio < 1.5:
            balance_status = "✅ ОТЛИЧНО СБАЛАНСИРОВАНЫ"
            expected_accuracy_drop = "0-5%"
        elif imbalance_ratio < 3:
            balance_status = "🟡 УМЕРЕННО СБАЛАНСИРОВАНЫ"
            expected_accuracy_drop = "5-15%"
        elif imbalance_ratio < 10:
            balance_status = "🟠 УМЕРЕННО НЕСБАЛАНСИРОВАНЫ"
            expected_accuracy_drop = "15-30%"
        else:
            balance_status = "🔴 СИЛЬНО НЕСБАЛАНСИРОВАНЫ"
            expected_accuracy_drop = "30-50%"

        print(f"   Статус: {balance_status}")
        print(f"   Ожидаемое падение accuracy в реальных условиях: {expected_accuracy_drop}")

        return imbalance_ratio, class_distribution

    def create_comprehensive_report(self):
        """Создает комплексный отчет с рекомендациями"""

        print("\n📋 КОМПЛЕКСНЫЙ ОТЧЕТ: ПРАВИЛЬНАЯ ОЦЕНКА МОДЕЛИ")
        print("=" * 80)

        # 1. Анализ дисбаланса
        imbalance_ratio, class_dist = self.analyze_class_imbalance_impact()

        # 2. Кросс-валидация
        cv_results, per_class_results = self.cross_validate_with_proper_metrics()

        # 3. Рекомендации по метрикам
        print(f"\n💡 РЕКОМЕНДАЦИИ ПО ВЫБОРУ МЕТРИК:")
        print("-" * 60)

        if imbalance_ratio < 2:
            print("✅ Для ваших сбалансированных данных:")
            print("   • Accuracy - подходящая метрика")
            print("   • Balanced Accuracy - дополнительная проверка")
            print("   • F1-macro - для научных публикаций")
            print("   • Per-class metrics - для детального анализа")
        else:
            print("⚠️ Для несбалансированных данных используйте:")
            print("   • Balanced Accuracy (ОСНОВНАЯ метрика)")
            print("   • F1-macro (для сравнения моделей)")
            print("   • Cohen's Kappa (статистическая значимость)")
            print("   • Per-class Precision/Recall (для каждого класса)")
            print("   • ❌ НЕ используйте простую Accuracy!")

        # 4. Рекомендации по улучшению
        print(f"\n🚀 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ МОДЕЛИ:")
        print("-" * 60)

        if imbalance_ratio > 5:
            print("🔧 Для несбалансированных данных:")
            print("   • Используйте class_weight='balanced'")
            print("   • Примените техники семплирования (SMOTE, ADASYN)")
            print("   • Рассмотрите cost-sensitive learning")
            print("   • Используйте ансамблевые методы")

        # 5. Предупреждения
        print(f"\n⚠️ ВАЖНЫЕ ПРЕДУПРЕЖДЕНИЯ:")
        print("-" * 60)
        print("   • НЕ экстраполируйте результаты на другие условия")
        print("   • Всегда указывайте условия эксперимента")
        print("   • Планируйте валидацию на реальных данных")
        print("   • Используйте несколько метрик одновременно")

        return cv_results, per_class_results, imbalance_ratio

    def visualize_metrics_comparison(self, cv_results):
        """Создает визуализацию сравнения метрик"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # График 1: Сравнение основных метрик
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_macro', 'cohen_kappa']
        means = [cv_results[m]['mean'] for m in metrics_to_plot if m in cv_results]
        stds = [cv_results[m]['std'] for m in metrics_to_plot if m in cv_results]
        labels = [m.replace('_', ' ').title() for m in metrics_to_plot if m in cv_results]

        bars = ax1.bar(labels, means, yerr=stds, capsize=5, alpha=0.7,
                      color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'])

        ax1.set_title('Сравнение метрик оценки', fontsize=14, weight='bold')
        ax1.set_ylabel('Значение метрики')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

        # График 2: Accuracy vs Balanced Accuracy
        if 'accuracy' in cv_results and 'balanced_accuracy' in cv_results:
            acc_mean = cv_results['accuracy']['mean']
            bal_acc_mean = cv_results['balanced_accuracy']['mean']

            comparison_data = {
                'Accuracy\n(может обманывать)': acc_mean,
                'Balanced Accuracy\n(честная оценка)': bal_acc_mean
            }

            bars2 = ax2.bar(comparison_data.keys(), comparison_data.values(),
                           color=['lightcoral', 'lightgreen'], alpha=0.8)

            ax2.set_title('Accuracy vs Balanced Accuracy', fontsize=14, weight='bold')
            ax2.set_ylabel('Значение метрики')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)

            # Добавляем разность
            diff = abs(acc_mean - bal_acc_mean)
            ax2.text(0.5, max(acc_mean, bal_acc_mean) + 0.05,
                    f'Разность: {diff:.3f}', ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

            # Добавляем значения на столбцы
            for bar, value in zip(bars2, comparison_data.values()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

        plt.tight_layout()
        plt.savefig('proper_evaluation_metrics.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Сохранен график: proper_evaluation_metrics.png")

        return fig

    def generate_production_ready_evaluation(self, test_size=0.2):
        """Генерирует production-ready систему оценки"""

        print("\n🚀 PRODUCTION-READY СИСТЕМА ОЦЕНКИ")
        print("=" * 80)

        from sklearn.model_selection import train_test_split

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )

        # Обучение модели
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        try:
            y_proba = self.model.predict_proba(X_test)
        except:
            y_proba = None

        # Вычисление финальных метрик
        final_metrics = self.calculate_all_metrics(y_test, y_pred, y_proba)

        print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
        print("-" * 60)

        # Основные метрики
        key_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'cohen_kappa']
        for metric in key_metrics:
            if metric in final_metrics:
                print(f"   {metric.replace('_', ' ').title():20s}: {final_metrics[metric]:.3f}")

        # Детальный отчет по классам
        print(f"\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
        print("-" * 60)

        report = classification_report(y_test, y_pred, target_names=self.class_names,
                                     output_dict=True, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            if class_name in report:
                class_metrics = report[class_name]
                print(f"   {class_name:15s}:")
                print(f"     Precision: {class_metrics['precision']:.3f}")
                print(f"     Recall:    {class_metrics['recall']:.3f}")
                print(f"     F1-score:  {class_metrics['f1-score']:.3f}")
                print(f"     Support:   {class_metrics['support']}")

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n📊 МАТРИЦА ОШИБОК:")
        print("-" * 60)
        print("Строки = истинные классы, Столбцы = предсказанные классы")

        # Создание DataFrame для красивого вывода
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        print(cm_df)

        # Рекомендации на основе результатов
        self._provide_final_recommendations(final_metrics)

        return final_metrics, report, cm

    def _provide_final_recommendations(self, metrics):
        """Предоставляет финальные рекомендации на основе результатов"""

        print(f"\n💡 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ:")
        print("=" * 60)

        accuracy = metrics.get('accuracy', 0)
        balanced_accuracy = metrics.get('balanced_accuracy', 0)
        f1_macro = metrics.get('f1_macro', 0)

        accuracy_gap = abs(accuracy - balanced_accuracy)

        if accuracy_gap < 0.05:
            print("✅ ОТЛИЧНЫЕ РЕЗУЛЬТАТЫ:")
            print("   • Минимальная разница между Accuracy и Balanced Accuracy")
            print("   • Модель хорошо работает на всех классах")
            print("   • Результаты надежны для данных условий")
        elif accuracy_gap < 0.15:
            print("🟡 ХОРОШИЕ РЕЗУЛЬТАТЫ С ОГОВОРКАМИ:")
            print("   • Умеренная разница между метриками")
            print("   • Некоторые классы могут классифицироваться хуже")
            print("   • Рекомендуется дополнительная оптимизация")
        else:
            print("🔴 ТРЕБУЕТСЯ УЛУЧШЕНИЕ:")
            print("   • Значительная разница между Accuracy и Balanced Accuracy")
            print("   • Модель плохо работает с редкими классами")
            print("   • Необходимо применить техники балансировки")

        print(f"\n📝 ДЛЯ ПУБЛИКАЦИИ ИСПОЛЬЗУЙТЕ:")
        print("   • Основная метрика: Balanced Accuracy")
        print("   • Дополнительно: F1-macro, Cohen's Kappa")
        print("   • Обязательно: Per-class Precision/Recall")
        print("   • Контекст: условия эксперимента и ограничения")


def main():
    """Демонстрация использования правильных метрик оценки"""

    print("📊 ДЕМОНСТРАЦИЯ ПРАВИЛЬНОЙ СИСТЕМЫ ОЦЕНКИ МОДЕЛЕЙ")
    print("=" * 80)
    print("Цель: Показать, как правильно оценивать модели и избегать misleading результатов")
    print("=" * 80)

    # Создаем демонстрационные данные
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier

    # Создаем несбалансированный датасет
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=5,
        n_informative=15,
        n_redundant=5,
        weights=[0.5, 0.2, 0.15, 0.1, 0.05],  # Несбалансированные классы
        random_state=42
    )

    class_names = ['Сосна', 'Береза', 'Ель', 'Дуб', 'Липа']

    # Создаем модель
    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Важно для несбалансированных данных!
    )

    # Создаем систему оценки
    evaluator = ProperEvaluationMetrics(model, X, y, class_names)

    # Выполняем комплексную оценку
    cv_results, per_class_results, imbalance_ratio = evaluator.create_comprehensive_report()

    # Создаем визуализацию
    evaluator.visualize_metrics_comparison(cv_results)

    # Production-ready оценка
    final_metrics, report, cm = evaluator.generate_production_ready_evaluation()

    print("\n" + "=" * 80)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("📊 Теперь вы знаете, как правильно оценивать модели")
    print("🎯 Используйте эти принципы для честной оценки ваших результатов")
    print("=" * 80)


if __name__ == "__main__":
    main()
