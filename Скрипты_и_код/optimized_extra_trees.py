#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОПТИМИЗИРОВАННЫЙ ExtraTreesClassifier ДЛЯ УСТОЙЧИВОСТИ К ШУМУ
==============================================================

Скрипт содержит оптимизированные конфигурации ExtraTreesClassifier
для повышения устойчивости к шуму на основе анализа результатов.

Дата: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class OptimizedExtraTreesConfig:
    """Класс с оптимизированными конфигурациями ExtraTreesClassifier"""

    @staticmethod
    def get_baseline_config():
        """Базовая конфигурация (текущая в системе)"""
        return {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }

    @staticmethod
    def get_noise_resistant_config():
        """Конфигурация, устойчивая к шуму"""
        return {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': False,
            'class_weight': 'balanced',
            'verbose': 0
        }

    @staticmethod
    def get_high_stability_config():
        """Конфигурация для максимальной стабильности"""
        return {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': False,
            'class_weight': 'balanced',
            'verbose': 0
        }

    @staticmethod
    def get_balanced_config():
        """Сбалансированная конфигурация (компромисс скорости и качества)"""
        return {
            'n_estimators': 250,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': False,
            'class_weight': 'balanced',
            'verbose': 0
        }

class NoiseResistantPipeline:
    """Пайплайн для устойчивого к шуму машинного обучения"""

    def __init__(self, config_type='noise_resistant'):
        """
        Инициализация пайплайна

        Args:
            config_type: тип конфигурации ('baseline', 'noise_resistant',
                        'high_stability', 'balanced')
        """
        self.config_type = config_type
        self.model = None
        self.scaler = None
        self.selector = None
        self.is_fitted = False

        # Получаем конфигурацию
        configs = {
            'baseline': OptimizedExtraTreesConfig.get_baseline_config(),
            'noise_resistant': OptimizedExtraTreesConfig.get_noise_resistant_config(),
            'high_stability': OptimizedExtraTreesConfig.get_high_stability_config(),
            'balanced': OptimizedExtraTreesConfig.get_balanced_config()
        }

        self.config = configs.get(config_type, configs['noise_resistant'])

    def create_model(self):
        """Создает модель с выбранной конфигурацией"""
        self.model = ExtraTreesClassifier(**self.config)
        return self.model

    def create_robust_scaler(self):
        """Создает устойчивый к выбросам скейлер"""
        self.scaler = RobustScaler()
        return self.scaler

    def create_feature_selector(self, k='auto'):
        """
        Создает селектор признаков

        Args:
            k: количество признаков ('auto' для 80% от общего количества)
        """
        self.selector = SelectKBest(f_classif, k=k)
        return self.selector

    def fit(self, X, y, use_feature_selection=True, feature_ratio=0.8):
        """
        Обучает полный пайплайн

        Args:
            X: признаки
            y: целевая переменная
            use_feature_selection: использовать ли отбор признаков
            feature_ratio: доля признаков для отбора
        """
        print(f"🚀 Обучение пайплайна с конфигурацией: {self.config_type}")

        # Масштабирование
        self.create_robust_scaler()
        X_scaled = self.scaler.fit_transform(X)

        # Отбор признаков
        if use_feature_selection:
            k = int(X.shape[1] * feature_ratio) if feature_ratio < 1 else 'all'
            self.create_feature_selector(k)
            X_scaled = self.selector.fit_transform(X_scaled, y)
            print(f"  📊 Отобрано признаков: {X_scaled.shape[1]} из {X.shape[1]}")

        # Обучение модели
        self.create_model()
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        print(f"  ✅ Модель обучена")

        return self

    def predict(self, X):
        """Предсказание"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Используйте fit() сначала.")

        X_scaled = self.scaler.transform(X)
        if self.selector:
            X_scaled = self.selector.transform(X_scaled)

        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Используйте fit() сначала.")

        X_scaled = self.scaler.transform(X)
        if self.selector:
            X_scaled = self.selector.transform(X_scaled)

        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        """Получает важность признаков"""
        if not self.is_fitted:
            raise ValueError("Модель не обучена.")

        return self.model.feature_importances_

class NoiseTestingFramework:
    """Фреймворк для тестирования устойчивости к шуму"""

    def __init__(self):
        self.results = {}

    def add_noise(self, X, noise_level):
        """
        Добавляет гауссов шум к данным

        Args:
            X: исходные данные
            noise_level: уровень шума (0.01 = 1%)
        """
        noise = np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
        return X + noise

    def test_configuration(self, X, y, config_type, noise_levels=[0.01, 0.05, 0.10],
                          n_iterations=10, test_size=0.3):
        """
        Тестирует конфигурацию при разных уровнях шума

        Args:
            X, y: данные
            config_type: тип конфигурации
            noise_levels: уровни шума для тестирования
            n_iterations: количество итераций
            test_size: размер тестовой выборки
        """
        print(f"\n🧪 Тестирование конфигурации: {config_type}")
        print("=" * 50)

        config_results = {}

        for noise_level in noise_levels:
            print(f"\n📊 Уровень шума: {noise_level*100:.1f}%")

            accuracies = []

            for i in range(n_iterations):
                # Добавляем шум
                X_noisy = self.add_noise(X, noise_level)

                # Разделяем данные
                X_train, X_test, y_train, y_test = train_test_split(
                    X_noisy, y, test_size=test_size, random_state=i, stratify=y
                )

                # Создаем и обучаем модель
                pipeline = NoiseResistantPipeline(config_type)
                pipeline.fit(X_train, y_train)

                # Предсказываем
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

                if (i + 1) % 5 == 0:
                    print(f"  Итерация {i+1}/{n_iterations}: {np.mean(accuracies[-5:]):.4f}")

            config_results[noise_level] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'values': accuracies
            }

            print(f"  Результат: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

        self.results[config_type] = config_results
        return config_results

    def compare_configurations(self, X, y, configs_to_test=None):
        """Сравнивает разные конфигурации"""

        if configs_to_test is None:
            configs_to_test = ['baseline', 'noise_resistant', 'high_stability', 'balanced']

        print("\n🏆 СРАВНЕНИЕ КОНФИГУРАЦИЙ")
        print("=" * 60)

        for config in configs_to_test:
            self.test_configuration(X, y, config)

        # Создаем сводную таблицу
        self.create_comparison_table()
        self.plot_comparison()

    def create_comparison_table(self):
        """Создает таблицу сравнения результатов"""

        print("\n📋 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 80)

        # Заголовок
        header = "Конфигурация".ljust(20)
        for noise_level in [0.01, 0.05, 0.10]:
            header += f"{noise_level*100:.0f}% шума".center(15)
        print(header)
        print("-" * 80)

        # Результаты
        for config, results in self.results.items():
            row = config.ljust(20)
            for noise_level in [0.01, 0.05, 0.10]:
                if noise_level in results:
                    mean_acc = results[noise_level]['mean']
                    std_acc = results[noise_level]['std']
                    row += f"{mean_acc:.3f}±{std_acc:.3f}".center(15)
                else:
                    row += "N/A".center(15)
            print(row)

    def plot_comparison(self):
        """Создает график сравнения"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # График 1: Средние значения
        noise_levels = [0.01, 0.05, 0.10]

        for config, results in self.results.items():
            means = [results[nl]['mean'] for nl in noise_levels if nl in results]
            stds = [results[nl]['std'] for nl in noise_levels if nl in results]

            ax1.errorbar([nl*100 for nl in noise_levels[:len(means)]], means,
                        yerr=stds, marker='o', label=config, linewidth=2, markersize=8)

        ax1.set_xlabel('Уровень шума (%)')
        ax1.set_ylabel('Точность')
        ax1.set_title('Сравнение конфигураций при разных уровнях шума')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Стабильность (стандартные отклонения)
        for config, results in self.results.items():
            stds = [results[nl]['std'] for nl in noise_levels if nl in results]

            ax2.plot([nl*100 for nl in noise_levels[:len(stds)]], stds,
                    marker='s', label=config, linewidth=2, markersize=8)

        ax2.set_xlabel('Уровень шума (%)')
        ax2.set_ylabel('Стандартное отклонение')
        ax2.set_title('Стабильность конфигураций')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('optimized_configurations_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_ensemble_model():
    """Создает ансамблевую модель для максимальной устойчивости"""

    # Разные конфигурации ExtraTreesClassifier
    et1 = ExtraTreesClassifier(**OptimizedExtraTreesConfig.get_noise_resistant_config())
    et2 = ExtraTreesClassifier(**OptimizedExtraTreesConfig.get_high_stability_config())

    # Дополнительные алгоритмы
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=8, random_state=42)
    svm = SVC(probability=True, random_state=42)

    # Ансамбль
    ensemble = VotingClassifier([
        ('extra_trees_1', et1),
        ('extra_trees_2', et2),
        ('gradient_boost', gb),
        ('svm', svm)
    ], voting='soft')

    return ensemble

def demonstrate_usage():
    """Демонстрирует использование оптимизированных конфигураций"""

    print("🎯 ДЕМОНСТРАЦИЯ ИСПОЛЬЗОВАНИЯ ОПТИМИЗИРОВАННЫХ КОНФИГУРАЦИЙ")
    print("=" * 70)

    # Генерируем тестовые данные
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=5,
        random_state=42
    )

    print(f"📊 Тестовые данные: {X.shape[0]} образцов, {X.shape[1]} признаков, {len(np.unique(y))} классов")

    # Базовая модель
    print("\n1️⃣ Базовая конфигурация:")
    baseline = NoiseResistantPipeline('baseline')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_score = accuracy_score(y_test, baseline.predict(X_test))
    print(f"   Точность: {baseline_score:.4f}")

    # Устойчивая к шуму модель
    print("\n2️⃣ Устойчивая к шуму конфигурация:")
    robust = NoiseResistantPipeline('noise_resistant')
    robust.fit(X_train, y_train)
    robust_score = accuracy_score(y_test, robust.predict(X_test))
    print(f"   Точность: {robust_score:.4f}")

    # Тестирование с шумом
    print("\n3️⃣ Тестирование с 10% шума:")
    framework = NoiseTestingFramework()
    X_noisy = framework.add_noise(X, 0.10)
    X_train_noisy, X_test_noisy, y_train, y_test = train_test_split(
        X_noisy, y, test_size=0.3, random_state=42
    )

    baseline_noisy = NoiseResistantPipeline('baseline')
    baseline_noisy.fit(X_train_noisy, y_train)
    baseline_noisy_score = accuracy_score(y_test, baseline_noisy.predict(X_test_noisy))

    robust_noisy = NoiseResistantPipeline('noise_resistant')
    robust_noisy.fit(X_train_noisy, y_train)
    robust_noisy_score = accuracy_score(y_test, robust_noisy.predict(X_test_noisy))

    print(f"   Базовая модель с шумом: {baseline_noisy_score:.4f}")
    print(f"   Устойчивая модель с шумом: {robust_noisy_score:.4f}")
    print(f"   Улучшение: {robust_noisy_score - baseline_noisy_score:.4f}")

if __name__ == "__main__":

    print("🌳 ОПТИМИЗИРОВАННЫЙ ExtraTreesClassifier")
    print("=" * 50)

    # Демонстрация использования
    demonstrate_usage()

    print("\n" + "=" * 50)
    print("✅ Демонстрация завершена!")
    print("\n📝 Для полного тестирования используйте:")
    print("   framework = NoiseTestingFramework()")
    print("   framework.compare_configurations(X, y)")
