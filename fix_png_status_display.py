#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕНИЕ ОТОБРАЖЕНИЯ СТАТУСА В PNG МАТРИЦАХ

Проблема: Правильные данные (0.903→0.509) показываются как "❌ ПРОБЛЕМА"
Решение: Исправить логику проверки статуса
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Русские шрифты для matplotlib
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

def load_spring_data_7_species():
    """Загрузка данных 7 видов (весенний период)"""
    base_path = "Исходные_данные/Спектры, весенний период, 7 видов"
    
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_data = []
    all_labels = []
    
    print("📂 Загрузка данных:")
    for i, tree_type in enumerate(tree_types):
        tree_path = os.path.join(base_path, tree_type)
        if not os.path.exists(tree_path):
            continue
            
        files = [f for f in os.listdir(tree_path) if f.endswith('.xlsx')]
        print(f"   {tree_type}: {len(files)} файлов")
        
        for file in files:
            try:
                file_path = os.path.join(tree_path, file)
                df = pd.read_excel(file_path)
                
                if df.shape[1] >= 2:
                    spectrum = df.iloc[:, 1].values
                    spectrum = spectrum[~np.isnan(spectrum)]
                    
                    if len(spectrum) > 100:
                        all_data.append(spectrum)
                        all_labels.append(i)
            except Exception as e:
                continue
    
    print(f"✅ Загружено: {len(all_data)} спектров, {len(set(all_labels))} классов")
    return np.array(all_data), np.array(all_labels), tree_types

def preprocess_data_properly(X, y):
    """ПРАВИЛЬНАЯ предобработка данных"""
    print("🔧 Предобработка данных:")
    
    min_length = min(len(spectrum) for spectrum in X)
    print(f"   Минимальная длина спектра: {min_length}")
    
    X_processed = np.array([spectrum[:min_length] for spectrum in X])
    scaler = RobustScaler()
    X_processed = scaler.fit_transform(X_processed)
    
    print(f"   Форма данных: {X_processed.shape}")
    print(f"   Стандартное отклонение данных: {X_processed.std():.6f}")
    
    return X_processed, y, scaler

def add_gaussian_noise(X, noise_level):
    """Добавление гауссовского шума к данным"""
    if noise_level == 0:
        return X
    
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def create_robust_classifier():
    """Создание устойчивого классификатора с правильным поведением вероятностей"""
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

def evaluate_with_noise_levels(model, X_test, y_test, noise_levels, tree_types):
    """Оценка модели с разными уровнями шума"""
    results = []
    
    print("\n🧪 Тестирование с разными уровнями шума:")
    
    for noise_level in noise_levels:
        X_test_noisy = add_gaussian_noise(X_test, noise_level)
        predictions = model.predict(X_test_noisy)
        probabilities = model.predict_proba(X_test_noisy)
        
        accuracy = accuracy_score(y_test, predictions)
        max_probabilities = np.max(probabilities, axis=1)
        mean_max_prob = np.mean(max_probabilities)
        std_max_prob = np.std(max_probabilities)
        
        cm = confusion_matrix(y_test, predictions)
        unique_probs = len(np.unique(np.round(max_probabilities, 4)))
        uniqueness_ratio = unique_probs / len(max_probabilities) * 100
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': noise_level * 100,
            'accuracy': accuracy,
            'mean_max_probability': mean_max_prob,
            'std_max_probability': std_max_prob,
            'unique_probs': unique_probs,
            'total_samples': len(max_probabilities),
            'uniqueness_ratio': uniqueness_ratio,
            'min_prob': np.min(max_probabilities),
            'max_prob': np.max(max_probabilities),
            'confusion_matrix': cm
        })
        
        print(f"   Шум {noise_level*100:3.0f}%: точность={accuracy:.3f}, "
              f"средняя_макс_вероятность={mean_max_prob:.3f}, "
              f"уникальность={uniqueness_ratio:.1f}%")
    
    return results

def create_corrected_confusion_matrices_png(results, tree_types, save_path="confusion_matrices_CORRECTED.png"):
    """
    Создание PNG с ИСПРАВЛЕННЫМ отображением статуса
    
    ИСПРАВЛЕНИЕ: Правильная логика определения статуса
    """
    n_matrices = len(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Матрицы ошибок: ПРАВИЛЬНОЕ отображение статуса (7 видов)', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    # ИСПРАВЛЕННАЯ ЛОГИКА: Проверка падения вероятностей
    baseline_prob = results[0]['mean_max_probability']  # Базовая вероятность без шума
    prob_threshold = baseline_prob * 0.8  # Порог снижения (20% от базовой)
    
    for i, result in enumerate(results):
        if i >= 6:
            break
            
        ax = axes_flat[i]
        cm = result['confusion_matrix']
        noise_level = result['noise_percent']
        accuracy = result['accuracy']
        mean_prob = result['mean_max_probability']
        
        # Нормализация матрицы для вероятностей
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Тепловая карта
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   ax=ax,
                   xticklabels=tree_types,
                   yticklabels=tree_types,
                   cbar_kws={'shrink': 0.8})
        
        # ИСПРАВЛЕННАЯ ЛОГИКА СТАТУСА
        if i == 0:
            # Для нулевого шума - всегда зеленый (базовая линия)
            color = 'green'
            status = '✅ БАЗОВАЯ ЛИНИЯ'
        elif mean_prob <= baseline_prob:
            # Если вероятность падает или равна базовой - хорошо
            color = 'green'
            status = '✅ ИСПРАВЛЕНО'
        elif mean_prob > baseline_prob * 1.1:
            # Если вероятность растет больше чем на 10% - проблема
            color = 'red'
            status = '❌ ПРОБЛЕМА'
        else:
            # Небольшие колебания - нейтрально
            color = 'orange'
            status = '⚠️ СТАБИЛЬНО'
            
        ax.set_title(f'Шум: {noise_level:.0f}% | Точность: {accuracy:.1%}\n'
                    f'Средняя вероятность: {mean_prob:.3f} {status}',
                    fontsize=12, color=color, fontweight='bold')
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 ИСПРАВЛЕННЫЕ PNG матрицы сохранены: {save_path}")
    plt.show()

def main():
    """Главная функция - создание PNG с правильным отображением статуса"""
    print("🎯 ИСПРАВЛЕНИЕ ОТОБРАЖЕНИЯ СТАТУСА В PNG МАТРИЦАХ\n")
    print("=" * 60)
    
    # 1. Загрузка данных
    X, y, tree_types = load_spring_data_7_species()
    
    # 2. Предобработка
    X_processed, y_processed, scaler = preprocess_data_properly(X, y)
    
    # 3. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    
    print(f"\n📊 Разделение данных:")
    print(f"   Обучение: {X_train.shape[0]} образцов")
    print(f"   Тест: {X_test.shape[0]} образцов")
    
    # 4. Создание и обучение модели
    print("\n🤖 Создание робастного классификатора...")
    model = create_robust_classifier()
    
    print("🎯 Обучение модели...")
    model.fit(X_train, y_train)
    print(f"   OOB Score: {model.oob_score_:.3f}")
    
    # 5. Тестирование с шумом
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    results = evaluate_with_noise_levels(model, X_test, y_test, noise_levels, tree_types)
    
    # 6. Создание ИСПРАВЛЕННЫХ PNG файлов
    print("\n🎨 Создание PNG с правильным отображением статуса...")
    create_corrected_confusion_matrices_png(results, tree_types, "confusion_matrices_STATUS_FIXED.png")
    
    # 7. Проверка исправления
    print("\n" + "="*60)
    print("🔍 ПРОВЕРКА ИСПРАВЛЕНИЯ СТАТУСА:")
    print("="*60)
    
    baseline_prob = results[0]['mean_max_probability']
    final_prob = results[-1]['mean_max_probability']
    
    print(f"\n📊 ВЕРОЯТНОСТИ:")
    for result in results:
        noise = result['noise_percent']
        prob = result['mean_max_probability']
        change = prob - baseline_prob
        print(f"   Шум {noise:3.0f}%: {prob:.3f} (изменение: {change:+.3f})")
    
    if final_prob < baseline_prob:
        print(f"\n✅ РЕЗУЛЬТАТ: Вероятности ПАДАЮТ ({baseline_prob:.3f} → {final_prob:.3f})")
        print("✅ СТАТУС: Теперь будет отображаться зеленым цветом!")
    else:
        print(f"\n❌ ПРОБЛЕМА: Вероятности НЕ падают ({baseline_prob:.3f} → {final_prob:.3f})")
    
    print(f"\n📁 Создан файл: confusion_matrices_STATUS_FIXED.png")

if __name__ == "__main__":
    main()