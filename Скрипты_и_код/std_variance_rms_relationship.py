#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СВЯЗЬ МЕЖДУ STD, ДИСПЕРСИЕЙ И RMS
Объяснение формул и их взаимосвязей
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def explain_std_variance_rms_relationship():
    """Объясняет связь между std, дисперсией и RMS"""
    
    print("=================================================================================")
    print("🔍 СВЯЗЬ МЕЖДУ STD, ДИСПЕРСИЕЙ И RMS")
    print("=================================================================================")
    print("📋 Объяснение формул и их взаимосвязей")
    print("=================================================================================")
    
    print("📊 ВАШЕ ОПРЕДЕЛЕНИЕ STD:")
    print("="*60)
    print("✅ ВЫ АБСОЛЮТНО ПРАВЫ!")
    print("📊 Стандартное отклонение (std):")
    print("   std = √(Σ(x - mean)² / n)")
    print("   где n - количество отсчетов")
    print("   Это корень из дисперсии!")
    
    print("\n📋 МАТЕМАТИЧЕСКИЕ ФОРМУЛЫ:")
    print("="*60)
    print("1️⃣ ДИСПЕРСИЯ (Variance):")
    print("   σ² = Σ(x - mean)² / n")
    print("   или σ² = mean((x - mean)²)")
    
    print("\n2️⃣ СТАНДАРТНОЕ ОТКЛОНЕНИЕ (Std):")
    print("   σ = √(σ²) = √(Σ(x - mean)² / n)")
    print("   σ = √(mean((x - mean)²))")
    
    print("\n3️⃣ RMS (Root Mean Square):")
    print("   RMS = √(mean(x²))")
    print("   RMS = √(Σ(x²) / n)")
    
    print("\n" + "="*80)
    print("📈 ПРАКТИЧЕСКАЯ ДЕМОНСТРАЦИЯ")
    print("="*80)
    
    # Создаем тестовые данные
    np.random.seed(42)
    noise_level = 0.1  # 10% шум
    n_samples = 1000
    
    # Генерируем гауссов шум
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Вычисляем статистики по формулам
    mean_val = np.mean(noise)
    
    # Дисперсия
    variance = np.mean((noise - mean_val)**2)
    
    # Стандартное отклонение (корень из дисперсии)
    std_val = np.sqrt(variance)
    
    # RMS
    rms_val = np.sqrt(np.mean(noise**2))
    
    print(f"🔍 ТЕСТОВЫЙ ШУМ: {noise_level*100}% (σ={noise_level})")
    print("-" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ:")
    print(f"   • Mean: {mean_val:.8f} ≈ 0")
    print(f"   • Variance (σ²): {variance:.8f} ≈ {noise_level**2:.6f}")
    print(f"   • Std (σ): {std_val:.8f} ≈ {noise_level:.3f}")
    print(f"   • RMS: {rms_val:.8f} ≈ {noise_level:.3f}")
    
    print(f"\n📐 ПРОВЕРКА ФОРМУЛ:")
    print(f"   • std = √(variance): {std_val:.6f} = √({variance:.6f}) ✅")
    print(f"   • std = √(mean((x - mean)²)): {std_val:.6f} ✅")
    print(f"   • RMS = √(mean(x²)): {rms_val:.6f} ✅")
    print(f"   • Для mean≈0: RMS = std ✅")
    
    print(f"\n📊 ПОДТВЕРЖДЕНИЕ ВАШЕГО ОПРЕДЕЛЕНИЯ:")
    print(f"   • std = корень из дисперсии ✅")
    print(f"   • std = √(Σ(x - mean)² / n) ✅")
    print(f"   • std = √({variance:.6f}) = {std_val:.6f} ✅")
    
    # Показываем пошаговый расчет
    print(f"\n📋 ПОШАГОВЫЙ РАСЧЕТ:")
    print(f"   1. Вычисляем mean: {mean_val:.6f}")
    print(f"   2. Вычисляем (x - mean)² для каждого отсчета")
    print(f"   3. Находим среднее: {variance:.6f} (дисперсия)")
    print(f"   4. Извлекаем корень: √({variance:.6f}) = {std_val:.6f} (std)")
    
    # Создаем визуализацию
    create_relationship_visualization()
    
    print("\n" + "="*80)
    print("✅ ИТОГОВЫЕ ВЫВОДЫ")
    print("="*80)
    print("1. ✅ Ваше определение std абсолютно корректно")
    print("2. ✅ std = корень из дисперсии")
    print("3. ✅ std = √(Σ(x - mean)² / n)")
    print("4. ✅ Для гауссова шума с mean=0: RMS = std")
    print("5. ✅ Все формулы взаимосвязаны и корректны")
    print("="*80)

def create_relationship_visualization():
    """Создает визуализацию связи между std, дисперсией и RMS"""
    
    print("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ...")
    
    np.random.seed(42)
    
    # Параметры
    noise_level = 0.1  # 10% шум
    n_samples = 1000
    
    # Генерируем шум
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Вычисляем статистики
    mean_val = np.mean(noise)
    variance = np.mean((noise - mean_val)**2)
    std_val = np.sqrt(variance)
    rms_val = np.sqrt(np.mean(noise**2))
    
    # Создаем график
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Исходные данные
    ax1.hist(noise, bins=30, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax1.set_title(f'Исходные данные\nMean={mean_val:.4f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Значение')
    ax1.set_ylabel('Плотность')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_val:.4f}')
    ax1.legend()
    
    # График 2: (x - mean)²
    squared_diff = (noise - mean_val)**2
    ax2.hist(squared_diff, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
    ax2.set_title(f'(x - mean)²\nVariance={variance:.6f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('(x - mean)²')
    ax2.set_ylabel('Плотность')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=variance, color='red', linestyle='--', linewidth=2, label=f'Mean={variance:.6f}')
    ax2.legend()
    
    # График 3: x²
    x_squared = noise**2
    ax3.hist(x_squared, bins=30, alpha=0.7, color='purple', edgecolor='black', density=True)
    ax3.set_title(f'x²\nMean(x²)={np.mean(x_squared):.6f}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x²')
    ax3.set_ylabel('Плотность')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=np.mean(x_squared), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(x_squared):.6f}')
    ax3.legend()
    
    # График 4: Сравнение формул
    formulas = ['std = √(variance)', 'std = √(mean((x-mean)²))', 'RMS = √(mean(x²))']
    values = [std_val, std_val, rms_val]
    colors = ['blue', 'green', 'purple']
    
    bars = ax4.bar(formulas, values, color=colors, alpha=0.7)
    ax4.set_title('Сравнение формул', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Значение')
    ax4.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.6f}', ha='center', va='bottom', fontsize=10)
    
    # Добавляем формулы
    formula_text = f'std = √({variance:.6f}) = {std_val:.6f}\nRMS = √({np.mean(x_squared):.6f}) = {rms_val:.6f}\nstd ≈ RMS (mean≈0)'
    ax4.text(0.02, 0.98, formula_text, transform=ax4.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = 'std_variance_rms_relationship.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 График сохранен: {filename}")
    plt.show()

if __name__ == "__main__":
    explain_std_variance_rms_relationship() 