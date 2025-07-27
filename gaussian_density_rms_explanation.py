#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОБЪЯСНЕНИЕ ПЛОТНОСТИ ВЕРОЯТНОСТИ ГАУССОВА РАСПРЕДЕЛЕНИЯ И RMS
Разъяснение формул и их взаимосвязи
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def explain_gaussian_density_and_rms():
    """Объясняет плотность вероятности гауссова распределения и RMS"""
    
    print("=================================================================================")
    print("🔍 ПЛОТНОСТЬ ВЕРОЯТНОСТИ ГАУССОВА РАСПРЕДЕЛЕНИЯ И RMS")
    print("=================================================================================")
    print("📋 Разъяснение формул и их взаимосвязи")
    print("=================================================================================")
    
    print("📊 ВАШ ВОПРОС О ПЛОТНОСТИ ВЕРОЯТНОСТИ:")
    print("="*60)
    print("✅ ВЫ АБСОЛЮТНО ПРАВЫ!")
    print("📊 Плотность вероятности гауссова распределения:")
    print("   f(x) = 1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))")
    print("   где μ = mean, σ = std (sigma)")
    
    print("\n📋 ДЛЯ ГАУССОВА ШУМА С MEAN=0:")
    print("   f(x) = 1/(σ√(2π)) * exp(-x²/(2σ²))")
    print("   где σ = std (sigma) = noise_level")
    
    print("\n" + "="*80)
    print("📈 ПРАКТИЧЕСКАЯ ПРОВЕРКА ФОРМУЛЫ")
    print("="*80)
    
    # Создаем тестовые данные
    np.random.seed(42)
    noise_level = 0.1  # 10% шум
    n_samples = 100000
    
    # Генерируем гауссов шум
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Вычисляем статистики
    mean_val = np.mean(noise)
    std_val = np.std(noise)
    rms_val = np.sqrt(np.mean(noise**2))
    
    print(f"🔍 ТЕСТОВЫЙ ШУМ: {noise_level*100}% (σ={noise_level})")
    print("-" * 50)
    print(f"📊 СТАТИСТИКИ:")
    print(f"   • Mean (μ): {mean_val:.8f} ≈ 0")
    print(f"   • Std (σ): {std_val:.8f} ≈ {noise_level:.3f}")
    print(f"   • RMS: {rms_val:.8f} ≈ {noise_level:.3f}")
    
    print(f"\n📐 ПРОВЕРКА ФОРМУЛ:")
    print(f"   • std = √(mean((x - mean)²)) = {std_val:.6f}")
    print(f"   • RMS = √(mean(x²)) = {rms_val:.6f}")
    print(f"   • Для mean≈0: RMS = std ✅")
    
    print(f"\n📊 ПЛОТНОСТЬ ВЕРОЯТНОСТИ:")
    print(f"   • f(x) = 1/({std_val:.6f}*√(2π)) * exp(-x²/(2*{std_val:.6f}²))")
    print(f"   • f(x) = {1/(std_val*np.sqrt(2*np.pi)):.6f} * exp(-x²/(2*{std_val**2:.6f}))")
    
    print("\n" + "="*80)
    print("❓ ОТВЕТ НА ВАШ ВОПРОС О RMS")
    print("="*80)
    
    print("\n🔍 ВАШ ВОПРОС: 'Про корень странная формула'")
    print("\n📊 ОБЪЯСНЕНИЕ ФОРМУЛЫ RMS:")
    print("   📐 RMS = Root Mean Square = √(mean(x²))")
    print("   📐 Это НЕ плотность вероятности!")
    print("   📐 Это статистическая характеристика сигнала")
    
    print("\n📋 РАЗНИЦА МЕЖДУ ФОРМУЛАМИ:")
    print("   1️⃣ ПЛОТНОСТЬ ВЕРОЯТНОСТИ (ваша формула):")
    print("      f(x) = 1/(σ√(2π)) * exp(-x²/(2σ²))")
    print("      Это функция, которая описывает вероятность значений")
    
    print("\n   2️⃣ RMS (наша формула):")
    print("      RMS = √(mean(x²))")
    print("      Это число, которое характеризует 'мощность' сигнала")
    
    print("\n📊 СВЯЗЬ МЕЖДУ НИМИ:")
    print("   • Для гауссова шума с mean=0: RMS = std")
    print("   • std входит в формулу плотности вероятности")
    print("   • RMS используется для характеристики амплитуды")
    
    # Создаем визуализацию
    create_gaussian_density_visualization()
    
    print("\n" + "="*80)
    print("✅ ИТОГОВЫЕ ВЫВОДЫ")
    print("="*80)
    print("1. ✅ Ваша формула плотности вероятности абсолютно правильная")
    print("2. ✅ RMS = √(mean(x²)) - это другая формула, не плотность")
    print("3. ✅ Для гауссова шума с mean=0: RMS = std")
    print("4. ✅ std входит в формулу плотности вероятности")
    print("5. ✅ Обе формулы корректны и взаимосвязаны")
    print("="*80)

def create_gaussian_density_visualization():
    """Создает визуализацию плотности вероятности и RMS"""
    
    print("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ...")
    
    np.random.seed(42)
    
    # Параметры
    noise_level = 0.1  # 10% шум
    n_samples = 10000
    
    # Генерируем шум
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Вычисляем статистики
    mean_val = np.mean(noise)
    std_val = np.std(noise)
    rms_val = np.sqrt(np.mean(noise**2))
    
    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Гистограмма и плотность вероятности
    ax1.hist(noise, bins=50, alpha=0.7, color='red', edgecolor='black', density=True, label='Гистограмма')
    
    # Теоретическая плотность вероятности
    x = np.linspace(-4*noise_level, 4*noise_level, 1000)
    y_density = (1/(std_val*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_val)/std_val)**2)
    ax1.plot(x, y_density, 'b-', linewidth=2, label='Плотность вероятности')
    
    ax1.set_title(f'Плотность вероятности гауссова шума\nσ={std_val:.4f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Значение шума')
    ax1.set_ylabel('Плотность вероятности')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Добавляем формулу
    formula_text = f'f(x) = 1/(σ√(2π)) * exp(-x²/(2σ²))\nσ = {std_val:.4f}'
    ax1.text(0.02, 0.98, formula_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8), fontsize=10)
    
    # Вертикальные линии
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Mean=0')
    ax1.axvline(x=std_val, color='orange', linestyle='--', alpha=0.7, label=f'Std={std_val:.3f}')
    ax1.axvline(x=-std_val, color='orange', linestyle='--', alpha=0.7)
    
    # График 2: RMS объяснение
    # Показываем x²
    x_squared = noise**2
    ax2.hist(x_squared, bins=50, alpha=0.7, color='purple', edgecolor='black', density=True, label='x²')
    
    # Среднее x²
    mean_x_squared = np.mean(x_squared)
    ax2.axvline(x=mean_x_squared, color='red', linestyle='--', linewidth=2, label=f'mean(x²)={mean_x_squared:.4f}')
    
    # RMS
    ax2.axvline(x=rms_val**2, color='blue', linestyle='--', linewidth=2, label=f'RMS²={rms_val**2:.4f}')
    
    ax2.set_title(f'Распределение x² и вычисление RMS\nRMS = √(mean(x²)) = {rms_val:.4f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x²')
    ax2.set_ylabel('Плотность')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Добавляем формулу RMS
    rms_formula = f'RMS = √(mean(x²))\nRMS = √({mean_x_squared:.4f})\nRMS = {rms_val:.4f}'
    ax2.text(0.02, 0.98, rms_formula, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor="white", alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = 'gaussian_density_rms_explanation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 График сохранен: {filename}")
    plt.show()

if __name__ == "__main__":
    explain_gaussian_density_and_rms() 