#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ДЕТАЛЬНЫЙ АНАЛИЗ RMS И STD
Разъяснение разницы между стандартным отклонением и RMS
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def detailed_rms_analysis():
    """Детальный анализ RMS и std для гауссова шума"""
    
    print("=================================================================================")
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ RMS И STD")
    print("=================================================================================")
    print("📋 Разъяснение разницы между стандартным отклонением и RMS")
    print("=================================================================================")
    
    np.random.seed(42)
    
    # Создаем тестовые данные
    n_samples = 100000
    noise_levels = [0.01, 0.05, 0.10, 0.20]
    
    print("📊 ТЕОРЕТИЧЕСКИЕ ОСНОВЫ:")
    print("="*60)
    print("1️⃣ ГАУССОВ БЕЛЫЙ ШУМ:")
    print("   ✅ Не коррелирует между отсчетами")
    print("   ✅ Mean = 0 (нулевое среднее)")
    print("   ✅ Используется для каждого спектрального отсчета")
    print("   ✅ Не влияет на точность классификации (в разумных пределах)")
    
    print("\n2️⃣ СТАНДАРТНОЕ ОТКЛОНЕНИЕ (STD):")
    print("   📊 std = √(mean((x - mean)²))")
    print("   📊 Для гауссова шума с mean=0: std = √(mean(x²))")
    print("   📊 Это мера разброса значений вокруг среднего")
    
    print("\n3️⃣ ROOT MEAN SQUARE (RMS):")
    print("   📊 RMS = √(mean(x²))")
    print("   📊 Для гауссова шума с mean=0: RMS = std")
    print("   📊 Это квадратичное среднее значение")
    
    print("\n" + "="*80)
    print("📈 ПРАКТИЧЕСКИЙ АНАЛИЗ")
    print("="*80)
    
    for noise_level in noise_levels:
        print(f"\n🔍 УРОВЕНЬ ШУМА: {noise_level*100}% ({noise_level})")
        print("-" * 60)
        
        # Генерируем шум
        noise = np.random.normal(0, noise_level, n_samples)
        
        # Вычисляем статистики
        mean_val = np.mean(noise)
        std_val = np.std(noise)
        rms_val = np.sqrt(np.mean(noise**2))
        variance = np.var(noise)
        
        print(f"📊 СТАТИСТИКИ ШУМА:")
        print(f"   • Mean: {mean_val:.8f} (должен быть ≈ 0)")
        print(f"   • Variance: {variance:.8f} (должна быть ≈ {noise_level**2:.6f})")
        print(f"   • Std: {std_val:.8f} (должен быть ≈ {noise_level:.6f})")
        print(f"   • RMS: {rms_val:.8f} (должен быть ≈ {noise_level:.6f})")
        
        # Проверяем теоретические соотношения
        print(f"\n📋 ПРОВЕРКА ТЕОРЕТИЧЕСКИХ СООТНОШЕНИЙ:")
        print(f"   • std² ≈ variance: {'✅' if abs(std_val**2 - variance) < 0.0001 else '❌'}")
        print(f"   • RMS ≈ std: {'✅' if abs(rms_val - std_val) < 0.0001 else '❌'}")
        print(f"   • RMS² ≈ variance: {'✅' if abs(rms_val**2 - variance) < 0.0001 else '❌'}")
        
        # Показываем формулы
        print(f"\n📐 ФОРМУЛЫ:")
        print(f"   • std = √(mean((x - {mean_val:.6f})²)) = {std_val:.6f}")
        print(f"   • RMS = √(mean(x²)) = {rms_val:.6f}")
        print(f"   • variance = mean((x - {mean_val:.6f})²) = {variance:.6f}")
        
        # Для гауссова шума с mean=0
        if abs(mean_val) < 0.001:
            print(f"\n✅ ДЛЯ ГАУССОВА ШУМА С MEAN≈0:")
            print(f"   • std = RMS = {std_val:.6f}")
            print(f"   • std² = RMS² = variance = {variance:.6f}")
    
    print("\n" + "="*80)
    print("❓ ОТВЕТ НА ВАШ ВОПРОС")
    print("="*80)
    
    print("\n🔍 ВАШ ВОПРОС: 'RMS как процент берется'?")
    print("\n📊 ОТВЕТ:")
    print("   ❌ НЕТ! RMS не берется как процент")
    print("   📊 RMS = √(mean(x²)) - это математическая формула")
    print("   📊 Для гауссова шума с mean=0: RMS = std")
    print("   📊 Процент в названии (1%, 5%, 10%) - это условное обозначение")
    
    print("\n📋 ЧТО ОЗНАЧАЕТ '1% ШУМ':")
    print("   📊 Это означает std шума = 0.01")
    print("   📊 RMS шума = 0.01 (так как mean=0)")
    print("   📊 Процент - это отношение к std данных")
    
    print("\n📊 ПРИМЕР:")
    print("   📊 Если данные имеют std=1 (нормализованные):")
    print("   📊 '1% шум' означает std_шума = 0.01 = 1% от std_данных")
    print("   📊 RMS_шума = 0.01 = 1% от std_данных")
    
    # Создаем визуализацию
    create_detailed_visualization()
    
    print("\n" + "="*80)
    print("✅ ИТОГОВЫЕ ВЫВОДЫ")
    print("="*80)
    print("1. ✅ Гауссов шум имеет mean=0 (нулевое среднее)")
    print("2. ✅ Для гауссова шума с mean=0: RMS = std")
    print("3. ✅ RMS не берется как процент - это математическая формула")
    print("4. ✅ Процент в названии - условное обозначение уровня")
    print("5. ✅ Белый шум не коррелирует между отсчетами")
    print("6. ✅ Шум применяется к каждому спектральному отсчету")
    print("="*80)

def create_detailed_visualization():
    """Создает детальную визуализацию RMS и std"""
    
    print("\n📊 СОЗДАНИЕ ДЕТАЛЬНОЙ ВИЗУАЛИЗАЦИИ...")
    
    np.random.seed(42)
    
    # Создаем данные
    noise_levels = [0.01, 0.05, 0.10, 0.20]
    n_samples = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, noise_level in enumerate(noise_levels):
        # Генерируем шум
        noise = np.random.normal(0, noise_level, n_samples)
        
        # Гистограмма
        axes[i].hist(noise, bins=50, alpha=0.7, color='red', edgecolor='black', density=True)
        
        # Теоретическая кривая
        x = np.linspace(-4*noise_level, 4*noise_level, 1000)
        y = (1/(noise_level*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-0)/noise_level)**2)
        axes[i].plot(x, y, 'b-', linewidth=2, label='Теоретическая')
        
        # Статистика
        mean_actual = np.mean(noise)
        std_actual = np.std(noise)
        rms_actual = np.sqrt(np.mean(noise**2))
        variance_actual = np.var(noise)
        
        axes[i].set_title(f'Шум {noise_level*100}% (std={noise_level:.3f})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Значение шума')
        axes[i].set_ylabel('Плотность вероятности')
        axes[i].grid(True, alpha=0.3)
        
        # Добавляем детальную статистику
        stats_text = f'Mean: {mean_actual:.6f}\nStd: {std_actual:.6f}\nRMS: {rms_actual:.6f}\nVar: {variance_actual:.6f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="white", alpha=0.8), fontsize=9)
        
        # Вертикальные линии
        axes[i].axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Mean=0')
        axes[i].axvline(x=std_actual, color='orange', linestyle='--', alpha=0.7, label=f'Std={std_actual:.3f}')
        axes[i].axvline(x=-std_actual, color='orange', linestyle='--', alpha=0.7)
        axes[i].axvline(x=rms_actual, color='purple', linestyle=':', alpha=0.7, label=f'RMS={rms_actual:.3f}')
        axes[i].axvline(x=-rms_actual, color='purple', linestyle=':', alpha=0.7)
        
        axes[i].legend()
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = 'detailed_rms_std_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 График сохранен: {filename}")
    plt.show()

if __name__ == "__main__":
    detailed_rms_analysis() 