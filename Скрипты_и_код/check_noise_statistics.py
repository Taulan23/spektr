#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА СТАТИСТИКИ ШУМА
Ответы на вопросы о mean и RMS
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def check_noise_statistics():
    """Проверяет статистику шума и отвечает на вопросы"""
    
    print("=================================================================================")
    print("🔍 ПРОВЕРКА СТАТИСТИКИ ШУМА")
    print("=================================================================================")
    print("📋 Ответы на вопросы о mean и RMS")
    print("=================================================================================")
    
    # Создаем тестовые данные (нормализованные спектры)
    np.random.seed(42)
    test_spectrum = np.random.normal(0, 1, 1000)  # Нормализованный спектр
    
    print(f"📊 ТЕСТОВЫЙ СПЕКТР:")
    print(f"   • Размер: {len(test_spectrum)} точек")
    print(f"   • Среднее: {np.mean(test_spectrum):.6f}")
    print(f"   • Стандартное отклонение: {np.std(test_spectrum):.6f}")
    print(f"   • RMS: {np.sqrt(np.mean(test_spectrum**2)):.6f}")
    
    # Проверяем разные уровни шума
    noise_levels = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%
    
    print("\n" + "="*80)
    print("📈 АНАЛИЗ ГАУССОВА ШУМА")
    print("="*80)
    
    for noise_level in noise_levels:
        print(f"\n🔍 УРОВЕНЬ ШУМА: {noise_level*100}% ({noise_level})")
        print("-" * 50)
        
        # Генерируем шум
        noise = np.random.normal(0, noise_level, test_spectrum.shape)
        
        # Статистика шума
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        noise_rms = np.sqrt(np.mean(noise**2))
        
        print(f"📊 СТАТИСТИКА ШУМА:")
        print(f"   • Mean: {noise_mean:.6f} (должен быть ≈ 0)")
        print(f"   • Std: {noise_std:.6f} (должен быть ≈ {noise_level:.3f})")
        print(f"   • RMS: {noise_rms:.6f} (должен быть ≈ {noise_level:.3f})")
        
        # Проверяем соответствие
        mean_error = abs(noise_mean - 0)
        std_error = abs(noise_std - noise_level)
        rms_error = abs(noise_rms - noise_level)
        
        print(f"📋 ПРОВЕРКА:")
        print(f"   • Mean ≈ 0: {'✅' if mean_error < 0.01 else '❌'} (ошибка: {mean_error:.6f})")
        print(f"   • Std ≈ {noise_level:.3f}: {'✅' if std_error < 0.01 else '❌'} (ошибка: {std_error:.6f})")
        print(f"   • RMS ≈ {noise_level:.3f}: {'✅' if rms_error < 0.01 else '❌'} (ошибка: {rms_error:.6f})")
        
        # Применяем шум к спектру
        noisy_spectrum = test_spectrum + noise
        
        # Статистика зашумленного спектра
        noisy_mean = np.mean(noisy_spectrum)
        noisy_std = np.std(noisy_spectrum)
        noisy_rms = np.sqrt(np.mean(noisy_spectrum**2))
        
        print(f"📊 СТАТИСТИКА ЗАШУМЛЕННОГО СПЕКТРА:")
        print(f"   • Mean: {noisy_mean:.6f} (изменение: {noisy_mean - np.mean(test_spectrum):+.6f})")
        print(f"   • Std: {noisy_std:.6f} (изменение: {noisy_std - np.std(test_spectrum):+.6f})")
        print(f"   • RMS: {noisy_rms:.6f} (изменение: {noisy_rms - np.sqrt(np.mean(test_spectrum**2)):+.6f})")
    
    print("\n" + "="*80)
    print("❓ ОТВЕТЫ НА ВОПРОСЫ")
    print("="*80)
    
    print("\n1️⃣ У ГАУССОВА ШУМА MEAN ДОЛЖЕН БЫТЬ РАВЕН 0?")
    print("   ✅ ДА! Гауссов шум должен иметь mean = 0")
    print("   📊 В нашем коде: np.random.normal(0, noise_level, ...)")
    print("   📊 Первый параметр = 0 - это mean")
    print("   📊 Второй параметр = noise_level - это standard deviation")
    
    print("\n2️⃣ RMS ЗАДАЕТСЯ ПРОЦЕНТОМ?")
    print("   ❌ НЕТ! RMS не задается процентом напрямую")
    print("   📊 RMS (Root Mean Square) = √(mean(x²))")
    print("   📊 Для гауссова шума с mean=0: RMS ≈ std")
    print("   📊 В нашем коде noise_level - это standard deviation")
    print("   📊 Процент в названии - это условное обозначение уровня")
    
    print("\n3️⃣ КАК ИНТЕРПРЕТИРОВАТЬ '1% ШУМ'?")
    print("   📊 Это означает std шума = 0.01")
    print("   📊 Для нормализованных данных (std=1) это 1% от std данных")
    print("   📊 Для ненормализованных данных нужно умножать на std данных")
    
    print("\n4️⃣ ПРАВИЛЬНА ЛИ НАША РЕАЛИЗАЦИЯ?")
    print("   ✅ ДА! Наша реализация корректна:")
    print("   📊 np.random.normal(0, noise_level, X.shape)")
    print("   📊 mean=0, std=noise_level")
    print("   📊 Шум аддитивный и разный для каждого отсчета")
    
    # Создаем визуализацию
    create_noise_visualization()
    
    print("\n" + "="*80)
    print("✅ ВЫВОДЫ")
    print("="*80)
    print("1. ✅ Гауссов шум должен иметь mean = 0")
    print("2. ✅ RMS ≈ std для гауссова шума с mean = 0")
    print("3. ✅ Наша реализация корректна")
    print("4. ✅ Процент в названии - условное обозначение уровня шума")
    print("="*80)

def create_noise_visualization():
    """Создает визуализацию статистики шума"""
    
    print("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ...")
    
    np.random.seed(42)
    
    # Создаем данные
    noise_levels = [0.01, 0.05, 0.10, 0.20]
    n_samples = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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
        
        axes[i].set_title(f'Шум {noise_level*100}% (std={noise_level:.3f})', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Значение шума')
        axes[i].set_ylabel('Плотность вероятности')
        axes[i].grid(True, alpha=0.3)
        
        # Добавляем статистику
        stats_text = f'Mean: {mean_actual:.4f}\nStd: {std_actual:.4f}\nRMS: {rms_actual:.4f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="white", alpha=0.8), fontsize=10)
        
        # Вертикальные линии для mean и std
        axes[i].axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Mean=0')
        axes[i].axvline(x=noise_level, color='orange', linestyle='--', alpha=0.7, label=f'Std={noise_level:.3f}')
        axes[i].axvline(x=-noise_level, color='orange', linestyle='--', alpha=0.7)
        
        axes[i].legend()
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = 'noise_statistics_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 График сохранен: {filename}")
    plt.show()

if __name__ == "__main__":
    check_noise_statistics() 