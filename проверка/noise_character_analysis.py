#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ ХАРАКТЕРА АДДИТИВНОГО ШУМА
Проверка: разный ли шум для каждого спектрального отсчета или одинаковый для всего спектра
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_sample_spectra():
    """Загружает несколько спектров для анализа шума"""
    
    print("🌱 ЗАГРУЗКА ОБРАЗЦОВ СПЕКТРОВ ДЛЯ АНАЛИЗА ШУМА...")
    
    # Загружаем несколько спектров из разных видов
    sample_spectra = []
    sample_labels = []
    
    # Проверяем разные папки
    folders_to_check = [
        "Спектры, весенний период, 20 видов/береза",
        "Спектры, весенний период, 7 видов/береза", 
        "береза"
    ]
    
    for folder in folders_to_check:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*.xlsx"))
            if files:
                print(f"   📁 Найдены файлы в: {folder}")
                
                # Берем первые 3 файла
                for i, file in enumerate(files[:3]):
                    try:
                        df = pd.read_excel(file, header=None)
                        spectrum = df.iloc[:, 1].values  # Вторая колонка
                        spectrum = spectrum[~pd.isna(spectrum)]  # Убираем NaN
                        
                        if len(spectrum) > 100:  # Минимальная длина
                            sample_spectra.append(spectrum)
                            sample_labels.append(f"Береза_{i+1}")
                            print(f"      ✅ Загружен спектр {i+1}: {len(spectrum)} точек")
                    except Exception as e:
                        print(f"      ❌ Ошибка в файле {file}: {e}")
                        continue
                break
    
    if not sample_spectra:
        print("❌ Не удалось загрузить спектры!")
        return [], []
    
    print(f"✅ Загружено {len(sample_spectra)} образцов спектров")
    return sample_spectra, sample_labels

def analyze_noise_characteristics(spectra, labels):
    """Анализирует характеристики шума"""
    
    print("\n🔍 АНАЛИЗ ХАРАКТЕРИСТИК ШУМА...")
    
    # Нормализуем спектры для лучшего сравнения
    scaler = StandardScaler()
    normalized_spectra = []
    
    for spectrum in spectra:
        # Нормализуем каждый спектр отдельно
        normalized = scaler.fit_transform(spectrum.reshape(-1, 1)).flatten()
        normalized_spectra.append(normalized)
    
    # Анализируем разные типы шума
    noise_levels = [0.05, 0.10]  # 5%, 10%
    
    for noise_level in noise_levels:
        print(f"\n📊 АНАЛИЗ ШУМА {noise_level*100}%:")
        
        for i, (spectrum, label) in enumerate(zip(normalized_spectra, labels)):
            print(f"\n   🌳 {label}:")
            
            # Тип 1: Разный шум для каждого отсчета (как в нашем коде)
            noise_individual = np.random.normal(0, noise_level, spectrum.shape)
            spectrum_with_individual_noise = spectrum + noise_individual
            
            # Тип 2: Одинаковый шум для всего спектра
            noise_uniform = np.random.normal(0, noise_level, 1)  # Один случайный шум
            spectrum_with_uniform_noise = spectrum + noise_uniform
            
            # Анализируем различия
            print(f"      📈 Разный шум для каждого отсчета:")
            print(f"         • Стандартное отклонение шума: {np.std(noise_individual):.6f}")
            print(f"         • Диапазон шума: [{np.min(noise_individual):.6f}, {np.max(noise_individual):.6f}]")
            print(f"         • Среднее шума: {np.mean(noise_individual):.6f}")
            
            print(f"      📊 Одинаковый шум для всего спектра:")
            print(f"         • Значение шума: {noise_uniform[0]:.6f}")
            print(f"         • Стандартное отклонение: 0.000000")
            print(f"         • Диапазон: [{noise_uniform[0]:.6f}, {noise_uniform[0]:.6f}]")
            
            # Сравниваем влияние на спектр
            original_std = np.std(spectrum)
            individual_noise_std = np.std(spectrum_with_individual_noise)
            uniform_noise_std = np.std(spectrum_with_uniform_noise)
            
            print(f"      📊 Влияние на спектр:")
            print(f"         • Исходное std: {original_std:.6f}")
            print(f"         • С разным шумом: {individual_noise_std:.6f}")
            print(f"         • С одинаковым шумом: {uniform_noise_std:.6f}")
            
            # Разница в стандартном отклонении
            individual_change = individual_noise_std - original_std
            uniform_change = uniform_noise_std - original_std
            
            print(f"      📉 Изменение std:")
            print(f"         • Разный шум: {individual_change:+.6f}")
            print(f"         • Одинаковый шум: {uniform_change:+.6f}")

def create_noise_comparison_visualizations(spectra, labels):
    """Создает визуализации сравнения типов шума"""
    
    print("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ СРАВНЕНИЯ ШУМА...")
    
    # Нормализуем спектры
    scaler = StandardScaler()
    normalized_spectra = []
    
    for spectrum in spectra:
        normalized = scaler.fit_transform(spectrum.reshape(-1, 1)).flatten()
        normalized_spectra.append(normalized)
    
    # Создаем графики для каждого спектра
    for i, (spectrum, label) in enumerate(zip(normalized_spectra, labels)):
        plt.figure(figsize=(16, 8))
        
        # Уровни шума для анализа
        noise_levels = [0.05, 0.10]
        
        for j, noise_level in enumerate(noise_levels):
            # Генерируем шумы
            np.random.seed(42)  # Для воспроизводимости
            noise_individual = np.random.normal(0, noise_level, spectrum.shape)
            noise_uniform = np.random.normal(0, noise_level, 1)
            
            # Применяем шумы
            spectrum_individual = spectrum + noise_individual
            spectrum_uniform = spectrum + noise_uniform
            
            # Subplot 1: Сравнение спектров
            plt.subplot(2, 3, j*3 + 1)
            plt.plot(spectrum[:200], 'b-', linewidth=1, alpha=0.7, label='Исходный')
            plt.plot(spectrum_individual[:200], 'r-', linewidth=1, alpha=0.7, label='Разный шум')
            plt.plot(spectrum_uniform[:200], 'g-', linewidth=1, alpha=0.7, label='Одинаковый шум')
            plt.title(f'{label}: Спектры ({noise_level*100}% шума)', fontsize=12, fontweight='bold')
            plt.xlabel('Спектральный канал')
            plt.ylabel('Нормализованная интенсивность')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Распределение шума (разный)
            plt.subplot(2, 3, j*3 + 2)
            plt.hist(noise_individual, bins=50, alpha=0.7, color='red', edgecolor='black')
            plt.title(f'Распределение разного шума ({noise_level*100}%)', fontsize=12, fontweight='bold')
            plt.xlabel('Значение шума')
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
            
            # Добавляем статистику
            plt.text(0.02, 0.98, f'std: {np.std(noise_individual):.4f}\nmean: {np.mean(noise_individual):.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Subplot 3: Распределение шума (одинаковый)
            plt.subplot(2, 3, j*3 + 3)
            plt.axvline(x=noise_uniform[0], color='green', linewidth=3, label=f'Шум: {noise_uniform[0]:.4f}')
            plt.title(f'Одинаковый шум ({noise_level*100}%)', fontsize=12, fontweight='bold')
            plt.xlabel('Значение шума')
            plt.ylabel('Частота')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'noise_character_analysis_{label}_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   📊 График сохранен: {filename}")
        plt.show()

def create_noise_impact_analysis(spectra, labels):
    """Анализирует влияние разных типов шума на характеристики спектров"""
    
    print("\n📈 АНАЛИЗ ВЛИЯНИЯ ШУМА НА ХАРАКТЕРИСТИКИ СПЕКТРОВ...")
    
    # Нормализуем спектры
    scaler = StandardScaler()
    normalized_spectra = []
    
    for spectrum in spectra:
        normalized = scaler.fit_transform(spectrum.reshape(-1, 1)).flatten()
        normalized_spectra.append(normalized)
    
    # Уровни шума
    noise_levels = [0.0, 0.01, 0.05, 0.10]
    
    # Результаты анализа
    results = {
        'individual_noise': {'std': [], 'mean': [], 'range': []},
        'uniform_noise': {'std': [], 'mean': [], 'range': []}
    }
    
    # Анализируем первый спектр как пример
    spectrum = normalized_spectra[0]
    
    for noise_level in noise_levels:
        if noise_level == 0:
            # Без шума
            results['individual_noise']['std'].append(np.std(spectrum))
            results['individual_noise']['mean'].append(np.mean(spectrum))
            results['individual_noise']['range'].append(np.ptp(spectrum))
            
            results['uniform_noise']['std'].append(np.std(spectrum))
            results['uniform_noise']['mean'].append(np.mean(spectrum))
            results['uniform_noise']['range'].append(np.ptp(spectrum))
        else:
            # С шумом
            np.random.seed(42)
            noise_individual = np.random.normal(0, noise_level, spectrum.shape)
            noise_uniform = np.random.normal(0, noise_level, 1)
            
            spectrum_individual = spectrum + noise_individual
            spectrum_uniform = spectrum + noise_uniform
            
            # Статистики для разного шума
            results['individual_noise']['std'].append(np.std(spectrum_individual))
            results['individual_noise']['mean'].append(np.mean(spectrum_individual))
            results['individual_noise']['range'].append(np.ptp(spectrum_individual))
            
            # Статистики для одинакового шума
            results['uniform_noise']['std'].append(np.std(spectrum_uniform))
            results['uniform_noise']['mean'].append(np.mean(spectrum_uniform))
            results['uniform_noise']['range'].append(np.ptp(spectrum_uniform))
    
    # Создаем графики сравнения
    plt.figure(figsize=(20, 15))
    
    # График 1: Стандартное отклонение
    plt.subplot(2, 3, 1)
    plt.plot(noise_levels, results['individual_noise']['std'], 'ro-', linewidth=2, markersize=8, label='Разный шум')
    plt.plot(noise_levels, results['uniform_noise']['std'], 'bo-', linewidth=2, markersize=8, label='Одинаковый шум')
    plt.title('Влияние шума на стандартное отклонение', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень шума')
    plt.ylabel('Стандартное отклонение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Среднее значение
    plt.subplot(2, 3, 2)
    plt.plot(noise_levels, results['individual_noise']['mean'], 'ro-', linewidth=2, markersize=8, label='Разный шум')
    plt.plot(noise_levels, results['uniform_noise']['mean'], 'bo-', linewidth=2, markersize=8, label='Одинаковый шум')
    plt.title('Влияние шума на среднее значение', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень шума')
    plt.ylabel('Среднее значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Диапазон
    plt.subplot(2, 3, 3)
    plt.plot(noise_levels, results['individual_noise']['range'], 'ro-', linewidth=2, markersize=8, label='Разный шум')
    plt.plot(noise_levels, results['uniform_noise']['range'], 'bo-', linewidth=2, markersize=8, label='Одинаковый шум')
    plt.title('Влияние шума на диапазон', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень шума')
    plt.ylabel('Диапазон (max-min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 4: Относительное изменение std
    plt.subplot(2, 3, 4)
    original_std = results['individual_noise']['std'][0]
    individual_change = [(std - original_std) / original_std * 100 for std in results['individual_noise']['std']]
    uniform_change = [(std - original_std) / original_std * 100 for std in results['uniform_noise']['std']]
    
    plt.plot(noise_levels, individual_change, 'ro-', linewidth=2, markersize=8, label='Разный шум')
    plt.plot(noise_levels, uniform_change, 'bo-', linewidth=2, markersize=8, label='Одинаковый шум')
    plt.title('Относительное изменение std (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень шума')
    plt.ylabel('Изменение std (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 5: Разность между типами шума
    plt.subplot(2, 3, 5)
    std_diff = [ind - uni for ind, uni in zip(results['individual_noise']['std'], results['uniform_noise']['std'])]
    plt.plot(noise_levels, std_diff, 'go-', linewidth=2, markersize=8)
    plt.title('Разность std (разный - одинаковый)', fontsize=14, fontweight='bold')
    plt.xlabel('Уровень шума')
    plt.ylabel('Разность std')
    plt.grid(True, alpha=0.3)
    
    # График 6: Сводная статистика
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Создаем текстовую статистику
    stats_text = f"""
    📊 СРАВНЕНИЕ ТИПОВ ШУМА
    
    🎯 Ключевые различия:
    
    📈 Разный шум для каждого отсчета:
       • Каждый спектральный канал получает свой случайный шум
       • Шум ~ N(0, σ) для каждого канала независимо
       • Увеличивает дисперсию спектра
       • Более реалистично для реальных измерений
    
    📊 Одинаковый шум для всего спектра:
       • Один случайный шум применяется ко всему спектру
       • Шум ~ N(0, σ) один раз для всего спектра
       • Сдвигает спектр без изменения формы
       • Менее реалистично для спектральных данных
    
    📉 Влияние на классификацию:
       • Разный шум: сложнее для моделей
       • Одинаковый шум: проще для моделей
       • Наш код использует РАЗНЫЙ шум (правильно!)
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'noise_impact_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 График сравнения сохранен: {filename}")
    plt.show()

def save_noise_analysis_report(spectra, labels):
    """Сохраняет отчет об анализе шума"""
    
    print("\n💾 СОХРАНЕНИЕ ОТЧЕТА ОБ АНАЛИЗЕ ШУМА...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'noise_character_analysis_report_{timestamp}.txt'
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ОТЧЕТ ОБ АНАЛИЗЕ ХАРАКТЕРА АДДИТИВНОГО ШУМА\n")
        f.write("="*80 + "\n\n")
        
        f.write("🎯 ОТВЕТ НА ВОПРОС ИССЛЕДОВАТЕЛЯ:\n\n")
        
        f.write("❓ Вопрос: Случайный шум аддитивно добавляющийся к спектральному отсчету:\n")
        f.write("   - он случайный и разный для каждого спектрального отсчета (одного спектра)\n")
        f.write("   - или он случайный, но одинаковый для каждого спектрального отсчета (одного спектра)?\n\n")
        
        f.write("✅ ОТВЕТ: В НАШЕМ КОДЕ ИСПОЛЬЗУЕТСЯ РАЗНЫЙ ШУМ ДЛЯ КАЖДОГО ОТСЧЕТА!\n\n")
        
        f.write("📋 ДЕТАЛЬНОЕ ОБЪЯСНЕНИЕ:\n\n")
        
        f.write("1️⃣ РАЗНЫЙ ШУМ ДЛЯ КАЖДОГО ОТСЧЕТА (наш код):\n")
        f.write("   • Каждый спектральный канал получает свой независимый случайный шум\n")
        f.write("   • Шум ~ N(0, σ) генерируется отдельно для каждого канала\n")
        f.write("   • Это более реалистично для реальных спектральных измерений\n")
        f.write("   • Увеличивает дисперсию спектра и усложняет классификацию\n\n")
        
        f.write("2️⃣ ОДИНАКОВЫЙ ШУМ ДЛЯ ВСЕГО СПЕКТРА:\n")
        f.write("   • Один случайный шум применяется ко всему спектру\n")
        f.write("   • Шум ~ N(0, σ) генерируется один раз и добавляется ко всем каналам\n")
        f.write("   • Просто сдвигает спектр без изменения его формы\n")
        f.write("   • Менее реалистично для спектральных данных\n\n")
        
        f.write("🔧 ТЕХНИЧЕСКАЯ РЕАЛИЗАЦИЯ В НАШЕМ КОДЕ:\n")
        f.write("   ```python\n")
        f.write("   # РАЗНЫЙ ШУМ (правильно!)\n")
        f.write("   noise = np.random.normal(0, noise_level, X.shape)\n")
        f.write("   X_noisy = X + noise\n")
        f.write("   ```\n\n")
        
        f.write("   vs\n\n")
        
        f.write("   ```python\n")
        f.write("   # ОДИНАКОВЫЙ ШУМ (неправильно для спектров)\n")
        f.write("   noise = np.random.normal(0, noise_level, 1)\n")
        f.write("   X_noisy = X + noise\n")
        f.write("   ```\n\n")
        
        f.write("📊 ВЛИЯНИЕ НА КЛАССИФИКАЦИЮ:\n")
        f.write("   • Разный шум: более сложная задача для моделей\n")
        f.write("   • Одинаковый шум: более простая задача\n")
        f.write("   • Наши результаты с разным шумом более реалистичны\n\n")
        
        f.write("🎯 ВЫВОД:\n")
        f.write("   Наш код правильно использует РАЗНЫЙ шум для каждого спектрального отсчета,\n")
        f.write("   что соответствует реальным условиям спектральных измерений.\n")
        f.write("   Это делает наши результаты более надежными и реалистичными.\n\n")
        
        f.write("="*80 + "\n")
        f.write("Отчет создан: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("="*80 + "\n")
    
    print(f"   📄 Отчет сохранен: {report_filename}")

def main():
    """Главная функция"""
    
    print("="*80)
    print("🔍 АНАЛИЗ ХАРАКТЕРА АДДИТИВНОГО ШУМА")
    print("="*80)
    print("📋 По запросу исследователя")
    print("="*80)
    
    # Загрузка образцов спектров
    spectra, labels = load_sample_spectra()
    
    if len(spectra) == 0:
        print("❌ Не удалось загрузить спектры!")
        return
    
    # Анализ характеристик шума
    analyze_noise_characteristics(spectra, labels)
    
    # Создание визуализаций
    create_noise_comparison_visualizations(spectra, labels)
    create_noise_impact_analysis(spectra, labels)
    
    # Сохранение отчета
    save_noise_analysis_report(spectra, labels)
    
    print("\n" + "="*80)
    print("✅ АНАЛИЗ ХАРАКТЕРА ШУМА ЗАВЕРШЕН!")
    print("🎯 Ответ на вопрос исследователя готов")
    print("📊 Все визуализации и отчеты сохранены")
    print("="*80)

if __name__ == "__main__":
    main() 