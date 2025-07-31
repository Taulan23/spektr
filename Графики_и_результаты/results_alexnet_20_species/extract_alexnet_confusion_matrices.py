#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ИЗВЛЕЧЕНИЕ ТОЧНЫХ CONFUSION MATRICES ДЛЯ 1D ALEXNET (20 ВИДОВ)
"""

import numpy as np
import pandas as pd
from datetime import datetime

def extract_confusion_matrices():
    """Извлекает точные числовые данные confusion matrices для всех уровней шума"""
    
    # Виды деревьев (в том же порядке что в модели)
    species_names = [
        'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 'клен_ам',
        'липа', 'лиственница', 'орех', 'осина', 'рябина', 'сирень', 'сосна',
        'тополь_бальзамический', 'тополь_черный', 'туя', 'черемуха', 'ясень'
    ]
    
    # Результаты по видам для каждого уровня шума (из alexnet_20_noise_analysis_report)
    species_accuracies = {
        0: [1.000, 1.000, 1.000, 1.000, 0.933, 1.000, 1.000, 1.000, 1.000, 1.000, 
            1.000, 1.000, 0.967, 1.000, 1.000, 1.000, 1.000, 1.000, 0.967, 1.000],
        1: [1.000, 1.000, 1.000, 1.000, 0.867, 1.000, 1.000, 1.000, 1.000, 1.000, 
            0.867, 1.000, 0.867, 1.000, 1.000, 1.000, 0.933, 1.000, 0.900, 1.000],
        5: [0.767, 0.700, 0.900, 1.000, 0.433, 0.533, 1.000, 0.500, 0.267, 0.867, 
            0.333, 0.467, 0.200, 0.867, 0.967, 0.633, 0.333, 0.933, 0.467, 0.800],
        10: [0.700, 0.300, 0.400, 1.000, 0.200, 0.233, 0.733, 0.067, 0.067, 0.167, 
             0.100, 0.033, 0.033, 0.567, 0.267, 0.767, 0.067, 0.500, 0.067, 0.467],
        20: [0.667, 0.000, 0.000, 0.400, 0.367, 0.233, 0.000, 0.000, 0.000, 0.000, 
             0.000, 0.000, 0.000, 0.467, 0.000, 0.100, 0.000, 0.000, 0.000, 0.233]
    }
    
    # Общие точности
    general_accuracies = {
        0: 0.993,
        1: 0.972,
        5: 0.648,
        10: 0.337,
        20: 0.123
    }
    
    print("📊 ТОЧНЫЕ ЧИСЛОВЫЕ ДАННЫЕ 1D ALEXNET (20 ВИДОВ)")
    print("=" * 80)
    
    # Создаем DataFrame для удобства
    df_data = []
    
    for noise_level in [0, 1, 5, 10, 20]:
        print(f"\n🔊 УРОВЕНЬ ШУМА: {noise_level}%")
        print(f"Общая точность: {general_accuracies[noise_level]:.3f} ({general_accuracies[noise_level]*100:.1f}%)")
        print("-" * 50)
        
        print("Правильная классификация по видам:")
        for i, species in enumerate(species_names):
            accuracy = species_accuracies[noise_level][i]
            print(f"  {species:25}: {accuracy:.3f}")
            
            # Добавляем в DataFrame
            df_data.append({
                'noise_level': f'{noise_level}%',
                'species': species,
                'accuracy': accuracy
            })
    
    # Создаем сводную таблицу
    df = pd.DataFrame(df_data)
    pivot_table = df.pivot(index='species', columns='noise_level', values='accuracy')
    
    print("\n" + "=" * 80)
    print("📋 СВОДНАЯ ТАБЛИЦА ТОЧНОСТЕЙ ПО ВИДАМ")
    print("=" * 80)
    print(pivot_table.round(3))
    
    # Сохраняем в CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'alexnet_20_species_detailed_results_{timestamp}.csv'
    pivot_table.to_csv(csv_filename)
    
    # Создаем детальный текстовый отчет
    txt_filename = f'alexnet_20_species_confusion_data_{timestamp}.txt'
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("ТОЧНЫЕ ЧИСЛОВЫЕ ДАННЫЕ 1D ALEXNET (20 ВИДОВ)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ОБЩИЕ ТОЧНОСТИ ПО УРОВНЯМ ШУМА:\n")
        f.write("-" * 40 + "\n")
        for noise_level in [0, 1, 5, 10, 20]:
            f.write(f"{noise_level:2}% шума: {general_accuracies[noise_level]:.4f} ({general_accuracies[noise_level]*100:.1f}%)\n")
        
        f.write("\n\nДЕТАЛИЗАЦИЯ ПО ВИДАМ:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Вид':25} | {'0%':8} | {'1%':8} | {'5%':8} | {'10%':8} | {'20%':8}\n")
        f.write("-" * 80 + "\n")
        
        for i, species in enumerate(species_names):
            line = f"{species:25} |"
            for noise_level in [0, 1, 5, 10, 20]:
                accuracy = species_accuracies[noise_level][i]
                line += f" {accuracy:7.3f} |"
            f.write(line + "\n")
        
        f.write("\n\nАНАЛИЗ ДЕГРАДАЦИИ:\n")
        f.write("-" * 40 + "\n")
        for i, species in enumerate(species_names):
            degradation = species_accuracies[0][i] - species_accuracies[20][i]
            status = "УСТОЙЧИВ" if degradation < 0.5 else "УМЕРЕННО" if degradation < 0.8 else "ЧУВСТВИТЕЛЕН"
            f.write(f"{species:25}: деградация {degradation:.3f} ({status})\n")
    
    print(f"\n💾 ФАЙЛЫ СОХРАНЕНЫ:")
    print(f"📊 CSV таблица: {csv_filename}")
    print(f"📋 Текстовый отчет: {txt_filename}")
    
    return pivot_table, csv_filename, txt_filename

def create_comparison_table():
    """Создает таблицу для сравнения с Extra Trees"""
    
    print("\n" + "🔬" * 30)
    print("🔬 ПОДГОТОВКА К СРАВНЕНИЮ С EXTRA TREES")
    print("🔬" * 30)
    
    # Данные Alexnet на 20% шума (ключевые виды)
    alexnet_20_results = {
        'береза': 0.667,
        'дуб': 0.000,
        'ель': 0.000,
        'ель_голубая': 0.400,
        'ива': 0.367,
        'каштан': 0.233,
        'клен': 0.000,
        'клен_ам': 0.000,
        'липа': 0.000,
        'лиственница': 0.000,
        'орех': 0.000,
        'осина': 0.000,
        'рябина': 0.000,
        'сирень': 0.467,
        'сосна': 0.000,
        'тополь_бальзамический': 0.100,
        'тополь_черный': 0.000,
        'туя': 0.000,
        'черемуха': 0.000,
        'ясень': 0.233
    }
    
    print("📊 1D ALEXNET НА 20% ШУМА:")
    print("-" * 40)
    
    working_species = []
    failed_species = []
    
    for species, accuracy in alexnet_20_results.items():
        print(f"  {species:25}: {accuracy:.3f}")
        if accuracy > 0.3:
            working_species.append(species)
        else:
            failed_species.append(species)
    
    print(f"\n✅ ВИДЫ С ПРИЕМЛЕМОЙ ТОЧНОСТЬЮ (>30%): {len(working_species)}")
    for species in working_species:
        print(f"   • {species}: {alexnet_20_results[species]:.3f}")
    
    print(f"\n❌ ВИДЫ С КРИТИЧНО НИЗКОЙ ТОЧНОСТЬЮ (<30%): {len(failed_species)}")
    for species in failed_species:
        print(f"   • {species}: {alexnet_20_results[species]:.3f}")
    
    print(f"\n📈 ОБЩАЯ ТОЧНОСТЬ ALEXNET НА 20% ШУМА: 12.3%")
    print(f"🎯 Для сравнения нужны результаты Extra Trees на том же уровне шума")
    
    return alexnet_20_results

if __name__ == "__main__":
    print("🌲 ИЗВЛЕЧЕНИЕ ДАННЫХ 1D ALEXNET")
    print("=" * 50)
    
    # Извлекаем данные
    pivot_table, csv_file, txt_file = extract_confusion_matrices()
    
    # Готовим к сравнению
    alexnet_results = create_comparison_table()
    
    print(f"\n🎯 ГОТОВО! Данные извлечены и готовы для анализа!") 