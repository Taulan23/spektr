#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
АНАЛИЗ ФАЙЛА et80_10.xlsx
Сравнение с нашими результатами
"""

import pandas as pd
import numpy as np

def analyze_et80_10():
    """Анализирует файл et80_10.xlsx"""
    
    print("="*80)
    print("📊 АНАЛИЗ ФАЙЛА et80_10.xlsx")
    print("="*80)
    
    # Загружаем файл
    df = pd.read_excel('et80_10.xlsx')
    
    print(f"📋 СТРУКТУРА ФАЙЛА:")
    print(f"   • Размер: {df.shape}")
    print(f"   • Строк: {len(df)}")
    print(f"   • Столбцов: {len(df.columns)}")
    
    # Анализируем заголовки
    print(f"\n📋 ЗАГОЛОВКИ:")
    for i in range(min(10, len(df))):
        if pd.notna(df.iloc[i, 0]):
            print(f"   • Строка {i}: {df.iloc[i, 0]}")
    
    # Ищем начало данных
    data_start = None
    species_names = []
    
    for i in range(len(df)):
        if pd.notna(df.iloc[i, 0]) and 'береза' in str(df.iloc[i, 0]):
            data_start = i
            break
    
    if data_start is None:
        print("❌ Не найдено начало данных!")
        return
    
    print(f"\n📊 НАЧАЛО ДАННЫХ: строка {data_start}")
    
    # Анализируем заголовки столбцов (строка 8)
    if data_start + 1 < len(df):
        headers_row = df.iloc[data_start + 1, 1:].values
        print(f"\n📋 ЗАГОЛОВКИ СТОЛБЦОВ:")
        for i, header in enumerate(headers_row):
            if pd.notna(header):
                print(f"   • Столбец {i+1}: {header}")
                species_names.append(header)
    
    # Анализируем данные классификации
    print(f"\n📊 АНАЛИЗ ДАННЫХ КЛАССИФИКАЦИИ:")
    
    # Считаем образцы для каждого вида
    species_counts = {}
    correct_classifications = {}
    
    current_species = None
    sample_count = 0
    
    for i in range(data_start + 2, len(df)):
        row = df.iloc[i, 1:].values
        
        # Проверяем, есть ли данные в строке
        if not any(pd.notna(val) for val in row):
            continue
        
        # Ищем вид в первом столбце
        first_col = df.iloc[i, 0]
        if pd.notna(first_col) and any(species in str(first_col) for species in species_names):
            # Это новый вид
            if current_species:
                species_counts[current_species] = sample_count
                print(f"   • {current_species}: {sample_count} образцов")
            
            current_species = str(first_col)
            sample_count = 0
            correct_classifications[current_species] = 0
        
        # Анализируем результаты классификации
        if current_species and any(pd.notna(val) for val in row):
            sample_count += 1
            
            # Ищем правильную классификацию (1 в соответствующем столбце)
            species_idx = None
            for j, species in enumerate(species_names):
                if current_species in species:
                    species_idx = j
                    break
            
            if species_idx is not None and species_idx < len(row):
                if pd.notna(row[species_idx]) and row[species_idx] == 1:
                    correct_classifications[current_species] += 1
    
    # Добавляем последний вид
    if current_species:
        species_counts[current_species] = sample_count
        print(f"   • {current_species}: {sample_count} образцов")
    
    # Вычисляем точность
    print(f"\n📈 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
    total_correct = 0
    total_samples = 0
    
    for species in species_counts:
        correct = correct_classifications.get(species, 0)
        total = species_counts[species]
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"   • {species}: {correct}/{total} ({accuracy:.1f}%)")
        
        total_correct += correct
        total_samples += total
    
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)")
    
    # Сравнение с нашими результатами
    print(f"\n📊 СРАВНЕНИЕ С НАШИМИ РЕЗУЛЬТАТАМИ:")
    print(f"   • et80_10.xlsx: {overall_accuracy:.1f}%")
    print(f"   • Наша модель (10% шума): ~90.3%")
    print(f"   • Разница: {overall_accuracy - 90.3:.1f}%")
    
    if overall_accuracy < 90.3:
        print(f"   • Вывод: В файле et80_10.xlsx точность ниже на {90.3 - overall_accuracy:.1f}%")
    else:
        print(f"   • Вывод: В файле et80_10.xlsx точность выше на {overall_accuracy - 90.3:.1f}%")
    
    print("="*80)

if __name__ == "__main__":
    analyze_et80_10() 