#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНЫЙ АНАЛИЗ ФАЙЛА et80_10.xlsx
"""

import pandas as pd
import numpy as np

def analyze_et80_10_complete():
    """Полный анализ файла et80_10.xlsx"""
    
    print("="*80)
    print("📊 ПОЛНЫЙ АНАЛИЗ ФАЙЛА et80_10.xlsx")
    print("="*80)
    
    # Загружаем файл
    df = pd.read_excel('et80_10.xlsx')
    
    print(f"📋 СТРУКТУРА ФАЙЛА:")
    print(f"   • Размер: {df.shape}")
    print(f"   • Строк: {len(df)}")
    print(f"   • Столбцов: {len(df.columns)}")
    
    # Анализируем заголовки
    print(f"\n📋 ЗАГОЛОВКИ ФАЙЛА:")
    for i in range(min(10, len(df))):
        if pd.notna(df.iloc[i, 0]):
            print(f"   • Строка {i}: {df.iloc[i, 0]}")
    
    # Заголовки столбцов (строка 8)
    headers = df.iloc[8, 1:].values
    print(f"\n📋 ЗАГОЛОВКИ СТОЛБЦОВ:")
    for i, header in enumerate(headers):
        if pd.notna(header):
            print(f"   • Столбец {i+1}: {header}")
    
    # Анализируем данные
    print(f"\n📊 АНАЛИЗ ДАННЫХ:")
    
    # Данные начинаются со строки 9
    data_start = 9
    
    # Считаем общее количество образцов
    total_samples = 0
    for i in range(data_start, len(df)):
        if any(pd.notna(val) for val in df.iloc[i, 1:].values):
            total_samples += 1
    
    print(f"   • Всего образцов: {total_samples}")
    
    # Анализируем правильные классификации
    print(f"\n📈 АНАЛИЗ ПРАВИЛЬНЫХ КЛАССИФИКАЦИЙ:")
    
    # Считаем единицы в каждом столбце
    correct_by_column = {}
    for j in range(1, len(df.columns)):
        column_name = headers[j-1] if pd.notna(headers[j-1]) else f"Столбец {j}"
        ones_count = 0
        
        for i in range(data_start, len(df)):
            if pd.notna(df.iloc[i, j]) and df.iloc[i, j] == 1:
                ones_count += 1
        
        correct_by_column[column_name] = ones_count
        print(f"   • {column_name}: {ones_count} правильных классификаций")
    
    # Вычисляем общую точность
    total_correct = sum(correct_by_column.values())
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    
    print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {total_correct}/{total_samples} ({overall_accuracy:.1f}%)")
    
    # Сравнение с нашими результатами
    print(f"\n📊 СРАВНЕНИЕ С НАШИМИ РЕЗУЛЬТАТАМИ:")
    print(f"   • et80_10.xlsx: {overall_accuracy:.1f}%")
    print(f"   • Наша модель (10% шума): ~90.3%")
    print(f"   • Разница: {overall_accuracy - 90.3:.1f}%")
    
    if overall_accuracy < 90.3:
        print(f"   • Вывод: В файле et80_10.xlsx точность НИЖЕ на {90.3 - overall_accuracy:.1f}%")
        print(f"   • Это подтверждает ваше наблюдение о большом падении точности!")
    else:
        print(f"   • Вывод: В файле et80_10.xlsx точность выше на {overall_accuracy - 90.3:.1f}%")
    
    # Анализируем структуру данных
    print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ СТРУКТУРЫ:")
    
    # Проверим, есть ли названия видов в первом столбце
    species_in_first_col = []
    for i in range(data_start, len(df)):
        if pd.notna(df.iloc[i, 0]) and str(df.iloc[i, 0]).strip():
            species_in_first_col.append(str(df.iloc[i, 0]))
    
    print(f"   • Названия видов в первом столбце: {len(species_in_first_col)}")
    if species_in_first_col:
        print(f"   • Виды: {species_in_first_col[:5]}...")  # Показываем первые 5
    
    # Проверим распределение единиц по столбцам
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ЕДИНИЦ ПО СТОЛБЦАМ:")
    for column_name, count in correct_by_column.items():
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"   • {column_name}: {count} ({percentage:.1f}%)")
    
    print("="*80)

if __name__ == "__main__":
    analyze_et80_10_complete() 