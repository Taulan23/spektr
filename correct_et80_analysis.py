#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛЬНЫЙ АНАЛИЗ ФАЙЛА et80_10.xlsx
Файл содержит только один вид "береза"
"""

import pandas as pd
import numpy as np

def correct_et80_analysis():
    """Правильный анализ файла et80_10.xlsx"""
    
    print("="*80)
    print("📊 ПРАВИЛЬНЫЙ АНАЛИЗ ФАЙЛА et80_10.xlsx")
    print("="*80)
    
    # Загружаем файл
    df = pd.read_excel('et80_10.xlsx')
    
    print(f"📋 СТРУКТУРА ФАЙЛА:")
    print(f"   • Размер: {df.shape}")
    print(f"   • Строк: {len(df)}")
    print(f"   • Столбцов: {len(df.columns)}")
    
    # Заголовки столбцов (строка 8)
    headers = df.iloc[8, 1:].values
    print(f"\n📋 ЗАГОЛОВКИ СТОЛБЦОВ:")
    for i, header in enumerate(headers):
        if pd.notna(header):
            print(f"   • Столбец {i+1}: {header}")
    
    # Данные начинаются со строки 9
    data_start = 9
    
    # Считаем общее количество образцов
    total_samples = 0
    for i in range(data_start, len(df)):
        if any(pd.notna(val) for val in df.iloc[i, 1:].values):
            total_samples += 1
    
    print(f"\n📊 АНАЛИЗ ДАННЫХ:")
    print(f"   • Всего образцов: {total_samples}")
    print(f"   • Вид: береза (единственный в файле)")
    
    # Анализируем правильные классификации
    print(f"\n📈 АНАЛИЗ ПРАВИЛЬНЫХ КЛАССИФИКАЦИЙ:")
    
    # Ищем столбец "ПО береза" (правильное обнаружение)
    po_beresa_col = None
    for j in range(1, len(df.columns)):
        if pd.notna(headers[j-1]) and "ПО береза" in str(headers[j-1]):
            po_beresa_col = j
            break
    
    if po_beresa_col is None:
        print("❌ Не найден столбец 'ПО береза'")
        return
    
    # Считаем правильные классификации березы
    correct_beresa = 0
    for i in range(data_start, len(df)):
        if pd.notna(df.iloc[i, po_beresa_col]) and df.iloc[i, po_beresa_col] == 1:
            correct_beresa += 1
    
    accuracy_beresa = correct_beresa / total_samples * 100 if total_samples > 0 else 0
    
    print(f"   • Правильные классификации березы: {correct_beresa}/{total_samples}")
    print(f"   • Точность классификации березы: {accuracy_beresa:.1f}%")
    
    # Анализируем ложные тревоги
    print(f"\n🚨 АНАЛИЗ ЛОЖНЫХ ТРЕВОГ:")
    
    false_alarms = {}
    for j in range(1, len(df.columns)):
        if pd.notna(headers[j-1]) and "ЛТ" in str(headers[j-1]):
            species_name = str(headers[j-1]).replace("ЛТ ", "")
            false_alarm_count = 0
            
            for i in range(data_start, len(df)):
                if pd.notna(df.iloc[i, j]) and df.iloc[i, j] == 1:
                    false_alarm_count += 1
            
            false_alarms[species_name] = false_alarm_count
            false_alarm_rate = false_alarm_count / total_samples * 100 if total_samples > 0 else 0
            print(f"   • {species_name}: {false_alarm_count} ложных тревог ({false_alarm_rate:.1f}%)")
    
    # Общая статистика
    total_false_alarms = sum(false_alarms.values())
    total_false_alarm_rate = total_false_alarms / total_samples * 100 if total_samples > 0 else 0
    
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   • Правильные классификации: {correct_beresa}")
    print(f"   • Ложные тревоги: {total_false_alarms}")
    print(f"   • Общая точность: {accuracy_beresa:.1f}%")
    print(f"   • Общая частота ложных тревог: {total_false_alarm_rate:.1f}%")
    
    # Сравнение с нашими результатами
    print(f"\n📊 СРАВНЕНИЕ С НАШИМИ РЕЗУЛЬТАТАМИ:")
    print(f"   • et80_10.xlsx (береза): {accuracy_beresa:.1f}%")
    print(f"   • Наша модель (береза, 10% шума): ~96.7% (29/30)")
    print(f"   • Разница: {accuracy_beresa - 96.7:.1f}%")
    
    if accuracy_beresa < 96.7:
        print(f"   • Вывод: В файле et80_10.xlsx точность НИЖЕ на {96.7 - accuracy_beresa:.1f}%")
        print(f"   • Это подтверждает ваше наблюдение о падении точности!")
    else:
        print(f"   • Вывод: В файле et80_10.xlsx точность выше на {accuracy_beresa - 96.7:.1f}%")
    
    # Анализ основных ошибок
    print(f"\n🔍 АНАЛИЗ ОСНОВНЫХ ОШИБОК:")
    sorted_false_alarms = sorted(false_alarms.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   • Топ-5 ложных тревог:")
    for i, (species, count) in enumerate(sorted_false_alarms[:5]):
        rate = count / total_samples * 100 if total_samples > 0 else 0
        print(f"     {i+1}. {species}: {count} ({rate:.1f}%)")
    
    print("="*80)

if __name__ == "__main__":
    correct_et80_analysis() 