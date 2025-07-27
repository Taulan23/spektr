import pandas as pd
import numpy as np

# Загружаем файл
df = pd.read_excel('et80_10.xlsx')

print("="*80)
print("📊 АНАЛИЗ ФАЙЛА et80_10.xlsx")
print("="*80)

# Данные начинаются со строки 9
data_start = 9

# Заголовки столбцов (строка 8)
headers = df.iloc[8, 1:].values
print("📋 ЗАГОЛОВКИ СТОЛБЦОВ:")
for i, header in enumerate(headers):
    if pd.notna(header):
        print(f"   • Столбец {i+1}: {header}")

# Анализируем данные
print(f"\n📊 АНАЛИЗ ДАННЫХ (строки {data_start}-{len(df)})")

# Считаем образцы для каждого вида
species_data = {}
current_species = None
sample_count = 0

for i in range(data_start, len(df)):
    row = df.iloc[i, 1:].values
    
    # Проверяем первый столбец на наличие названия вида
    first_col = df.iloc[i, 0]
    if pd.notna(first_col) and str(first_col).strip():
        # Это новый вид
        if current_species:
            species_data[current_species] = sample_count
            print(f"   • {current_species}: {sample_count} образцов")
        
        current_species = str(first_col)
        sample_count = 0
    
    # Считаем образцы
    if any(pd.notna(val) for val in row):
        sample_count += 1

# Добавляем последний вид
if current_species:
    species_data[current_species] = sample_count
    print(f"   • {current_species}: {sample_count} образцов")

# Теперь анализируем правильные классификации
print(f"\n📈 АНАЛИЗ ПРАВИЛЬНЫХ КЛАССИФИКАЦИЙ:")

correct_classifications = {}
total_correct = 0
total_samples = 0

for species in species_data:
    correct_count = 0
    sample_count = species_data[species]
    
    # Ищем индекс столбца для этого вида
    species_col_idx = None
    for i, header in enumerate(headers):
        if pd.notna(header) and species in str(header):
            species_col_idx = i + 1  # +1 потому что индексация с 0
            break
    
    if species_col_idx is not None:
        # Считаем правильные классификации
        for i in range(data_start, len(df)):
            if pd.notna(df.iloc[i, 0]) and str(df.iloc[i, 0]).strip() == species:
                # Это образец данного вида
                if pd.notna(df.iloc[i, species_col_idx]) and df.iloc[i, species_col_idx] == 1:
                    correct_count += 1
    
    correct_classifications[species] = correct_count
    accuracy = correct_count / sample_count * 100 if sample_count > 0 else 0
    
    print(f"   • {species}: {correct_count}/{sample_count} ({accuracy:.1f}%)")
    
    total_correct += correct_count
    total_samples += sample_count

# Общая точность
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

print("="*80) 