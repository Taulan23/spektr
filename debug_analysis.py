import pandas as pd

# Загружаем файл
df = pd.read_excel('et80_10.xlsx')

print("=== ОТЛАДОЧНЫЙ АНАЛИЗ ===")

# Посмотрим на строки 8-15 более детально
print("Строки 8-15:")
for i in range(8, 16):
    print(f"Строка {i}: {df.iloc[i, 0]} | {df.iloc[i, 1:5].tolist()}")

# Посмотрим на строки с единицами
print("\nСтроки с единицами (первые 20):")
ones_count = 0
for i in range(len(df)):
    for j in range(len(df.columns)):
        if pd.notna(df.iloc[i, j]) and df.iloc[i, j] == 1:
            ones_count += 1
            if ones_count <= 20:
                print(f"Единица {ones_count}: строка {i}, столбец {j} | {df.iloc[i, 0]}")

print(f"\nВсего единиц: {ones_count}")

# Посмотрим на строки с названиями видов
print("\nСтроки с названиями видов:")
for i in range(len(df)):
    if pd.notna(df.iloc[i, 0]) and any(species in str(df.iloc[i, 0]) for species in ['береза', 'дуб', 'ель', 'клен', 'сосна']):
        print(f"Строка {i}: {df.iloc[i, 0]}") 