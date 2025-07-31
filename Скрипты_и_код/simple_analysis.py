import pandas as pd

# Загружаем файл
df = pd.read_excel('et80_10.xlsx')

print("=== ПОИСК ДАННЫХ В ФАЙЛЕ ===")
print(f"Всего строк: {len(df)}")

# Ищем строки с данными
data_found = False
for i in range(20, min(100, len(df))):
    if pd.notna(df.iloc[i, 0]) and str(df.iloc[i, 0]).strip():
        print(f"Строка {i}: {df.iloc[i, 0]}")
        data_found = True

if not data_found:
    print("Данные не найдены в первых 100 строках")

# Проверим последние строки
print("\n=== ПОСЛЕДНИЕ 10 СТРОК ===")
for i in range(max(0, len(df)-10), len(df)):
    if pd.notna(df.iloc[i, 0]):
        print(f"Строка {i}: {df.iloc[i, 0]}")

# Проверим, есть ли единицы в данных
print("\n=== ПОИСК ЕДИНИЦ В ДАННЫХ ===")
ones_found = 0
for i in range(len(df)):
    for j in range(len(df.columns)):
        if pd.notna(df.iloc[i, j]) and df.iloc[i, j] == 1:
            ones_found += 1
            if ones_found <= 10:  # Показываем первые 10
                print(f"Единица найдена: строка {i}, столбец {j}")

print(f"Всего единиц найдено: {ones_found}") 