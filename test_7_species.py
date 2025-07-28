import glob
import os
import pandas as pd

print("Начинаем тест загрузки данных...")

species_folders = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
data = []
labels = []

for species in species_folders:
    print(f"Загрузка данных для {species}...")
    folder_path = f'Спектры, весенний период, 7 видов/{species}'
    
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не найдена!")
        continue
        
    files = glob.glob(f'{folder_path}/*.xlsx')
    print(f"Найдено {len(files)} файлов")
    files = files[:30]  # Берем первые 30 файлов
    
    for file in files:
        try:
            df = pd.read_excel(file)
            spectral_data = df.iloc[:, 1:].values.flatten()
            
            if len(spectral_data) > 0:
                data.append(spectral_data)
                labels.append(species)
                print(f"Загружен файл: {file}")
        except Exception as e:
            print(f"Ошибка при чтении файла {file}: {e}")
            continue

print(f"Всего загружено {len(data)} спектров")
print("Тест завершен!") 