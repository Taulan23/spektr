import os
import glob
import pandas as pd
import numpy as np

def test_data_structure():
    """Тестирует структуру данных"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    total_files = 0
    total_spectra = 0
    
    print("Проверка структуры данных...")
    print("="*50)
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            valid_files = 0
            
            print(f"\n{tree_type.upper()}:")
            print(f"  Папка: {folder_path}")
            print(f"  Найдено файлов: {len(excel_files)}")
            
            spectrum_lengths = []
            
            for file_path in excel_files[:5]:  # Проверяем первые 5 файлов
                try:
                    df = pd.read_excel(file_path)
                    
                    if df.shape[1] >= 2:
                        spectrum_data = df.iloc[:, 1].values
                        # Удаляем NaN
                        spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                        
                        if len(spectrum_data) > 10:
                            valid_files += 1
                            spectrum_lengths.append(len(spectrum_data))
                            total_spectra += 1
                            
                            print(f"    ✓ {os.path.basename(file_path)}: {len(spectrum_data)} точек")
                        else:
                            print(f"    ✗ {os.path.basename(file_path)}: слишком мало данных ({len(spectrum_data)} точек)")
                    else:
                        print(f"    ✗ {os.path.basename(file_path)}: недостаточно столбцов ({df.shape[1]})")
                        
                except Exception as e:
                    print(f"    ✗ {os.path.basename(file_path)}: ошибка - {e}")
            
            if spectrum_lengths:
                print(f"  Валидных файлов: {valid_files}")
                print(f"  Длина спектров: {min(spectrum_lengths)} - {max(spectrum_lengths)}")
                print(f"  Средняя длина: {np.mean(spectrum_lengths):.1f}")
            
            total_files += len(excel_files)
            
        else:
            print(f"\n{tree_type.upper()}:")
            print(f"  ✗ Папка не найдена: {folder_path}")
    
    print("\n" + "="*50)
    print("ИТОГО:")
    print(f"  Всего файлов: {total_files}")
    print(f"  Валидных спектров: {total_spectra}")
    
    if total_spectra < 50:
        print("\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Малое количество данных для обучения!")
        print("   Рекомендуется минимум 50 спектров для надежной классификации.")
    
    return total_spectra > 0

def check_sample_spectrum():
    """Проверяет образец спектра"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    
    print("\n" + "="*50)
    print("АНАЛИЗ ОБРАЗЦА СПЕКТРА:")
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            if excel_files:
                try:
                    df = pd.read_excel(excel_files[0])
                    print(f"\nПример файла: {excel_files[0]}")
                    print(f"Размер DataFrame: {df.shape}")
                    print(f"Столбцы: {list(df.columns)}")
                    
                    if df.shape[1] >= 2:
                        print(f"Первые 5 значений спектра:")
                        spectrum = df.iloc[:5, 1].values
                        for i, val in enumerate(spectrum):
                            print(f"  Точка {i+1}: {val}")
                    
                    return True
                    
                except Exception as e:
                    print(f"Ошибка при анализе {excel_files[0]}: {e}")
    
    return False

def main():
    """Основная функция тестирования"""
    print("🔍 ТЕСТИРОВАНИЕ ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ СПЕКТРОВ ДЕРЕВЬЕВ")
    print("="*60)
    
    # Проверяем структуру
    data_ok = test_data_structure()
    
    if data_ok:
        # Анализируем образец
        check_sample_spectrum()
        
        print("\n" + "="*60)
        print("✅ ТЕСТ ПРОЙДЕН: Данные готовы для обучения!")
        print("   Запустите: python main.py")
    else:
        print("\n" + "="*60)
        print("❌ ТЕСТ НЕ ПРОЙДЕН: Проблемы с данными!")
        print("   Проверьте структуру папок и Excel файлы.")

if __name__ == "__main__":
    main() 