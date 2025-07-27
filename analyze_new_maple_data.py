import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_original_maple_data():
    """Загружает исходные данные клена"""
    spring_path = "Спектры, весенний период, 7 видов/клен"
    summer_path = "клен"
    
    spring_data = []
    summer_data = []
    
    # Весенние данные
    if os.path.exists(spring_path):
        excel_files = glob.glob(os.path.join(spring_path, '*.xlsx'))
        print(f"Найдено {len(excel_files)} весенних файлов клена (оригинальные)")
        
        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                if df.shape[1] >= 2:
                    spectrum_data = df.iloc[:, 1].values
                    if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                        spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                        if len(spectrum_data) > 10:
                            spring_data.append(spectrum_data)
            except Exception as e:
                continue
    
    # Летние данные
    if os.path.exists(summer_path):
        excel_files = glob.glob(os.path.join(summer_path, '*.xlsx'))
        print(f"Найдено {len(excel_files)} летних файлов клена (оригинальные)")
        
        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                if df.shape[1] >= 2:
                    spectrum_data = df.iloc[:, 1].values
                    if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                        spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                        if len(spectrum_data) > 10:
                            summer_data.append(spectrum_data)
            except Exception as e:
                continue
    
    return spring_data, summer_data

def load_new_maple_data():
    """Загружает новые данные клена из папки клен_ам"""
    new_path = "клен_ам"
    new_data = []
    
    if os.path.exists(new_path):
        excel_files = glob.glob(os.path.join(new_path, '*.xlsx'))
        print(f"Найдено {len(excel_files)} новых файлов клена (клен_ам)")
        
        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                if df.shape[1] >= 2:
                    spectrum_data = df.iloc[:, 1].values
                    if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                        spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                        if len(spectrum_data) > 10:
                            new_data.append(spectrum_data)
            except Exception as e:
                print(f"Ошибка при чтении {file_path}: {e}")
                continue
    
    return new_data

def compare_maple_datasets():
    """Сравнивает разные наборы данных клена"""
    print("АНАЛИЗ ДАННЫХ КЛЕНА")
    print("="*60)
    
    # Загружаем все данные
    spring_original, summer_original = load_original_maple_data()
    new_maple = load_new_maple_data()
    
    print(f"Весенние данные (оригинальные): {len(spring_original)} спектров")
    print(f"Летние данные (оригинальные): {len(summer_original)} спектров")
    print(f"Новые данные (клен_ам): {len(new_maple)} спектров")
    
    # Находим минимальную длину
    all_spectra = spring_original + summer_original + new_maple
    if not all_spectra:
        print("Нет данных для анализа!")
        return
    
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    # Обрезаем до одинаковой длины
    spring_trimmed = np.array([spectrum[:min_length] for spectrum in spring_original])
    summer_trimmed = np.array([spectrum[:min_length] for spectrum in summer_original])
    new_trimmed = np.array([spectrum[:min_length] for spectrum in new_maple])
    
    # Создаем объединенный датасет
    X_all = []
    labels = []
    
    if len(spring_trimmed) > 0:
        X_all.extend(spring_trimmed)
        labels.extend(['Клен весна'] * len(spring_trimmed))
    
    if len(summer_trimmed) > 0:
        X_all.extend(summer_trimmed)
        labels.extend(['Клен лето'] * len(summer_trimmed))
    
    if len(new_trimmed) > 0:
        X_all.extend(new_trimmed)
        labels.extend(['Клен новый'] * len(new_trimmed))
    
    X_all = np.array(X_all)
    
    return X_all, labels, spring_trimmed, summer_trimmed, new_trimmed

def plot_maple_comparison(spring_data, summer_data, new_data):
    """Строит сравнительные графики для разных наборов данных клена"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    datasets = [
        ('Клен - Весна (оригинальные)', spring_data, 'green'),
        ('Клен - Лето (оригинальные)', summer_data, 'orange'),
        ('Клен - Новые данные (клен_ам)', new_data, 'red')
    ]
    
    # 1. Средние спектры
    for i, (label, data, color) in enumerate(datasets):
        if len(data) > 0:
            mean_spectrum = np.mean(data, axis=0)
            std_spectrum = np.std(data, axis=0)
            
            axes[0].plot(mean_spectrum, label=f'{label} (n={len(data)})', 
                        color=color, linewidth=2)
            axes[0].fill_between(range(len(mean_spectrum)), 
                               mean_spectrum - std_spectrum, 
                               mean_spectrum + std_spectrum, 
                               alpha=0.2, color=color)
    
    axes[0].set_title('Средние спектры клена - сравнение наборов данных')
    axes[0].set_xlabel('Канал')
    axes[0].set_ylabel('Интенсивность')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Различия между наборами
    if len(summer_data) > 0 and len(new_data) > 0:
        summer_mean = np.mean(summer_data, axis=0)
        new_mean = np.mean(new_data, axis=0)
        difference = new_mean - summer_mean
        
        axes[1].plot(difference, color='purple', linewidth=2, label='Новые - Летние')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('Разность: Новые данные клена - Летние данные')
        axes[1].set_xlabel('Канал')
        axes[1].set_ylabel('Разность интенсивности')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Находим наиболее различающиеся каналы
        abs_diff = np.abs(difference)
        top_channels = np.argsort(abs_diff)[-10:]
        print("\nТоп-10 наиболее различающихся каналов (новые vs летние клен):")
        for ch in reversed(top_channels):
            print(f"  Канал {ch}: разность = {difference[ch]:.3f}")
    
    # 3. Стандартные отклонения
    for i, (label, data, color) in enumerate(datasets):
        if len(data) > 0:
            std_spectrum = np.std(data, axis=0)
            axes[2].plot(std_spectrum, label=label, color=color, linewidth=2)
    
    axes[2].set_title('Стандартные отклонения по каналам')
    axes[2].set_xlabel('Канал')
    axes[2].set_ylabel('Стандартное отклонение')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Распределение интенсивностей
    for i, (label, data, color) in enumerate(datasets):
        if len(data) > 0:
            mean_intensities = np.mean(data, axis=1)
            axes[3].hist(mean_intensities, bins=20, alpha=0.6, 
                        label=label, color=color, density=True)
    
    axes[3].set_title('Распределение средних интенсивностей')
    axes[3].set_xlabel('Средняя интенсивность')
    axes[3].set_ylabel('Плотность')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('maple_datasets_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def pca_maple_analysis(X_all, labels):
    """PCA анализ для разных наборов данных клена"""
    if len(X_all) == 0:
        print("Нет данных для PCA анализа")
        return
    
    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    
    unique_labels = list(set(labels))
    colors = ['green', 'orange', 'red']
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=f'{label} (n={np.sum(mask)})', 
                   alpha=0.7, s=60, c=colors[i % len(colors)])
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('PCA - Сравнение разных наборов данных клена')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('maple_pca_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PCA объясняет {sum(pca.explained_variance_ratio_):.1%} дисперсии")

def analyze_maple_correlations(spring_data, summer_data, new_data):
    """Анализирует корреляции между разными наборами данных клена"""
    correlations = {}
    
    if len(spring_data) > 0 and len(summer_data) > 0:
        spring_mean = np.mean(spring_data, axis=0)
        summer_mean = np.mean(summer_data, axis=0)
        corr_spring_summer = np.corrcoef(spring_mean, summer_mean)[0, 1]
        correlations['Весна ↔ Лето'] = corr_spring_summer
    
    if len(summer_data) > 0 and len(new_data) > 0:
        summer_mean = np.mean(summer_data, axis=0)
        new_mean = np.mean(new_data, axis=0)
        corr_summer_new = np.corrcoef(summer_mean, new_mean)[0, 1]
        correlations['Лето ↔ Новые'] = corr_summer_new
    
    if len(spring_data) > 0 and len(new_data) > 0:
        spring_mean = np.mean(spring_data, axis=0)
        new_mean = np.mean(new_data, axis=0)
        corr_spring_new = np.corrcoef(spring_mean, new_mean)[0, 1]
        correlations['Весна ↔ Новые'] = corr_spring_new
    
    print("\nКорреляции между наборами данных клена:")
    for pair, corr in correlations.items():
        status = "❌ Низкая" if corr < 0.6 else "⚠️ Умеренная" if corr < 0.8 else "✅ Высокая"
        print(f"{pair:>15}: {corr:.3f} {status}")
    
    return correlations

def main():
    """Основная функция анализа"""
    print("АНАЛИЗ НОВЫХ ДАННЫХ КЛЕНА")
    print("="*60)
    
    # Сравниваем наборы данных
    X_all, labels, spring_data, summer_data, new_data = compare_maple_datasets()
    
    if len(X_all) == 0:
        print("Не удалось загрузить данные!")
        return
    
    # Строим сравнительные графики
    print("\n1. Построение сравнительных графиков...")
    plot_maple_comparison(spring_data, summer_data, new_data)
    
    # PCA анализ
    print("\n2. PCA анализ...")
    pca_maple_analysis(X_all, labels)
    
    # Корреляционный анализ
    print("\n3. Корреляционный анализ...")
    correlations = analyze_maple_correlations(spring_data, summer_data, new_data)
    
    # Статистика
    print("\n" + "="*60)
    print("СТАТИСТИКА ДАННЫХ")
    print("="*60)
    
    datasets_info = [
        ("Весенние (оригинальные)", spring_data),
        ("Летние (оригинальные)", summer_data),
        ("Новые (клен_ам)", new_data)
    ]
    
    for name, data in datasets_info:
        if len(data) > 0:
            mean_values = [np.mean(spectrum) for spectrum in data]
            std_values = [np.std(spectrum) for spectrum in data]
            
            print(f"\n{name}:")
            print(f"  Количество спектров: {len(data)}")
            print(f"  Средняя интенсивность: {np.mean(mean_values):.3f} ± {np.std(mean_values):.3f}")
            print(f"  Средняя вариабельность: {np.mean(std_values):.3f} ± {np.std(std_values):.3f}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*60)
    print("Созданные файлы:")
    print("- maple_datasets_comparison.png")
    print("- maple_pca_comparison.png")

if __name__ == "__main__":
    main() 