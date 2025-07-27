import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def load_data_by_type(base_path, tree_type, data_type="летние"):
    """Загружает данные определенного типа для конкретного вида дерева"""
    if data_type == "весенние":
        folder_path = os.path.join(base_path, "Спектры, весенний период, 7 видов", tree_type)
    else:  # летние
        folder_path = os.path.join(base_path, tree_type)
    
    data = []
    if os.path.exists(folder_path):
        excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                if df.shape[1] >= 2:
                    spectrum_data = df.iloc[:, 1].values
                    if len(spectrum_data) > 0 and not np.all(np.isnan(spectrum_data)):
                        spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
                        if len(spectrum_data) > 10:
                            data.append(spectrum_data)
            except Exception as e:
                continue
    return data

def analyze_spectral_characteristics():
    """Анализирует спектральные характеристики"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    base_path = "."
    
    # Загружаем все данные
    spring_data = {}
    summer_data = {}
    
    for tree_type in tree_types:
        spring_data[tree_type] = load_data_by_type(base_path, tree_type, "весенние")
        summer_data[tree_type] = load_data_by_type(base_path, tree_type, "летние")
        print(f"{tree_type}: {len(spring_data[tree_type])} весенних, {len(summer_data[tree_type])} летних")
    
    # Находим минимальную длину
    all_spectra = []
    for tree_type in tree_types:
        all_spectra.extend(spring_data[tree_type])
        all_spectra.extend(summer_data[tree_type])
    
    min_length = min(len(spectrum) for spectrum in all_spectra)
    print(f"Минимальная длина спектра: {min_length}")
    
    # Обрезаем до одинаковой длины и создаем датасет
    X_spring = []
    y_spring = []
    X_summer = []
    y_summer = []
    
    for tree_type in tree_types:
        # Весенние данные
        for spectrum in spring_data[tree_type]:
            X_spring.append(spectrum[:min_length])
            y_spring.append(tree_type)
        
        # Летние данные
        for spectrum in summer_data[tree_type]:
            X_summer.append(spectrum[:min_length])
            y_summer.append(tree_type)
    
    X_spring = np.array(X_spring)
    X_summer = np.array(X_summer)
    
    return X_spring, y_spring, X_summer, y_summer, tree_types, min_length

def plot_mean_spectra(X_spring, y_spring, X_summer, y_summer, tree_types):
    """Строит средние спектры для каждого вида"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(tree_types)))
    
    # Весенние спектры
    for i, tree_type in enumerate(tree_types):
        spring_mask = np.array(y_spring) == tree_type
        if np.sum(spring_mask) > 0:
            mean_spectrum = np.mean(X_spring[spring_mask], axis=0)
            std_spectrum = np.std(X_spring[spring_mask], axis=0)
            
            ax1.plot(mean_spectrum, label=f'{tree_type} (n={np.sum(spring_mask)})', 
                    color=colors[i], linewidth=2)
            ax1.fill_between(range(len(mean_spectrum)), 
                           mean_spectrum - std_spectrum, 
                           mean_spectrum + std_spectrum, 
                           alpha=0.2, color=colors[i])
    
    ax1.set_title('Средние спектры - Весенние данные', fontsize=14)
    ax1.set_xlabel('Канал')
    ax1.set_ylabel('Интенсивность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Летние спектры
    for i, tree_type in enumerate(tree_types):
        summer_mask = np.array(y_summer) == tree_type
        if np.sum(summer_mask) > 0:
            mean_spectrum = np.mean(X_summer[summer_mask], axis=0)
            std_spectrum = np.std(X_summer[summer_mask], axis=0)
            
            ax2.plot(mean_spectrum, label=f'{tree_type} (n={np.sum(summer_mask)})', 
                    color=colors[i], linewidth=2)
            ax2.fill_between(range(len(mean_spectrum)), 
                           mean_spectrum - std_spectrum, 
                           mean_spectrum + std_spectrum, 
                           alpha=0.2, color=colors[i])
    
    ax2.set_title('Средние спектры - Летние данные', fontsize=14)
    ax2.set_xlabel('Канал')
    ax2.set_ylabel('Интенсивность')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_comparison_spring_summer.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_problematic_species(X_spring, y_spring, X_summer, y_summer):
    """Детальный анализ проблемных видов (дуб, клен)"""
    problematic = ['дуб', 'клен']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, tree_type in enumerate(problematic):
        # Весенние данные
        spring_mask = np.array(y_spring) == tree_type
        summer_mask = np.array(y_summer) == tree_type
        
        if np.sum(spring_mask) > 0 and np.sum(summer_mask) > 0:
            spring_spectra = X_spring[spring_mask]
            summer_spectra = X_summer[summer_mask]
            
            # Средние спектры
            spring_mean = np.mean(spring_spectra, axis=0)
            summer_mean = np.mean(summer_spectra, axis=0)
            
            axes[idx*2].plot(spring_mean, label=f'Весна (n={len(spring_spectra)})', 
                           color='green', linewidth=2)
            axes[idx*2].plot(summer_mean, label=f'Лето (n={len(summer_spectra)})', 
                           color='orange', linewidth=2)
            axes[idx*2].set_title(f'{tree_type.capitalize()} - Сравнение сезонов')
            axes[idx*2].legend()
            axes[idx*2].grid(True, alpha=0.3)
            
            # Разность спектров
            difference = summer_mean - spring_mean
            axes[idx*2+1].plot(difference, color='red', linewidth=2)
            axes[idx*2+1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[idx*2+1].set_title(f'{tree_type.capitalize()} - Разность (Лето - Весна)')
            axes[idx*2+1].grid(True, alpha=0.3)
            
            # Находим наиболее различающиеся каналы
            abs_diff = np.abs(difference)
            top_channels = np.argsort(abs_diff)[-10:]
            print(f"\n{tree_type.upper()} - Топ-10 наиболее различающихся каналов:")
            for ch in reversed(top_channels):
                print(f"  Канал {ch}: разность = {difference[ch]:.3f}")
    
    plt.tight_layout()
    plt.savefig('problematic_species_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def pca_analysis(X_spring, y_spring, X_summer, y_summer, tree_types):
    """PCA анализ для понимания кластеризации видов"""
    # Объединяем все данные
    X_all = np.vstack([X_spring, X_summer])
    y_all = y_spring + y_summer
    season_all = ['Весна'] * len(y_spring) + ['Лето'] * len(y_summer)
    
    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Создаем DataFrame для удобства
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Вид': y_all,
        'Сезон': season_all
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # График по видам
    for tree_type in tree_types:
        mask = df_pca['Вид'] == tree_type
        axes[0].scatter(df_pca[mask]['PC1'], df_pca[mask]['PC2'], 
                       label=tree_type, alpha=0.6, s=50)
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PCA - По видам деревьев')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График по сезонам
    colors = {'Весна': 'green', 'Лето': 'orange'}
    for season in ['Весна', 'Лето']:
        mask = df_pca['Сезон'] == season
        axes[1].scatter(df_pca[mask]['PC1'], df_pca[mask]['PC2'], 
                       label=season, alpha=0.6, s=50, c=colors[season])
    
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('PCA - По сезонам')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PCA объясняет {sum(pca.explained_variance_ratio_):.1%} дисперсии")
    
    return df_pca

def correlation_analysis(X_spring, y_spring, X_summer, y_summer, tree_types):
    """Анализ корреляции между весенними и летними спектрами одних видов"""
    correlations = {}
    
    for tree_type in tree_types:
        spring_mask = np.array(y_spring) == tree_type
        summer_mask = np.array(y_summer) == tree_type
        
        if np.sum(spring_mask) > 0 and np.sum(summer_mask) > 0:
            spring_mean = np.mean(X_spring[spring_mask], axis=0)
            summer_mean = np.mean(X_summer[summer_mask], axis=0)
            
            corr = np.corrcoef(spring_mean, summer_mean)[0, 1]
            correlations[tree_type] = corr
    
    # Строим график корреляций
    plt.figure(figsize=(10, 6))
    species = list(correlations.keys())
    corr_values = list(correlations.values())
    
    colors = ['red' if species[i] in ['дуб', 'клен'] else 'blue' for i in range(len(species))]
    
    bars = plt.bar(species, corr_values, color=colors, alpha=0.7)
    plt.axhline(y=0.8, color='green', linestyle='--', label='Хорошая корреляция (0.8)')
    plt.axhline(y=0.6, color='orange', linestyle='--', label='Умеренная корреляция (0.6)')
    
    plt.title('Корреляция между весенними и летними спектрами')
    plt.ylabel('Коэффициент корреляции')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, corr_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('seasonal_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nКорреляция между весенними и летними спектрами:")
    for tree_type, corr in sorted(correlations.items(), key=lambda x: x[1]):
        status = "❌ Низкая" if corr < 0.6 else "⚠️ Умеренная" if corr < 0.8 else "✅ Высокая"
        print(f"{tree_type:>8}: {corr:.3f} {status}")
    
    return correlations

def main():
    """Основная функция анализа"""
    print("АНАЛИЗ СПЕКТРАЛЬНЫХ РАЗЛИЧИЙ")
    print("="*60)
    
    # Загружаем данные
    X_spring, y_spring, X_summer, y_summer, tree_types, min_length = analyze_spectral_characteristics()
    
    print(f"\nВсего весенних спектров: {len(X_spring)}")
    print(f"Всего летних спектров: {len(X_summer)}")
    print(f"Длина спектра: {min_length}")
    
    # 1. Сравнение средних спектров
    print("\n1. Построение средних спектров...")
    plot_mean_spectra(X_spring, y_spring, X_summer, y_summer, tree_types)
    
    # 2. Анализ проблемных видов
    print("\n2. Анализ проблемных видов (дуб, клен)...")
    analyze_problematic_species(X_spring, y_spring, X_summer, y_summer)
    
    # 3. PCA анализ
    print("\n3. PCA анализ...")
    df_pca = pca_analysis(X_spring, y_spring, X_summer, y_summer, tree_types)
    
    # 4. Корреляционный анализ
    print("\n4. Корреляционный анализ...")
    correlations = correlation_analysis(X_spring, y_spring, X_summer, y_summer, tree_types)
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*60)
    print("Созданные файлы:")
    print("- spectral_comparison_spring_summer.png")
    print("- problematic_species_analysis.png") 
    print("- pca_analysis.png")
    print("- seasonal_correlation.png")

if __name__ == "__main__":
    main() 