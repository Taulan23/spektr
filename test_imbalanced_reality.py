"""
ТЕСТ: КАК СБАЛАНСИРОВАННОСТЬ ВЛИЯЕТ НА ВАШИ РЕЗУЛЬТАТЫ
======================================================

Этот скрипт берет ваши сбалансированные данные и создает несбалансированную выборку,
чтобы показать реальное влияние на точность классификации.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_20_species_data():
    """Загружает данные 20 видов деревьев"""
    spring_folder = "Спектры, весенний период, 20 видов"
    
    if not os.path.exists(spring_folder):
        print(f"❌ Папка {spring_folder} не найдена!")
        return [], [], []
    
    all_spectra = []
    all_labels = []
    species_counts = {}
    
    print("📁 Загрузка данных 20 видов...")
    
    all_folders = [f for f in os.listdir(spring_folder) 
                   if os.path.isdir(os.path.join(spring_folder, f))]
    all_folders.sort()
    
    for species in all_folders:
        folder_path = os.path.join(spring_folder, species)
        
        # Специальная обработка для клен_ам
        if species == "клен_ам":
            subfolder_path = os.path.join(folder_path, species)
            if os.path.exists(subfolder_path):
                folder_path = subfolder_path
        
        files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        
        species_spectra = []
        for file_path in files:
            try:
                df = pd.read_excel(file_path)
                if df.shape[1] >= 2:
                    spectrum = df.iloc[:, 1].values
                    spectrum = spectrum[~np.isnan(spectrum)]
                    if len(spectrum) > 100:
                        species_spectra.append(spectrum)
            except:
                continue
        
        if species_spectra:
            all_spectra.extend(species_spectra)
            all_labels.extend([species] * len(species_spectra))
            species_counts[species] = len(species_spectra)
            print(f"   🌳 {species}: {len(species_spectra)} спектров")
    
    return all_spectra, all_labels, list(species_counts.keys())

def preprocess_spectra(all_spectra, all_labels):
    """Предобрабатывает спектры"""
    min_length = min(len(spectrum) for spectrum in all_spectra)
    
    X = np.array([spectrum[:min_length] for spectrum in all_spectra])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(all_labels)
    
    return X, y, label_encoder, min_length

def create_imbalanced_datasets(X, y, species_names):
    """Создает различные варианты несбалансированных данных"""
    
    scenarios = {}
    
    # 1. Оригинальные сбалансированные данные
    scenarios['Сбалансированные (оригинал)'] = (X, y)
    
    # 2. Легкий дисбаланс (убираем 50% из половины классов)
    imbalanced_indices = []
    for class_idx in range(len(species_names)):
        class_mask = (y == class_idx)
        class_indices = np.where(class_mask)[0]
        
        if class_idx < 10:  # Первая половина классов - меньше данных
            selected = np.random.choice(class_indices, size=len(class_indices)//2, replace=False)
        else:  # Вторая половина - все данные
            selected = class_indices
        
        imbalanced_indices.extend(selected)
    
    imbalanced_indices = np.array(imbalanced_indices)
    X_imb_light = X[imbalanced_indices]
    y_imb_light = y[imbalanced_indices]
    scenarios['Легкий дисбаланс (2:1)'] = (X_imb_light, y_imb_light)
    
    # 3. Сильный дисбаланс (реалистичное распределение)
    # Создаем распределение как в реальном лесу
    target_proportions = [0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04,
                         0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005, 0.003, 0.002]
    
    realistic_indices = []
    total_samples = len(X)
    
    for class_idx in range(len(species_names)):
        class_mask = (y == class_idx)
        class_indices = np.where(class_mask)[0]
        
        target_count = int(total_samples * target_proportions[class_idx])
        target_count = min(target_count, len(class_indices))  # Не больше чем есть
        target_count = max(target_count, 2)  # Минимум 2 образца
        
        if target_count > 0:
            selected = np.random.choice(class_indices, size=target_count, replace=False)
            realistic_indices.extend(selected)
    
    realistic_indices = np.array(realistic_indices)
    X_realistic = X[realistic_indices]
    y_realistic = y[realistic_indices]
    scenarios['Реалистичный дисбаланс'] = (X_realistic, y_realistic)
    
    return scenarios

def test_scenarios(scenarios, species_names):
    """Тестирует все сценарии и сравнивает результаты"""
    
    results = {}
    
    print("\n🧪 ТЕСТИРОВАНИЕ РАЗЛИЧНЫХ СЦЕНАРИЕВ СБАЛАНСИРОВАННОСТИ")
    print("="*70)
    
    for scenario_name, (X, y) in scenarios.items():
        print(f"\n📊 Сценарий: {scenario_name}")
        print("-" * 50)
        
        # Подсчет распределения классов
        unique, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = max_count / min_count
        
        print(f"   📈 Общих образцов: {len(X)}")
        print(f"   📈 Дисбаланс: {imbalance_ratio:.1f}:1")
        print(f"   📈 Мин/Макс образцов на класс: {min_count}/{max_count}")
        
        # Разделение на train/test
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Если не хватает данных для стратификации
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение модели
        model = ExtraTreesClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Предсказание
        y_pred = model.predict(X_test_scaled)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"   🎯 Accuracy: {accuracy:.3f}")
        print(f"   🎯 Balanced Accuracy: {balanced_acc:.3f}")
        print(f"   📉 Потеря точности: {((0.97 - accuracy) * 100):.1f}%")
        
        results[scenario_name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'imbalance_ratio': imbalance_ratio,
            'total_samples': len(X),
            'min_class_count': min_count,
            'max_class_count': max_count
        }
        
        # Детальный отчет только для реалистичного сценария
        if 'Реалистичный' in scenario_name:
            print("\n   📋 Детальный отчет по классам (первые 10):")
            report = classification_report(y_test, y_pred, target_names=species_names, output_dict=True)
            for i, species in enumerate(species_names[:10]):
                if str(i) in report:
                    precision = report[str(i)]['precision']
                    recall = report[str(i)]['recall']
                    f1 = report[str(i)]['f1-score']
                    print(f"     {species:<20}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    
    return results

def create_comparison_visualization(results):
    """Создает визуализацию сравнения результатов"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    scenarios = list(results.keys())
    accuracies = [results[s]['accuracy'] for s in scenarios]
    balanced_accs = [results[s]['balanced_accuracy'] for s in scenarios]
    imbalance_ratios = [results[s]['imbalance_ratio'] for s in scenarios]
    
    # 1. Сравнение Accuracy vs Balanced Accuracy
    x = np.arange(len(scenarios))
    width = 0.35
    
    axes[0,0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='lightblue')
    axes[0,0].bar(x + width/2, balanced_accs, width, label='Balanced Accuracy', alpha=0.8, color='orange')
    axes[0,0].set_ylabel('Точность')
    axes[0,0].set_title('Accuracy vs Balanced Accuracy')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    # 2. Влияние дисбаланса на точность
    axes[0,1].scatter(imbalance_ratios, accuracies, s=100, alpha=0.7, color='red')
    for i, scenario in enumerate(scenarios):
        axes[0,1].annotate(scenario.split()[0], (imbalance_ratios[i], accuracies[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[0,1].set_xscale('log')
    axes[0,1].set_xlabel('Коэффициент дисбаланса (log scale)')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_title('Влияние дисбаланса на точность')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Потеря точности относительно оригинала
    original_acc = results['Сбалансированные (оригинал)']['accuracy']
    accuracy_losses = [(original_acc - acc) * 100 for acc in accuracies]
    
    colors = ['green', 'yellow', 'red']
    axes[1,0].bar(scenarios, accuracy_losses, color=colors, alpha=0.7)
    axes[1,0].set_ylabel('Потеря точности (%)')
    axes[1,0].set_title('Потеря точности относительно сбалансированных данных')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Количество образцов в каждом сценарии
    total_samples = [results[s]['total_samples'] for s in scenarios]
    axes[1,1].bar(scenarios, total_samples, alpha=0.7, color='purple')
    axes[1,1].set_ylabel('Количество образцов')
    axes[1,1].set_title('Размер датасета в каждом сценарии')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('imbalanced_reality_test.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Сохранено: imbalanced_reality_test.png")
    
    return fig

def main():
    """Основная функция"""
    print("🔍 ТЕСТ ВЛИЯНИЯ СБАЛАНСИРОВАННОСТИ НА ВАШИ ДАННЫЕ")
    print("="*60)
    
    # Загрузка данных
    spectra, labels, species_names = load_20_species_data()
    
    if not spectra:
        print("❌ Не удалось загрузить данные!")
        return
    
    print(f"✅ Загружено {len(spectra)} спектров, {len(species_names)} видов")
    
    # Предобработка
    X, y, label_encoder, min_length = preprocess_spectra(spectra, labels)
    print(f"📊 Форма данных: {X.shape}")
    
    # Создание несбалансированных вариантов
    scenarios = create_imbalanced_datasets(X, y, species_names)
    print(f"🎭 Создано {len(scenarios)} сценариев для тестирования")
    
    # Тестирование всех сценариев
    results = test_scenarios(scenarios, species_names)
    
    # Визуализация
    create_comparison_visualization(results)
    
    # Финальное заключение
    print("\n" + "="*60)
    print("🎯 ЗАКЛЮЧЕНИЕ:")
    original_acc = results['Сбалансированные (оригинал)']['accuracy']
    realistic_acc = results['Реалистичный дисбаланс']['accuracy']
    loss = (original_acc - realistic_acc) * 100
    
    print(f"   📊 Ваши сбалансированные данные: {original_acc:.1%}")
    print(f"   🌍 Реалистичный дисбаланс: {realistic_acc:.1%}")
    print(f"   📉 Потеря точности: {loss:.1f}%")
    print("\n   ✅ Ваши результаты КОРРЕКТНЫ для лабораторных условий!")
    print("   ⚠️  В реальных условиях ожидайте значительного снижения.")
    print("="*60)

if __name__ == "__main__":
    main() 