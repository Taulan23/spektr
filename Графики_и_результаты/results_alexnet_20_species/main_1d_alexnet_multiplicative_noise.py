import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
warnings.filterwarnings('ignore')

# Фиксируем random seeds для воспроизводимости
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Устанавливаем device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Используемое устройство: {device}")
print(f"🎲 Random seed зафиксирован: {RANDOM_SEED}")

def load_spectral_data():
    """Загружает спектральные данные растительности для 1D-AlexNet"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_spectra = []
    all_labels = []
    
    print("🌿 Загрузка спектральных данных растительности...")
    print("="*60)
    
    for tree_type in tree_types:
        folder_path = os.path.join('.', tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            print(f"📁 {tree_type}: {len(excel_files)} файлов")
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    
                    if df.shape[1] >= 2:
                        # Берем спектральные данные (второй столбец)
                        spectrum = df.iloc[:, 1].values
                        
                        # Очистка от NaN
                        spectrum = spectrum[~np.isnan(spectrum)]
                        
                        if len(spectrum) >= 100:  # Минимум для надежной классификации
                            all_spectra.append(spectrum)
                            all_labels.append(tree_type)
                            
                except Exception as e:
                    print(f"❗️ Ошибка при чтении файла {file_path}: {e}")
                    continue
    
    print(f"✅ Загружено {len(all_spectra)} спектров растительности")
    return all_spectra, all_labels, tree_types

def preprocess_spectra_for_1d_alexnet(spectra, labels, target_length=None):
    """
    Предобрабатывает спектры для 1D-AlexNet с использованием интерполяции.
    """
    print("\n🔧 Предобработка спектров для 1D-AlexNet...")
    
    # Автоматически определяем наиболее частую длину спектра
    if target_length is None:
        lengths = [len(s) for s in spectra]
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        target_length = unique_lengths[np.argmax(counts)]
        print(f"📏 Автоматически определенная длина спектра: {target_length}")
        print(f"📊 Распределение длин: {dict(zip(unique_lengths, counts))}")
    else:
        print(f"📏 Заданная длина спектра: {target_length}")
    
    processed_spectra = []
    processed_labels = []
    
    for i, spectrum in enumerate(spectra):
        # Пропускаем очень короткие спектры
        if len(spectrum) < 50:
            continue
            
        # Интерполируем, если длина не совпадает
        if len(spectrum) != target_length:
            processed_spectrum = np.interp(
                np.linspace(0, len(spectrum) - 1, target_length),
                np.arange(len(spectrum)),
                spectrum
            )
        else:
            processed_spectrum = spectrum
            
        processed_spectra.append(processed_spectrum)
        processed_labels.append(labels[i])
    
    # Преобразуем в numpy массив
    X = np.array(processed_spectra, dtype=np.float32)
    
    # Кодируем метки классов
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(processed_labels)
    
    print(f"📊 Форма данных: {X.shape}")
    print(f"🎯 Количество классов: {len(np.unique(y))}")
    print(f"🏷️ Классы: {label_encoder.classes_}")
    
    return X, y, label_encoder, target_length

class AlexNet1D(nn.Module):
    """
    Точная реализация 1D AlexNet согласно присланной схеме
    (адаптированная для автоматически определенной длины спектра)
    """
    
    def __init__(self, input_length=300, num_classes=7):
        super(AlexNet1D, self).__init__()
        
        # Сверточные слои согласно схеме (адаптированные размеры)
        self.conv1 = nn.Conv1d(1, 10, kernel_size=25, stride=4, padding=2)  # 10 фильтров, уменьшенный размер ядра
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)                   # размер 3, stride 2
        
        self.conv2 = nn.Conv1d(10, 20, kernel_size=15, stride=1, padding=2)  # 20 фильтров, уменьшенный размер ядра  
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)                   # размер 3, stride 2
        
        self.conv3 = nn.Conv1d(20, 50, kernel_size=2, stride=1, padding=1)   # 50 фильтров, размер 2, stride 1
        self.conv4 = nn.Conv1d(50, 50, kernel_size=2, stride=1, padding=1)   # 50 фильтров, размер 2, stride 1
        self.conv5 = nn.Conv1d(50, 25, kernel_size=2, stride=1, padding=1)   # 25 фильтров, размер 2, stride 1
        
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)                   # размер 3, stride 2
        
        # Вычисляем размер после сверточных слоев
        self._calculate_fc_input_size(input_length)
        
        # Полносвязные слои согласно схеме
        self.fc1 = nn.Linear(self.fc_input_size, 200)  # 200 нейронов
        self.fc2 = nn.Linear(200, 200)                 # 200 нейронов
        self.fc3 = nn.Linear(200, num_classes)         # количество классов
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_fc_input_size(self, input_length):
        """Вычисляет размер входа для первого полносвязного слоя"""
        # Проводим dummy forward pass для определения размера
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            x = self._conv_forward(dummy_input)
            self.fc_input_size = x.numel()
    
    def _conv_forward(self, x):
        """Прохождение через сверточные слои"""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = self.pool3(x)
        
        return x
    
    def forward(self, x):
        # Сверточные слои
        x = self._conv_forward(x)
        
        # Flatten для полносвязных слоев
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

def train_model(model, train_loader, val_loader, epochs=400, learning_rate=0.001, momentum=0.3):
    """
    Обучение модели с параметрами из статьи
    """
    print(f"\n🚀 ОБУЧЕНИЕ 1D AlexNet")
    print("="*60)
    print(f"📋 Параметры:")
    print(f"   - Оптимизатор: RMSprop")
    print(f"   - Learning Rate: {learning_rate}")
    print(f"   - Momentum: {momentum}")
    print(f"   - Эпохи: {epochs}")
    print("="*60)
    
    # Создаем оптимизатор RMSprop согласно статье
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_alexnet1d_multiplicative_model.pth')
        
        # Вывод прогресса каждые 50 эпох
        if (epoch + 1) % 50 == 0:
            print(f"Эпоха [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%, "
                  f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_alexnet1d_multiplicative_model.pth'))
    
    return model, train_losses, val_accuracies, best_val_acc

def test_with_multiplicative_gaussian_noise(model, X_test, y_test, tree_types, noise_levels, n_realizations=1000):
    """
    Тестирование с МУЛЬТИПЛИКАТИВНЫМ гауссовским шумом
    Каждый спектральный отчет умножается на (1 + дельта)
    где дельта ~ N(0, процент_шума)
    """
    print("\n" + "="*80)
    print("🎲 ТЕСТИРОВАНИЕ С МУЛЬТИПЛИКАТИВНЫМ ГАУССОВСКИМ ШУМОМ")
    print("📋 МЕТОДОЛОГИЯ:")
    print("   - Каждый спектральный отчет умножается на (1 + дельта)")
    print("   - Дельта распределена по нормальному закону со средним = 0")
    print("   - СКО дельты = процент шума")
    print("   - X_noisy = X * (1 + delta), где delta ~ N(0, σ)")
    print("   - 1000 реализаций для каждого уровня шума")
    print("="*80)
    
    model.eval()
    results = {}
    
    # Преобразуем данные в PyTorch тензоры
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)  # добавляем канал
    
    for noise_level in noise_levels:
        print(f"\n🔊 Уровень шума: {noise_level * 100:.1f}%")
        print("-" * 50)
        
        accuracies = []
        all_confusion_matrices = []
        
        # 1000 реализаций шума
        for realization in range(n_realizations):
            if realization % 100 == 0:
                print(f"  Реализация {realization + 1}/{n_realizations}...")
            
            # Добавляем МУЛЬТИПЛИКАТИВНЫЙ гауссовский шум
            if noise_level > 0:
                # delta ~ N(0, noise_level)
                delta = torch.normal(0, noise_level, X_test_tensor.shape).to(device)
                # X_noisy = X * (1 + delta)
                X_test_noisy = X_test_tensor * (1 + delta)
            else:
                X_test_noisy = X_test_tensor
            
            # Предсказание
            with torch.no_grad():
                outputs = model(X_test_noisy)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu().numpy()
            
            accuracy = accuracy_score(y_test, predicted)
            accuracies.append(accuracy)
            
            # Вычисляем матрицу ошибок для каждой реализации
            cm = confusion_matrix(y_test, predicted, labels=range(len(tree_types)))
            all_confusion_matrices.append(cm)
        
        # Статистики общей точности
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"📊 Средняя точность: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"📈 Минимальная точность: {np.min(accuracies):.4f}")
        print(f"📈 Максимальная точность: {np.max(accuracies):.4f}")
        
        # Усредняем матрицы ошибок по всем реализациям
        mean_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
        
        # Вычисляем среднюю точность по классам
        mean_class_accuracies = []
        std_class_accuracies = []
        for i in range(len(tree_types)):
            class_accuracies_all_realizations = []
            for cm in all_confusion_matrices:
                if cm.sum(axis=1)[i] > 0:  # избегаем деления на ноль
                    class_acc = cm[i, i] / cm.sum(axis=1)[i]
                    class_accuracies_all_realizations.append(class_acc)
            
            if class_accuracies_all_realizations:
                mean_class_acc = np.mean(class_accuracies_all_realizations)
                std_class_acc = np.std(class_accuracies_all_realizations)
                mean_class_accuracies.append(mean_class_acc)
                std_class_accuracies.append(std_class_acc)
            else:
                mean_class_accuracies.append(0.0)
                std_class_accuracies.append(0.0)
        
        # Отчет о классификации с усредненными данными
        print(f"\n📋 Средние результаты по {n_realizations} реализациям (шум {noise_level * 100:.1f}%):")
        print(f"\n✅ Средние вероятности правильной классификации по классам:")
        for i, tree in enumerate(tree_types):
            print(f"  {tree}: {mean_class_accuracies[i]:.4f} ± {std_class_accuracies[i]:.4f}")
        
        # Средняя матрица ошибок
        print(f"\n📊 Средняя матрица ошибок (округленная):")
        print(np.round(mean_confusion_matrix).astype(int))
        
        # Сохранение результатов
        results[noise_level] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'class_accuracies': mean_class_accuracies,
            'std_class_accuracies': std_class_accuracies,
            'confusion_matrix': np.round(mean_confusion_matrix).astype(int),
            'all_accuracies': accuracies,
            'mean_confusion_matrix': mean_confusion_matrix
        }
    
    return results

def print_confusion_matrix_table(results, tree_types):
    """
    Выводит матрицы ошибок в табличном формате для заданных уровней шума
    согласно формату из приложенного изображения
    """
    print("\n" + "="*100)
    print("📊 МАТРИЦЫ ОШИБОК В ТАБЛИЧНОМ ФОРМАТЕ (МУЛЬТИПЛИКАТИВНЫЙ ШУМ)")
    print("📋 Каждый спектральный отчет умножается на (1 + дельта)")
    print("📋 Дельта ~ N(0, σ) где σ = процент шума")
    print("="*100)
    
    # Матрицы для шума 1%, 5%, 10%
    target_noise_levels = [0.01, 0.05, 0.10]
    
    for noise_level in target_noise_levels:
        if noise_level in results:
            print(f"\n🔊 ТАБЛИЦА 3 - МАТРИЦА ОШИБОК ДЛЯ ШУМА {noise_level*100:.0f}%")
            print("В таблице для примера показана сводная матрица")
            print("средних (по реализациям шума) значений вероятностей")
            print("правильной и неправильной классификации пород деревьев")
            print("-" * 80)
            
            cm = results[noise_level]['confusion_matrix']
            total_samples_per_class = cm.sum(axis=1)
            
            # Нормализуем матрицу для получения вероятностей
            normalized_cm = np.zeros_like(cm, dtype=float)
            for i in range(len(tree_types)):
                if total_samples_per_class[i] > 0:
                    normalized_cm[i] = cm[i] / total_samples_per_class[i]
            
            # Заголовок таблицы
            header = "порода  "
            for tree in tree_types:
                short_name = tree[:6]  # Сокращаем названия для форматирования
                header += f"| {short_name:>8}"
            print(header)
            print("-" * len(header))
            
            # Строки матрицы с вероятностями
            for i, tree in enumerate(tree_types):
                short_name = tree[:6]
                row = f"{short_name:<8}"
                for j in range(len(tree_types)):
                    if normalized_cm[i,j] > 0:
                        row += f"| {normalized_cm[i,j]:>8.2f}"
                    else:
                        row += f"| {'0':>8}"
                print(row)
            
            # Добавляем информацию о средних точностях
            print(f"\n✅ Средние вероятности правильной классификации:")
            for i, tree in enumerate(tree_types):
                prob = results[noise_level]['class_accuracies'][i]
                std_prob = results[noise_level].get('std_class_accuracies', [0]*len(tree_types))[i]
                print(f"   {tree}: {prob:.3f} ± {std_prob:.3f}")
                
            print(f"\n📈 Общая точность: {results[noise_level]['mean_accuracy']:.4f} ± {results[noise_level]['std_accuracy']:.4f}")
            print("=" * 80)

def save_results_to_file(results, tree_types, best_val_acc, input_length):
    """Сохраняет результаты в файл для анализа"""
    with open('results_analysis_multiplicative_noise.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ РАСТИТЕЛЬНОСТИ 1D-AlexNet\n")
        f.write("МУЛЬТИПЛИКАТИВНЫЙ ГАУССОВСКИЙ ШУМ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("АРХИТЕКТУРА СЕТИ (согласно схеме):\n")
        f.write("- Conv1d: 10 фильтров, размер ядра 25, stride 4, padding 2\n")
        f.write("- MaxPool1d: размер 3, stride 2\n")
        f.write("- Conv1d: 20 фильтров, размер ядра 15, stride 1, padding 2\n")
        f.write("- MaxPool1d: размер 3, stride 2\n")
        f.write("- Conv1d: 50 фильтров, размер ядра 2, stride 1, padding 1\n")
        f.write("- Conv1d: 50 фильтров, размер ядра 2, stride 1, padding 1\n")
        f.write("- Conv1d: 25 фильтров, размер ядра 2, stride 1, padding 1\n")
        f.write("- MaxPool1d: размер 3, stride 2\n")
        f.write("- Linear: 200 нейронов\n")
        f.write("- Linear: 200 нейронов\n")
        f.write("- Linear: количество классов\n\n")
        
        f.write("ПАРАМЕТРЫ ОБУЧЕНИЯ:\n")
        f.write("- Оптимизатор: RMSprop\n")
        f.write("- Learning Rate: 0.001\n")
        f.write("- Momentum: 0.3\n")
        f.write("- Эпохи: 400\n")
        f.write("- Количество реализаций шума: 1000\n")
        f.write("- Тип шума: МУЛЬТИПЛИКАТИВНЫЙ (X * (1 + delta))\n")
        f.write("- delta ~ N(0, σ) где σ = процент шума\n")
        f.write(f"- Длина спектра: {input_length} точек\n\n")
        
        f.write(f"ЛУЧШАЯ ТОЧНОСТЬ НА ВАЛИДАЦИИ: {best_val_acc:.4f}\n\n")
        
        for noise_level, result in results.items():
            f.write(f"УРОВЕНЬ ШУМА: {noise_level * 100:.1f}%\n")
            f.write("-" * 50 + "\n")
            f.write(f"Средняя точность: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n")
            f.write(f"Минимальная точность: {result['min_accuracy']:.4f}\n")
            f.write(f"Максимальная точность: {result['max_accuracy']:.4f}\n\n")
            
            f.write("Вероятности правильной классификации по классам:\n")
            for i, tree in enumerate(tree_types):
                std_acc = result.get('std_class_accuracies', [0]*len(tree_types))[i]
                f.write(f"  {tree}: {result['class_accuracies'][i]:.4f} ± {std_acc:.4f}\n")
            f.write("\n")
            
            f.write("Матрица ошибок:\n")
            f.write(str(result['confusion_matrix']) + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("КЛЮЧЕВЫЕ ОТЛИЧИЯ ОТ АДДИТИВНОГО ШУМА:\n")
        f.write("1. Мультипликативный шум: X_noisy = X * (1 + delta)\n")
        f.write("2. Аддитивный шум: X_noisy = X + noise\n")
        f.write("3. Мультипликативный шум масштабирует сигнал пропорционально\n")
        f.write("4. Аддитивный шум добавляет постоянную составляющую\n")
        f.write("5. При мультипликативном шуме сохраняется форма спектра\n")
        f.write("=" * 80 + "\n")

def plot_noise_analysis(results, tree_types):
    """Строит графики анализа устойчивости к мультипликативному шуму"""
    noise_levels = list(results.keys())
    mean_accuracies = [results[noise]['mean_accuracy'] for noise in noise_levels]
    std_accuracies = [results[noise]['std_accuracy'] for noise in noise_levels]
    
    # График общей точности
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.errorbar([n*100 for n in noise_levels], mean_accuracies, yerr=std_accuracies, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность классификации')
    plt.title('Устойчивость 1D-AlexNet к мультипликативному шуму')
    plt.grid(True, alpha=0.3)
    
    # График точности по классам
    plt.subplot(2, 2, 2)
    for i, tree in enumerate(tree_types):
        class_accs = [results[noise]['class_accuracies'][i] for noise in noise_levels]
        plt.plot([n*100 for n in noise_levels], class_accs, marker='o', label=tree)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность по классам')
    plt.title('Точность классификации по видам растительности')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Гистограмма точностей для максимального шума
    plt.subplot(2, 2, 3)
    max_noise = max(noise_levels)
    accuracies = results[max_noise]['all_accuracies']
    plt.hist(accuracies, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Точность')
    plt.ylabel('Частота')
    plt.title(f'Распределение точности (мультипликативный шум {max_noise*100}%)')
    plt.grid(True, alpha=0.3)
    
    # Тепловая карта матрицы ошибок
    plt.subplot(2, 2, 4)
    cm = results[0.0]['confusion_matrix']  # Без шума
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (без шума)')
    plt.colorbar()
    tick_marks = np.arange(len(tree_types))
    plt.xticks(tick_marks, tree_types, rotation=45)
    plt.yticks(tick_marks, tree_types)
    
    plt.tight_layout()
    plt.savefig('1d_alexnet_multiplicative_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция для реализации 1D-AlexNet с мультипликативным шумом"""
    print("🌲 КЛАССИФИКАЦИЯ РАСТИТЕЛЬНОСТИ С 1D-AlexNet (МУЛЬТИПЛИКАТИВНЫЙ ШУМ)")
    print("=" * 80)
    print("📄 ТОЧНАЯ АРХИТЕКТУРА ИЗ ПРИСЛАННОЙ СХЕМЫ")
    print("🎯 Параметры: RMSprop, Rate=0.001, Moment=0.3, Epochs=400")
    print("🔊 ШУМ: X_noisy = X * (1 + delta), где delta ~ N(0, σ)")
    print("=" * 80)

    # Загрузка данных
    spectra, labels, tree_types = load_spectral_data()

    if len(spectra) == 0:
        print("❌ Не удалось загрузить данные!")
        return

    # Предобработка (автоматическое определение длины спектра)
    X, y, label_encoder, input_length = preprocess_spectra_for_1d_alexnet(spectra, labels)

    # Разделение данных 50/50
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    print(f"\n📏 Размеры данных:")
    print(f"  Обучающая выборка: {X_train.shape}")
    print(f"  Тестовая выборка: {X_test.shape}")

    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем scaler и label_encoder
    with open('scaler_multiplicative.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder_multiplicative.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Создаем DataLoader для PyTorch
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # добавляем канал
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_val_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Создание модели
    model = AlexNet1D(input_length=input_length, num_classes=len(tree_types))
    model.to(device)
    
    print(f"\n🏗️ Архитектура модели:")
    print(model)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Общее количество параметров: {total_params:,}")
    print(f"📊 Обучаемых параметров: {trainable_params:,}")
    
    # Обучение модели
    model, train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, 
        epochs=400, learning_rate=0.001, momentum=0.3
    )
    
    # Тестирование с мультипликативным шумом (0%, 1%, 5%, 10%)
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    results = test_with_multiplicative_gaussian_noise(
        model, X_test_scaled, y_test, tree_types, noise_levels, n_realizations=1000
    )
    
    # Сохранение результатов
    save_results_to_file(results, tree_types, best_val_acc, input_length)
    
    # Построение графиков
    plot_noise_analysis(results, tree_types)
    
    # Вывод матриц ошибок в табличном формате
    print_confusion_matrix_table(results, tree_types)
    
    print("\n" + "="*80)
    print("✅ АНАЛИЗ С МУЛЬТИПЛИКАТИВНЫМ ШУМОМ ЗАВЕРШЕН УСПЕШНО!")
    print("📊 Результаты с исправленной методологией шума:")
    print(f"   - Лучшая точность на валидации: {best_val_acc:.4f}")
    print(f"   - Длина спектра: {input_length} точек")
    print(f"   - Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print("   - Архитектура: точно по присланной схеме")
    print("   - RMSprop, Rate=0.001, Moment=0.3, Epochs=400")
    print("   - 1000 реализаций для каждого уровня шума")
    print("   - МУЛЬТИПЛИКАТИВНЫЙ шум: X * (1 + delta)")
    print("📁 Файлы сохранены:")
    print("   - results_analysis_multiplicative_noise.txt (детальные результаты)")
    print("   - best_alexnet1d_multiplicative_model.pth (лучшая модель)")
    print("   - scaler_multiplicative.pkl, label_encoder_multiplicative.pkl")
    print("   - 1d_alexnet_multiplicative_noise_analysis.png (графики)")
    print("="*80)

if __name__ == "__main__":
    main() 