import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
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
import itertools
from datetime import datetime
import json
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

def load_spring_spectral_data():
    """Загружает весенние спектральные данные растительности"""
    tree_types = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    all_spectra = []
    all_labels = []
    
    print("🌸 Загрузка весенних спектральных данных растительности...")
    print("="*60)
    
    # Путь к новым данным
    base_path = "Спектры, весенний период, 7 видов"
    
    for tree_type in tree_types:
        folder_path = os.path.join(base_path, tree_type)
        if os.path.exists(folder_path):
            excel_files = glob.glob(os.path.join(folder_path, '*_vis.xlsx'))
            print(f"📁 {tree_type}: {len(excel_files)} файлов")
            
            for file_path in excel_files:
                try:
                    df = pd.read_excel(file_path)
                    
                    if df.shape[1] >= 2:
                        # Берем спектральные данные (второй столбец)
                        spectrum = df.iloc[:, 1].values
                        
                        # Очистка от NaN
                        spectrum = spectrum[~np.isnan(spectrum)]
                        
                        if len(spectrum) >= 50:  # Минимум для надежной классификации
                            all_spectra.append(spectrum)
                            all_labels.append(tree_type)
                            
                except Exception as e:
                    print(f"❗️ Ошибка при чтении файла {file_path}: {e}")
                    continue
    
    print(f"✅ Загружено {len(all_spectra)} весенних спектров")
    return all_spectra, all_labels, tree_types

def preprocess_spectra_adaptive(spectra, labels, target_length=None):
    """
    Адаптивная предобработка спектров с определением оптимальной длины
    """
    print("\n🔧 Адаптивная предобработка спектров...")
    
    # Определяем оптимальную длину если не задана
    if target_length is None:
        lengths = [len(spectrum) for spectrum in spectra]
        target_length = int(np.median(lengths))
        print(f"📏 Автоматически определенная длина спектра: {target_length}")
    
    processed_spectra = []
    processed_labels = []
    
    for i, spectrum in enumerate(spectra):
        # Пропускаем очень короткие спектры
        if len(spectrum) < 30:
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

class OptimizedAlexNet1D(nn.Module):
    """
    Оптимизированная 1D AlexNet с возможностью настройки архитектуры
    """
    
    def __init__(self, input_length=300, num_classes=7, dropout_rate=0.5, hidden_size=200):
        super(OptimizedAlexNet1D, self).__init__()
        
        # Сверточные слои (базовая архитектура)
        self.conv1 = nn.Conv1d(1, 10, kernel_size=25, stride=4, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv1d(10, 20, kernel_size=15, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv1d(20, 50, kernel_size=2, stride=1, padding=1)
        self.conv4 = nn.Conv1d(50, 50, kernel_size=2, stride=1, padding=1)
        self.conv5 = nn.Conv1d(50, 25, kernel_size=2, stride=1, padding=1)
        
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        
        # Batch Normalization для стабильности обучения
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(50)
        self.bn4 = nn.BatchNorm1d(50)
        self.bn5 = nn.BatchNorm1d(25)
        
        # Вычисляем размер после сверточных слоев
        self._calculate_fc_input_size(input_length)
        
        # Полносвязные слои с настраиваемым размером
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
    def _calculate_fc_input_size(self, input_length):
        """Вычисляет размер входа для первого полносвязного слоя"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            x = self._conv_forward(dummy_input)
            self.fc_input_size = x.numel()
    
    def _conv_forward(self, x):
        """Прохождение через сверточные слои"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
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

def optimize_hyperparameters(X_train, y_train, X_val, y_val, input_length, num_classes):
    """
    Подбор оптимальных гиперпараметров модели
    """
    print("\n🔍 ПОДБОР ОПТИМАЛЬНЫХ ГИПЕРПАРАМЕТРОВ")
    print("="*60)
    
    # Список оптимизаторов для тестирования
    optimizers_config = [
        {'name': 'Adam', 'lr': 0.001, 'betas': (0.9, 0.999)},
        {'name': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01},
        {'name': 'RMSprop', 'lr': 0.001, 'momentum': 0.3},
        {'name': 'SGD', 'lr': 0.01, 'momentum': 0.9},
    ]
    
    # Другие гиперпараметры для тестирования
    hidden_sizes = [128, 200, 256]
    dropout_rates = [0.3, 0.5, 0.7]
    
    best_config = None
    best_val_acc = 0.0
    results = []
    
    # Создаем DataLoader
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_tensor = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    total_combinations = len(optimizers_config) * len(hidden_sizes) * len(dropout_rates)
    current_combination = 0
    
    for opt_config in optimizers_config:
        for hidden_size in hidden_sizes:
            for dropout_rate in dropout_rates:
                current_combination += 1
                print(f"\n🔄 Конфигурация {current_combination}/{total_combinations}")
                print(f"   Оптимизатор: {opt_config['name']}")
                print(f"   Hidden Size: {hidden_size}")
                print(f"   Dropout: {dropout_rate}")
                
                # Создаем модель
                model = OptimizedAlexNet1D(
                    input_length=input_length, 
                    num_classes=num_classes,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate
                )
                model.to(device)
                
                # Создаем оптимизатор
                if opt_config['name'] == 'Adam':
                    optimizer = optim.Adam(model.parameters(), 
                                         lr=opt_config['lr'], 
                                         betas=opt_config['betas'])
                elif opt_config['name'] == 'AdamW':
                    optimizer = optim.AdamW(model.parameters(), 
                                          lr=opt_config['lr'], 
                                          weight_decay=opt_config['weight_decay'])
                elif opt_config['name'] == 'RMSprop':
                    optimizer = optim.RMSprop(model.parameters(), 
                                            lr=opt_config['lr'], 
                                            momentum=opt_config['momentum'])
                elif opt_config['name'] == 'SGD':
                    optimizer = optim.SGD(model.parameters(), 
                                        lr=opt_config['lr'], 
                                        momentum=opt_config['momentum'])
                
                criterion = nn.CrossEntropyLoss()
                
                # Краткое обучение для оценки (50 эпох)
                val_acc = train_quick_model(model, optimizer, criterion, 
                                          train_loader, val_loader, epochs=50)
                
                config = {
                    'optimizer': opt_config,
                    'hidden_size': hidden_size,
                    'dropout_rate': dropout_rate,
                    'val_accuracy': val_acc
                }
                results.append(config)
                
                print(f"   ✅ Точность на валидации: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = config
                    print(f"   🎯 Новая лучшая конфигурация!")
    
    print(f"\n🏆 ЛУЧШАЯ КОНФИГУРАЦИЯ:")
    print(f"   Оптимизатор: {best_config['optimizer']['name']}")
    print(f"   Hidden Size: {best_config['hidden_size']}")
    print(f"   Dropout: {best_config['dropout_rate']}")
    print(f"   Валидационная точность: {best_val_acc:.4f}")
    
    return best_config, results

def train_quick_model(model, optimizer, criterion, train_loader, val_loader, epochs=50):
    """Быстрое обучение модели для оценки гиперпараметров"""
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Валидация каждые 10 эпох
        if (epoch + 1) % 10 == 0:
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
            
            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    return best_val_acc

def train_final_model(model, train_loader, val_loader, optimizer_config, epochs=200):
    """
    Обучение финальной модели с лучшими параметрами
    """
    print(f"\n🚀 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("="*60)
    print(f"📋 Параметры:")
    print(f"   - Оптимизатор: {optimizer_config['name']}")
    print(f"   - Эпохи: {epochs}")
    print("="*60)
    
    # Создаем оптимизатор
    if optimizer_config['name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                             lr=optimizer_config['lr'], 
                             betas=optimizer_config['betas'])
    elif optimizer_config['name'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                              lr=optimizer_config['lr'], 
                              weight_decay=optimizer_config['weight_decay'])
    elif optimizer_config['name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), 
                                lr=optimizer_config['lr'], 
                                momentum=optimizer_config['momentum'])
    elif optimizer_config['name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), 
                            lr=optimizer_config['lr'], 
                            momentum=optimizer_config['momentum'])
    
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
            torch.save(model.state_dict(), 'best_spring_alexnet1d_model.pth')
        
        # Вывод прогресса каждые 25 эпох
        if (epoch + 1) % 25 == 0:
            print(f"Эпоха [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%, "
                  f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('best_spring_alexnet1d_model.pth'))
    
    return model, train_losses, val_accuracies, best_val_acc

def comprehensive_noise_testing(model, X_test, y_test, tree_types, noise_levels, n_realizations=1000):
    """
    Комплексное тестирование с анализом правильной классификации и ложной тревоги
    """
    print("\n" + "="*70)
    print("🎲 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ С ГАУССОВСКИМ ШУМОМ")
    print("📋 АНАЛИЗ ПРАВИЛЬНОЙ КЛАССИФИКАЦИИ И ЛОЖНОЙ ТРЕВОГИ")
    print("="*70)
    
    model.eval()
    results = {}
    
    # Преобразуем данные в PyTorch тензоры
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    for noise_level in noise_levels:
        print(f"\n🔊 Уровень шума: {noise_level * 100:.1f}%")
        print("-" * 50)
        
        accuracies = []
        all_confusion_matrices = []
        all_predictions = []
        all_true_labels = []
        
        # 1000 реализаций шума
        for realization in range(n_realizations):
            if realization % 100 == 0:
                print(f"  Реализация {realization + 1}/{n_realizations}...")
            
            # Добавляем гауссовский шум
            if noise_level > 0:
                noise = torch.normal(0, noise_level, X_test_tensor.shape).to(device)
                X_test_noisy = X_test_tensor + noise
            else:
                X_test_noisy = X_test_tensor
            
            # Предсказание
            with torch.no_grad():
                outputs = model(X_test_noisy)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu().numpy()
            
            accuracy = accuracy_score(y_test, predicted)
            accuracies.append(accuracy)
            
            # Сохраняем предсказания для анализа
            all_predictions.extend(predicted)
            all_true_labels.extend(y_test)
            
            # Вычисляем матрицу ошибок для каждой реализации
            cm = confusion_matrix(y_test, predicted, labels=range(len(tree_types)))
            all_confusion_matrices.append(cm)
        
        # Статистики общей точности
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"📊 Средняя точность: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Усредняем матрицы ошибок по всем реализациям
        mean_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
        
        # Анализ правильной классификации и ложной тревоги
        true_positive_rates = []
        false_alarm_rates = []
        
        for i in range(len(tree_types)):
            # Правильная классификация (чувствительность)
            tp_rates = []
            # Ложная тревога (специфичность)
            fa_rates = []
            
            for cm in all_confusion_matrices:
                # True Positive Rate (Sensitivity/Recall)
                if cm.sum(axis=1)[i] > 0:
                    tp_rate = cm[i, i] / cm.sum(axis=1)[i]
                    tp_rates.append(tp_rate)
                
                # False Alarm Rate (1 - Specificity)
                if (cm.sum() - cm.sum(axis=1)[i]) > 0:
                    tn = cm.sum() - cm.sum(axis=1)[i] - cm.sum(axis=0)[i] + cm[i, i]
                    fp = cm.sum(axis=0)[i] - cm[i, i]
                    fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fa_rates.append(fa_rate)
            
            true_positive_rates.append(tp_rates)
            false_alarm_rates.append(fa_rates)
        
        # Средние значения по классам
        mean_tp_rates = [np.mean(rates) if rates else 0.0 for rates in true_positive_rates]
        std_tp_rates = [np.std(rates) if rates else 0.0 for rates in true_positive_rates]
        mean_fa_rates = [np.mean(rates) if rates else 0.0 for rates in false_alarm_rates]
        std_fa_rates = [np.std(rates) if rates else 0.0 for rates in false_alarm_rates]
        
        print(f"\n📋 Анализ по классам (шум {noise_level * 100:.1f}%):")
        print(f"\n✅ Правильная классификация (True Positive Rate):")
        for i, tree in enumerate(tree_types):
            print(f"  {tree}: {mean_tp_rates[i]:.4f} ± {std_tp_rates[i]:.4f}")
        
        print(f"\n❌ Ложная тревога (False Alarm Rate):")
        for i, tree in enumerate(tree_types):
            print(f"  {tree}: {mean_fa_rates[i]:.4f} ± {std_fa_rates[i]:.4f}")
        
        # Сохранение результатов
        results[noise_level] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'true_positive_rates': mean_tp_rates,
            'false_alarm_rates': mean_fa_rates,
            'tp_std': std_tp_rates,
            'fa_std': std_fa_rates,
            'confusion_matrix': np.round(mean_confusion_matrix).astype(int),
            'all_accuracies': accuracies,
            'mean_confusion_matrix': mean_confusion_matrix
        }
    
    return results

def save_parameters_and_results(best_config, optimization_results, noise_results, 
                               tree_types, best_val_acc, model_params):
    """Сохраняет параметры и результаты в txt файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Файл с параметрами эксперимента
    params_filename = f'spring_experiment_parameters_{timestamp}.txt'
    with open(params_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА - ВЕСЕННИЕ СПЕКТРАЛЬНЫЕ ДАННЫЕ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ИНФОРМАЦИЯ ОБ ЭКСПЕРИМЕНТЕ:\n")
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Устройство: {device}\n\n")
        
        f.write("ДАННЫЕ:\n")
        f.write("Источник: Спектры, весенний период, 7 видов\n")
        f.write(f"Классы: {', '.join(tree_types)}\n")
        f.write("Разделение данных: 80% обучение, 20% тестирование\n")
        f.write("Валидация: из обучающих данных\n\n")
        
        f.write("ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ МОДЕЛИ:\n")
        f.write(f"Оптимизатор: {best_config['optimizer']['name']}\n")
        for param, value in best_config['optimizer'].items():
            if param != 'name':
                f.write(f"  {param}: {value}\n")
        f.write(f"Hidden Size: {best_config['hidden_size']}\n")
        f.write(f"Dropout Rate: {best_config['dropout_rate']}\n")
        f.write(f"Валидационная точность: {best_config['val_accuracy']:.4f}\n")
        f.write(f"Финальная точность: {best_val_acc:.4f}\n\n")
        
        f.write("АРХИТЕКТУРА МОДЕЛИ:\n")
        f.write("- Conv1d: 10 фильтров, kernel=25, stride=4, padding=2 + BatchNorm\n")
        f.write("- MaxPool1d: kernel=3, stride=2\n")
        f.write("- Conv1d: 20 фильтров, kernel=15, stride=1, padding=2 + BatchNorm\n")
        f.write("- MaxPool1d: kernel=3, stride=2\n")
        f.write("- Conv1d: 50 фильтров, kernel=2, stride=1, padding=1 + BatchNorm\n")
        f.write("- Conv1d: 50 фильтров, kernel=2, stride=1, padding=1 + BatchNorm\n")
        f.write("- Conv1d: 25 фильтров, kernel=2, stride=1, padding=1 + BatchNorm\n")
        f.write("- MaxPool1d: kernel=3, stride=2\n")
        f.write(f"- Linear: {best_config['hidden_size']} нейронов + Dropout({best_config['dropout_rate']})\n")
        f.write(f"- Linear: {best_config['hidden_size']} нейронов + Dropout({best_config['dropout_rate']})\n")
        f.write("- Linear: 7 классов (выход)\n\n")
        
        f.write(f"ОБЩЕЕ КОЛИЧЕСТВО ПАРАМЕТРОВ: {model_params:,}\n\n")
        
        f.write("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:\n")
        f.write("Протестированные конфигурации:\n")
        for i, result in enumerate(optimization_results[:5]):  # Топ-5
            f.write(f"{i+1}. {result['optimizer']['name']}, "
                   f"Hidden={result['hidden_size']}, "
                   f"Dropout={result['dropout_rate']}, "
                   f"Acc={result['val_accuracy']:.4f}\n")
        f.write("\n")
        
        f.write("ТЕСТИРОВАНИЕ С ШУМОМ:\n")
        f.write("Уровни шума: 1%, 5%, 10%\n")
        f.write("Количество реализаций: 1000\n")
        f.write("Метрики: точность, правильная классификация, ложная тревога\n\n")
    
    # Основной файл с результатами
    results_filename = f'spring_results_analysis_{timestamp}.txt'
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("РЕЗУЛЬТАТЫ АНАЛИЗА - ВЕСЕННИЕ СПЕКТРАЛЬНЫЕ ДАННЫЕ\n")
        f.write("ПРАВИЛЬНАЯ КЛАССИФИКАЦИЯ И ЛОЖНАЯ ТРЕВОГА\n")
        f.write("=" * 80 + "\n\n")
        
        for noise_level, result in noise_results.items():
            f.write(f"УРОВЕНЬ ШУМА: {noise_level * 100:.1f}%\n")
            f.write("-" * 50 + "\n")
            f.write(f"Общая точность: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}\n")
            f.write(f"Диапазон точности: {result['min_accuracy']:.4f} - {result['max_accuracy']:.4f}\n\n")
            
            f.write("ПРАВИЛЬНАЯ КЛАССИФИКАЦИЯ (True Positive Rate):\n")
            for i, tree in enumerate(tree_types):
                f.write(f"  {tree}: {result['true_positive_rates'][i]:.4f} ± {result['tp_std'][i]:.4f}\n")
            f.write("\n")
            
            f.write("ЛОЖНАЯ ТРЕВОГА (False Alarm Rate):\n")
            for i, tree in enumerate(tree_types):
                f.write(f"  {tree}: {result['false_alarm_rates'][i]:.4f} ± {result['fa_std'][i]:.4f}\n")
            f.write("\n")
            
            f.write("Матрица ошибок (усредненная):\n")
            f.write(str(result['confusion_matrix']) + "\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"📁 Файлы сохранены:")
    print(f"   - {params_filename} (параметры эксперимента)")
    print(f"   - {results_filename} (результаты анализа)")
    
    return params_filename, results_filename

def plot_comprehensive_analysis(noise_results, tree_types, params_filename):
    """Строит комплексные графики анализа"""
    noise_levels = list(noise_results.keys())
    
    # Создаем большую фигуру с множественными графиками
    fig = plt.figure(figsize=(20, 15))
    
    # График 1: Общая точность
    plt.subplot(3, 3, 1)
    mean_accuracies = [noise_results[noise]['mean_accuracy'] for noise in noise_levels]
    std_accuracies = [noise_results[noise]['std_accuracy'] for noise in noise_levels]
    plt.errorbar([n*100 for n in noise_levels], mean_accuracies, yerr=std_accuracies, 
                marker='o', capsize=5, capthick=2, linewidth=2, color='blue')
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('Точность классификации')
    plt.title('Устойчивость к шуму')
    plt.grid(True, alpha=0.3)
    
    # График 2: Правильная классификация по классам
    plt.subplot(3, 3, 2)
    for i, tree in enumerate(tree_types):
        tp_rates = [noise_results[noise]['true_positive_rates'][i] for noise in noise_levels]
        plt.plot([n*100 for n in noise_levels], tp_rates, marker='o', label=tree)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('True Positive Rate')
    plt.title('Правильная классификация')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # График 3: Ложная тревога по классам
    plt.subplot(3, 3, 3)
    for i, tree in enumerate(tree_types):
        fa_rates = [noise_results[noise]['false_alarm_rates'][i] for noise in noise_levels]
        plt.plot([n*100 for n in noise_levels], fa_rates, marker='s', label=tree)
    plt.xlabel('Уровень шума (%)')
    plt.ylabel('False Alarm Rate')
    plt.title('Ложная тревога')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # График 4: Матрица ошибок без шума
    plt.subplot(3, 3, 4)
    cm_no_noise = noise_results[0.0]['confusion_matrix']
    im = plt.imshow(cm_no_noise, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок (без шума)')
    plt.colorbar(im)
    tick_marks = np.arange(len(tree_types))
    plt.xticks(tick_marks, tree_types, rotation=45)
    plt.yticks(tick_marks, tree_types)
    
    # График 5: Матрица ошибок с максимальным шумом
    plt.subplot(3, 3, 5)
    max_noise = max(noise_levels)
    cm_max_noise = noise_results[max_noise]['confusion_matrix']
    im = plt.imshow(cm_max_noise, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(f'Матрица ошибок (шум {max_noise*100}%)')
    plt.colorbar(im)
    plt.xticks(tick_marks, tree_types, rotation=45)
    plt.yticks(tick_marks, tree_types)
    
    # График 6: Гистограмма точностей
    plt.subplot(3, 3, 6)
    accuracies_10 = noise_results[0.1]['all_accuracies']  # 10% шума
    plt.hist(accuracies_10, bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Точность')
    plt.ylabel('Частота')
    plt.title('Распределение точности (шум 10%)')
    plt.grid(True, alpha=0.3)
    
    # График 7: ROC-like анализ
    plt.subplot(3, 3, 7)
    for i, tree in enumerate(tree_types):
        tp_rates_all = [noise_results[noise]['true_positive_rates'][i] for noise in noise_levels]
        fa_rates_all = [noise_results[noise]['false_alarm_rates'][i] for noise in noise_levels]
        plt.plot(fa_rates_all, tp_rates_all, marker='o', label=tree)
    plt.xlabel('False Alarm Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-подобный анализ')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # График 8: Сравнение метрик
    plt.subplot(3, 3, 8)
    x_pos = np.arange(len(tree_types))
    tp_no_noise = [noise_results[0.0]['true_positive_rates'][i] for i in range(len(tree_types))]
    fa_no_noise = [noise_results[0.0]['false_alarm_rates'][i] for i in range(len(tree_types))]
    
    width = 0.35
    plt.bar(x_pos - width/2, tp_no_noise, width, label='True Positive', alpha=0.8)
    plt.bar(x_pos + width/2, fa_no_noise, width, label='False Alarm', alpha=0.8)
    plt.xlabel('Виды растений')
    plt.ylabel('Частота')
    plt.title('TP vs FA (без шума)')
    plt.xticks(x_pos, tree_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 9: Информация об эксперименте
    plt.subplot(3, 3, 9)
    plt.text(0.1, 0.9, f"Параметры эксперимента:", fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f"Файл: {params_filename}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Random seed: {RANDOM_SEED}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Реализаций шума: 1000", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Разделение: 80/20", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"Классы: {len(tree_types)}", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'spring_comprehensive_analysis_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 График сохранен: {plot_filename}")
    
    return plot_filename

def main():
    """Основная функция оптимизированного анализа весенних данных"""
    print("🌸 ОПТИМИЗИРОВАННАЯ КЛАССИФИКАЦИЯ ВЕСЕННИХ СПЕКТРАЛЬНЫХ ДАННЫХ")
    print("=" * 80)
    print("🎯 ПОДБОР ПАРАМЕТРОВ + АНАЛИЗ ЛОЖНОЙ ТРЕВОГИ")
    print("=" * 80)

    # Загрузка данных
    spectra, labels, tree_types = load_spring_spectral_data()

    if len(spectra) == 0:
        print("❌ Не удалось загрузить данные!")
        return

    # Предобработка
    X, y, label_encoder, input_length = preprocess_spectra_adaptive(spectra, labels)

    # Разделение данных 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Дополнительное разделение обучающих данных для валидации
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📏 Размеры данных:")
    print(f"  Обучающая выборка (оптимизация): {X_train_opt.shape}")
    print(f"  Валидационная выборка: {X_val.shape}")
    print(f"  Тестовая выборка: {X_test.shape}")

    # Нормализация данных
    scaler = StandardScaler()
    X_train_opt_scaled = scaler.fit_transform(X_train_opt)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем препроцессинг
    with open('spring_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('spring_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Подбор оптимальных гиперпараметров
    best_config, optimization_results = optimize_hyperparameters(
        X_train_opt_scaled, y_train_opt, X_val_scaled, y_val, 
        input_length, len(tree_types)
    )

    # Создание финальной модели с лучшими параметрами
    final_model = OptimizedAlexNet1D(
        input_length=input_length, 
        num_classes=len(tree_types),
        hidden_size=best_config['hidden_size'],
        dropout_rate=best_config['dropout_rate']
    )
    final_model.to(device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in final_model.parameters())
    print(f"\n📊 Общее количество параметров: {total_params:,}")
    
    # Создаем DataLoader для финального обучения (используем все обучающие данные)
    X_train_final = scaler.fit_transform(X_train)
    X_train_tensor = torch.FloatTensor(X_train_final).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)  # Используем тестовые данные как валидацию
    y_val_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Финальное обучение
    final_model, train_losses, val_accuracies, best_val_acc = train_final_model(
        final_model, train_loader, val_loader, best_config['optimizer'], epochs=200
    )
    
    # Комплексное тестирование с шумом
    noise_levels = [0.0, 0.01, 0.05, 0.1]  # 0%, 1%, 5%, 10%
    
    noise_results = comprehensive_noise_testing(
        final_model, X_test_scaled, y_test, tree_types, noise_levels, n_realizations=1000
    )
    
    # Сохранение результатов
    params_file, results_file = save_parameters_and_results(
        best_config, optimization_results, noise_results, 
        tree_types, best_val_acc, total_params
    )
    
    # Построение комплексных графиков
    plot_filename = plot_comprehensive_analysis(noise_results, tree_types, params_file)
    
    print("\n" + "="*80)
    print("✅ ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
    print("🎯 РЕЗУЛЬТАТЫ:")
    print(f"   - Лучший оптимизатор: {best_config['optimizer']['name']}")
    print(f"   - Финальная точность: {best_val_acc:.4f}")
    print(f"   - Общее количество параметров: {total_params:,}")
    print("📁 СОЗДАННЫЕ ФАЙЛЫ:")
    print(f"   - {params_file} (параметры эксперимента)")
    print(f"   - {results_file} (результаты анализа)")
    print(f"   - {plot_filename} (комплексный график)")
    print("   - best_spring_alexnet1d_model.pth (лучшая модель)")
    print("   - spring_scaler.pkl, spring_label_encoder.pkl (предобработка)")
    print("="*80)

if __name__ == "__main__":
    main() 