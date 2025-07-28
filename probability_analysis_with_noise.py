import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob
from datetime import datetime

def load_spectral_data_7_species():
    """Загружает данные для 7 видов деревьев"""
    data = []
    labels = []
    
    species = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
    
    for species_name in species:
        print(f"Загрузка данных для {species_name}...")
        
        folder_path = f'Спектры, весенний период, 7 видов/{species_name}'
        files = glob.glob(f'{folder_path}/*.xlsx')
        
        for file in files[:50]:
            try:
                df = pd.read_excel(file)
                spectral_data = df.iloc[:, 1:].values.flatten()
                
                if len(spectral_data) > 0 and not np.any(np.isnan(spectral_data)):
                    spectral_data = (spectral_data - np.min(spectral_data)) / (np.max(spectral_data) - np.min(spectral_data))
                    data.append(spectral_data)
                    labels.append(species_name)
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    
    return np.array(data), np.array(labels)

def create_improved_1d_cnn(input_shape, num_classes):
    """Создает улучшенную 1D CNN модель"""
    model = keras.Sequential([
        layers.Conv1D(64, 7, strides=2, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(3, strides=2),
        
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(3, strides=2),
        
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.MaxPooling1D(3, strides=2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def add_noise(data, noise_level):
    """Добавляет шум к данным"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def analyze_probabilities_with_noise(model, X_test, y_test, class_names, noise_levels=[0, 0.01, 0.05, 0.1]):
    """Анализирует вероятности классификации с разными уровнями шума"""
    results = {}
    
    for noise_level in noise_levels:
        print(f"\nАнализ с шумом {noise_level*100}%...")
        
        # Добавляем шум к тестовым данным
        if noise_level > 0:
            X_test_noisy = add_noise(X_test, noise_level)
        else:
            X_test_noisy = X_test
        
        # Получаем вероятности
        probabilities = model.predict(X_test_noisy, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        # Точность
        accuracy = accuracy_score(y_test, predictions)
        
        # Анализ вероятностей
        max_probabilities = np.max(probabilities, axis=1)
        mean_confidence = np.mean(max_probabilities)
        std_confidence = np.std(max_probabilities)
        
        # Анализ по классам
        class_accuracies = {}
        class_confidences = {}
        
        for i, class_name in enumerate(class_names):
            class_mask = (y_test == i)
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_test == i) & (predictions == i)) / np.sum(class_mask)
                class_conf = np.mean(max_probabilities[class_mask])
                
                class_accuracies[class_name] = class_acc
                class_confidences[class_name] = class_conf
        
        results[noise_level] = {
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'class_accuracies': class_accuracies,
            'class_confidences': class_confidences,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        print(f"  Точность: {accuracy:.4f}")
        print(f"  Средняя уверенность: {mean_confidence:.4f} ± {std_confidence:.4f}")
    
    return results

def plot_probability_analysis(results, class_names, filename):
    """Создает графики анализа вероятностей"""
    noise_levels = list(results.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # График 1: Точность vs уровень шума
    accuracies = [results[noise]['accuracy'] for noise in noise_levels]
    ax1.plot([n*100 for n in noise_levels], accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Уровень шума (%)', fontsize=12)
    ax1.set_ylabel('Точность', fontsize=12)
    ax1.set_title('Точность классификации vs уровень шума', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    for i, acc in enumerate(accuracies):
        ax1.annotate(f'{acc:.3f}', (noise_levels[i]*100, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # График 2: Уверенность vs уровень шума
    confidences = [results[noise]['mean_confidence'] for noise in noise_levels]
    conf_std = [results[noise]['std_confidence'] for noise in noise_levels]
    
    ax2.errorbar([n*100 for n in noise_levels], confidences, yerr=conf_std, 
                fmt='ro-', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Уровень шума (%)', fontsize=12)
    ax2.set_ylabel('Средняя уверенность', fontsize=12)
    ax2.set_title('Уверенность модели vs уровень шума', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # График 3: Точность по классам для разных уровней шума
    x_pos = np.arange(len(class_names))
    width = 0.2
    
    for i, noise in enumerate(noise_levels):
        class_accs = [results[noise]['class_accuracies'].get(name, 0) for name in class_names]
        ax3.bar(x_pos + i*width, class_accs, width, 
               label=f'{noise*100}% шум', alpha=0.8)
    
    ax3.set_xlabel('Классы', fontsize=12)
    ax3.set_ylabel('Точность', fontsize=12)
    ax3.set_title('Точность по классам для разных уровней шума', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos + width * 1.5)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # График 4: Уверенность по классам для разных уровней шума
    for i, noise in enumerate(noise_levels):
        class_confs = [results[noise]['class_confidences'].get(name, 0) for name in class_names]
        ax4.bar(x_pos + i*width, class_confs, width, 
               label=f'{noise*100}% шум', alpha=0.8)
    
    ax4.set_xlabel('Классы', fontsize=12)
    ax4.set_ylabel('Средняя уверенность', fontsize=12)
    ax4.set_title('Уверенность по классам для разных уровней шума', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos + width * 1.5)
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_probability_table(results, class_names, filename):
    """Создает таблицу с вероятностями"""
    noise_levels = list(results.keys())
    
    # Создаем DataFrame для таблицы
    data = []
    for noise in noise_levels:
        for class_name in class_names:
            acc = results[noise]['class_accuracies'].get(class_name, 0)
            conf = results[noise]['class_confidences'].get(class_name, 0)
            data.append({
                'Уровень шума (%)': f'{noise*100}%',
                'Класс': class_name,
                'Точность': f'{acc:.3f}',
                'Уверенность': f'{conf:.3f}'
            })
    
    df = pd.DataFrame(data)
    
    # Сохраняем в CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    # Выводим таблицу
    print("\nТАБЛИЦА ВЕРОЯТНОСТЕЙ КЛАССИФИКАЦИИ:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df

def main():
    """Основная функция анализа вероятностей"""
    print("АНАЛИЗ ВЕРОЯТНОСТЕЙ КЛАССИФИКАЦИИ С ШУМОМ")
    print("=" * 60)
    
    # Загружаем данные
    X, y = load_spectral_data_7_species()
    
    # Кодируем метки
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Изменяем форму для CNN
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Создаем и обучаем модель
    print("Создание и обучение модели...")
    model = create_improved_1d_cnn(
        input_shape=(X_train_reshaped.shape[1], 1),
        num_classes=len(label_encoder.classes_)
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Обучаем модель
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Анализируем вероятности с разными уровнями шума
    print("\nАнализ вероятностей классификации...")
    results = analyze_probabilities_with_noise(
        model, X_test_reshaped, y_test, label_encoder.classes_,
        noise_levels=[0, 0.01, 0.05, 0.1]  # 0%, 1%, 5%, 10%
    )
    
    # Создаем графики
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'probability_analysis_with_noise_{timestamp}.png'
    table_filename = f'probability_table_{timestamp}.csv'
    
    plot_probability_analysis(results, label_encoder.classes_, plot_filename)
    create_probability_table(results, label_encoder.classes_, table_filename)
    
    print(f"\nРезультаты сохранены:")
    print(f"  Графики: {plot_filename}")
    print(f"  Таблица: {table_filename}")
    
    # Открываем результаты
    import subprocess
    subprocess.run(['open', plot_filename])
    subprocess.run(['open', table_filename])

if __name__ == "__main__":
    main() 