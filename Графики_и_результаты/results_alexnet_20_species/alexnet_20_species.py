#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D ALEXNET ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ
Адаптированная архитектура Alexnet для спектральных данных
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_20_species_data():
    """Загружает данные для всех 20 видов деревьев"""
    
    print("🌲 ЗАГРУЗКА ДАННЫХ 20 ВИДОВ...")
    
    spring_folder = "Спектры, весенний период, 20 видов"
    
    # Получаем все папки видов
    all_folders = [d for d in os.listdir(spring_folder) 
                   if os.path.isdir(os.path.join(spring_folder, d))]
    
    print(f"   Найдено папок: {len(all_folders)}")
    for folder in sorted(all_folders):
        print(f"   - {folder}")
    
    spring_data = []
    spring_labels = []
    
    for species in sorted(all_folders):
        folder_path = os.path.join(spring_folder, species)
        
        # Для клен_ам проверяем вложенную папку
        if species == "клен_ам":
            subfolder_path = os.path.join(folder_path, species)
            if os.path.exists(subfolder_path):
                folder_path = subfolder_path
        
        files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        print(f"   {species}: {len(files)} файлов (путь: {folder_path})")
        
        species_count = 0
        for file in files:
            try:
                df = pd.read_excel(file)
                if not df.empty and len(df.columns) >= 2:
                    spectrum = df.iloc[:, 1].values
                    if len(spectrum) > 100:
                        spring_data.append(spectrum)
                        spring_labels.append(species)
                        species_count += 1
            except Exception as e:
                continue
        
        if species_count == 0:
            print(f"   ⚠️  {species}: НЕТ ДАННЫХ - исключаем из классификации")
    
    # Удаляем виды без данных
    unique_labels = sorted(list(set(spring_labels)))
    print(f"\n✅ Загружено {len(spring_data)} образцов по {len(unique_labels)} видам")
    print(f"   Виды с данными: {unique_labels}")
    
    return spring_data, spring_labels

def preprocess_spectra(spectra_list):
    """Предобработка спектров для CNN"""
    
    # Находим минимальную длину
    min_length = min(len(spectrum) for spectrum in spectra_list)
    print(f"   Минимальная длина спектра: {min_length}")
    
    # Обрезаем все спектры до минимальной длины и очищаем от NaN
    processed_spectra = []
    for spectrum in spectra_list:
        spectrum_clean = spectrum[~np.isnan(spectrum)]
        if len(spectrum_clean) >= min_length:
            processed_spectra.append(spectrum_clean[:min_length])
    
    # Преобразуем в numpy массив
    X = np.array(processed_spectra)
    
    # Добавляем размерность канала для CNN
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"   Форма данных для CNN: {X.shape}")
    return X

def create_1d_alexnet(input_shape, num_classes, species_names):
    """Создает 1D адаптацию Alexnet для спектральных данных"""
    
    print(f"🧠 СОЗДАНИЕ 1D ALEXNET...")
    print(f"   Входная форма: {input_shape}")
    print(f"   Количество классов: {num_classes}")
    
    inputs = Input(shape=input_shape, name='spectrum_input')
    
    # Первый сверточный блок (аналог оригинального Alexnet)
    x = Conv1D(96, 11, strides=4, activation='relu', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling1D(3, strides=2, name='pool1')(x)
    
    # Второй сверточный блок
    x = Conv1D(256, 5, padding='same', activation='relu', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling1D(3, strides=2, name='pool2')(x)
    
    # Третий сверточный блок
    x = Conv1D(384, 3, padding='same', activation='relu', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    
    # Четвертый сверточный блок
    x = Conv1D(384, 3, padding='same', activation='relu', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    
    # Пятый сверточный блок
    x = Conv1D(256, 3, padding='same', activation='relu', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = MaxPooling1D(3, strides=2, name='pool5')(x)
    
    # Переход к полносвязным слоям
    x = GlobalAveragePooling1D(name='global_pool')(x)
    
    # Полносвязные слои (FC layers)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = BatchNormalization(name='bn_fc2')(x)
    
    # Специализированные ветки для групп видов
    print("   Создание специализированных веток...")
    
    # Ветка для хвойных
    conifer_species = ['ель', 'ель_голубая', 'лиственница', 'сосна', 'туя']
    conifer_branch = Dense(1024, activation='relu', name='conifer_branch')(x)
    conifer_branch = Dropout(0.3)(conifer_branch)
    conifer_branch = BatchNormalization()(conifer_branch)
    
    # Ветка для лиственных
    deciduous_species = ['береза', 'дуб', 'клен', 'клен_ам', 'липа', 'осина', 'ясень', 'каштан', 'орех']
    deciduous_branch = Dense(1024, activation='relu', name='deciduous_branch')(x)
    deciduous_branch = Dropout(0.3)(deciduous_branch)
    deciduous_branch = BatchNormalization()(deciduous_branch)
    
    # Ветка для кустарников и особых видов
    special_species = ['сирень', 'черемуха', 'рябина', 'тополь_черный', 'тополь_бальзамический', 'ива']
    special_branch = Dense(1024, activation='relu', name='special_branch')(x)
    special_branch = Dropout(0.3)(special_branch)
    special_branch = BatchNormalization()(special_branch)
    
    # Объединяем ветки
    from tensorflow.keras.layers import Concatenate
    combined = Concatenate(name='combine_branches')([conifer_branch, deciduous_branch, special_branch])
    
    # Финальные слои
    x = Dense(2048, activation='relu', name='fc_final1')(combined)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    
    x = Dense(1024, activation='relu', name='fc_final2')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # Выходной слой
    outputs = Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Создаем модель
    model = Model(inputs=inputs, outputs=outputs, name='Alexnet1D_20Species')
    
    # Компилируем
    model.compile(
        optimizer=Adam(learning_rate=0.0001, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Выводим архитектуру
    print("\n📋 АРХИТЕКТУРА МОДЕЛИ:")
    model.summary()
    
    return model

def create_data_augmentation(X, y, augmentation_factor=2):
    """Создает аугментированные данные для улучшения обучения"""
    
    print(f"🔄 СОЗДАНИЕ АУГМЕНТИРОВАННЫХ ДАННЫХ (фактор: {augmentation_factor})...")
    
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        spectrum = X[i].flatten()
        label = y[i]
        
        # Оригинальный спектр
        X_aug.append(X[i])
        y_aug.append(label)
        
        for _ in range(augmentation_factor):
            # Добавляем небольшой шум
            noise_level = 0.02
            noisy_spectrum = spectrum + np.random.normal(0, noise_level * np.std(spectrum), spectrum.shape)
            
            # Небольшое масштабирование
            scale_factor = np.random.uniform(0.95, 1.05)
            scaled_spectrum = noisy_spectrum * scale_factor
            
            # Небольшой сдвиг
            shift = np.random.uniform(-0.01, 0.01) * np.mean(spectrum)
            shifted_spectrum = scaled_spectrum + shift
            
            # Добавляем к аугментированным данным
            X_aug.append(shifted_spectrum.reshape(-1, 1))
            y_aug.append(label)
    
    X_augmented = np.array(X_aug)
    y_augmented = np.array(y_aug)
    
    print(f"   Исходных образцов: {len(X)}")
    print(f"   Аугментированных: {len(X_augmented)}")
    
    return X_augmented, y_augmented

def analyze_results(model, X_test, y_test, species_names, history):
    """Анализирует и визуализирует результаты"""
    
    print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ...")
    
    # Предсказания
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Общая точность
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Детальный отчет
    report = classification_report(y_true, y_pred, target_names=species_names, output_dict=True, zero_division=0)
    
    print(f"\n📋 РЕЗУЛЬТАТЫ ПО ВИДАМ:")
    for species in species_names:
        if species in report:
            precision = report[species]['precision']
            recall = report[species]['recall']
            f1 = report[species]['f1-score']
            print(f"   {species:25} P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    # Создаем визуализации
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. История обучения
    ax1 = axes[0, 0]
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Точность обучения', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Потери
    ax2 = axes[0, 1]
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Функция потерь', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    im = ax3.imshow(cm_normalized, cmap='Blues', aspect='auto')
    ax3.set_xticks(range(len(species_names)))
    ax3.set_yticks(range(len(species_names)))
    ax3.set_xticklabels(species_names, rotation=45, ha='right')
    ax3.set_yticklabels(species_names)
    ax3.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Добавляем значения в ячейки
    for i in range(len(species_names)):
        for j in range(len(species_names)):
            value = cm_normalized[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax3.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=8)
    
    plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # 4. Точность по видам
    ax4 = axes[1, 1]
    species_accuracy = []
    for i, species in enumerate(species_names):
        mask = y_true == i
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            species_accuracy.append(acc)
        else:
            species_accuracy.append(0)
    
    bars = ax4.bar(range(len(species_names)), species_accuracy, 
                   color=['green' if acc > 0.8 else 'orange' if acc > 0.5 else 'red' for acc in species_accuracy],
                   alpha=0.8)
    ax4.set_title('Точность по видам', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Виды')
    ax4.set_ylabel('Точность')
    ax4.set_xticks(range(len(species_names)))
    ax4.set_xticklabels(species_names, rotation=45, ha='right')
    ax4.set_ylim(0, 1)
    
    # Добавляем значения на столбцы
    for i, bar in enumerate(bars):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{species_accuracy[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('alexnet_20_species_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, report

def main():
    """Главная функция"""
    
    print("🌲" * 25)
    print("🚀 1D ALEXNET ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ")
    print("🌲" * 25)
    
    start_time = time.time()
    
    # 1. Загрузка данных
    spring_data, spring_labels = load_20_species_data()
    
    if len(spring_data) == 0:
        print("❌ Нет данных для обучения!")
        return
    
    # 2. Предобработка
    print("\n🔧 ПРЕДОБРАБОТКА ДАННЫХ...")
    X = preprocess_spectra(spring_data)
    
    # 3. Подготовка меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(spring_labels)
    species_names = label_encoder.classes_
    y_categorical = to_categorical(y_encoded)
    
    print(f"\n📊 СТАТИСТИКА ДАННЫХ:")
    print(f"   Всего образцов: {len(X)}")
    print(f"   Форма спектра: {X.shape[1:]}")
    print(f"   Количество видов: {len(species_names)}")
    
    # Статистика по видам
    unique, counts = np.unique(y_encoded, return_counts=True)
    for i, (species, count) in enumerate(zip(species_names, counts)):
        print(f"   {species:25} {count:3d} образцов")
    
    # 4. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"\n📈 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   Обучающая выборка: {len(X_train)}")
    print(f"   Тестовая выборка: {len(X_test)}")
    
    # 5. Аугментация данных
    X_train_aug, y_train_aug = create_data_augmentation(X_train, y_train, augmentation_factor=1)
    
    # 6. Нормализация
    print(f"\n🔢 НОРМАЛИЗАЦИЯ ДАННЫХ...")
    
    # Нормализуем по каналам
    scaler = StandardScaler()
    X_train_scaled = X_train_aug.copy()
    X_test_scaled = X_test.copy()
    
    for i in range(X_train_aug.shape[0]):
        X_train_scaled[i, :, 0] = scaler.fit_transform(X_train_aug[i, :, 0].reshape(-1, 1)).flatten()
    
    for i in range(X_test.shape[0]):
        X_test_scaled[i, :, 0] = scaler.fit_transform(X_test[i, :, 0].reshape(-1, 1)).flatten()
    
    # 7. Создание модели
    model = create_1d_alexnet(X_train_scaled.shape[1:], len(species_names), species_names)
    
    # 8. Настройка обучения
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_alexnet_20_species.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # 9. Обучение
    print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ...")
    print(f"   Эпохи: 100")
    print(f"   Batch size: 32")
    print(f"   Аугментированных образцов: {len(X_train_scaled)}")
    
    history = model.fit(
        X_train_scaled, y_train_aug,
        batch_size=32,
        epochs=100,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 10. Анализ результатов
    accuracy, report = analyze_results(model, X_test_scaled, y_test, species_names, history)
    
    # 11. Сохранение
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем модель и метаданные
    model.save(f'alexnet_20_species_final_{timestamp}.keras')
    
    import joblib
    joblib.dump(label_encoder, f'alexnet_20_species_label_encoder_{timestamp}.pkl')
    joblib.dump(scaler, f'alexnet_20_species_scaler_{timestamp}.pkl')
    
    # Сохраняем отчет
    report_text = f"""
🏆 РЕЗУЛЬТАТЫ 1D ALEXNET ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ
==============================================

⏱️  Время обучения: {time.time() - start_time:.1f} секунд
🎯 Общая точность: {accuracy:.3f} ({accuracy*100:.1f}%)
📊 Количество видов: {len(species_names)}
🔢 Обучающих образцов: {len(X_train_scaled)}
🧪 Тестовых образцов: {len(X_test_scaled)}

📋 ДЕТАЛИЗАЦИЯ ПО ВИДАМ:
{chr(10).join([f"   {species:25} P={report[species]['precision']:.3f}, R={report[species]['recall']:.3f}, F1={report[species]['f1-score']:.3f}" 
               for species in species_names if species in report])}

🏆 СТАТУС: {'УСПЕХ' if accuracy > 0.7 else 'ТРЕБУЕТ УЛУЧШЕНИЙ'}

📁 ФАЙЛЫ:
   - alexnet_20_species_final_{timestamp}.keras
   - alexnet_20_species_label_encoder_{timestamp}.pkl
   - alexnet_20_species_scaler_{timestamp}.pkl
   - alexnet_20_species_results.png
    """
    
    with open(f'alexnet_20_species_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 12. Финальный отчет
    total_time = time.time() - start_time
    
    print(f"\n🎉 ОБУЧЕНИЕ 1D ALEXNET ЗАВЕРШЕНО!")
    print(f"⏱️  Общее время: {total_time:.1f} секунд")
    print(f"🏆 Финальная точность: {accuracy:.1%}")
    print(f"📁 Файлы сохранены с timestamp: {timestamp}")
    print(f"🎯 Статус: {'УСПЕХ' if accuracy > 0.7 else 'ТРЕБУЕТ УЛУЧШЕНИЙ'}")
    
    return model, accuracy, report

if __name__ == "__main__":
    main() 