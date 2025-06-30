import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse

def load_model_and_preprocessors():
    """Загружает обученную модель и предобработчики"""
    try:
        model = keras.models.load_model('tree_classification_model.h5')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, scaler, label_encoder
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл {e.filename}")
        print("Убедитесь, что вы запустили обучение модели (main.py)")
        return None, None, None

def preprocess_spectrum(spectrum_data, scaler, target_length):
    """Предобрабатывает один спектр"""
    # Удаляем NaN значения
    spectrum_data = spectrum_data[~np.isnan(spectrum_data)]
    
    # Обрезаем или дополняем до нужной длины
    if len(spectrum_data) >= target_length:
        spectrum_data = spectrum_data[:target_length]
    else:
        # Дополняем нулями, если спектр короче
        padding = target_length - len(spectrum_data)
        spectrum_data = np.pad(spectrum_data, (0, padding), mode='constant')
    
    # Нормализуем
    spectrum_data = spectrum_data.reshape(1, -1)
    spectrum_data = scaler.transform(spectrum_data)
    
    return spectrum_data

def predict_single_file(file_path, model, scaler, label_encoder):
    """Предсказывает класс для одного файла"""
    try:
        # Читаем Excel файл
        df = pd.read_excel(file_path)
        
        if df.shape[1] < 2:
            print(f"Ошибка: Файл {file_path} должен содержать минимум 2 столбца")
            return None
        
        # Берем спектральные данные из второго столбца
        spectrum_data = df.iloc[:, 1].values
        
        # Определяем длину спектра (используем размер входа модели)
        input_shape = model.input_shape[1]
        
        # Предобрабатываем спектр
        processed_spectrum = preprocess_spectrum(spectrum_data, scaler, input_shape)
        
        # Делаем предсказание
        prediction = model.predict(processed_spectrum, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        # Декодируем класс
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_class, confidence, prediction[0]
        
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

def predict_folder(folder_path, model, scaler, label_encoder):
    """Предсказывает классы для всех файлов в папке"""
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    
    if not excel_files:
        print(f"В папке {folder_path} не найдено Excel файлов")
        return
    
    print(f"\nОбработка {len(excel_files)} файлов в папке {folder_path}")
    print("-" * 60)
    
    results = []
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        result = predict_single_file(file_path, model, scaler, label_encoder)
        
        if result:
            predicted_class, confidence, probabilities = result
            results.append({
                'file': file,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            print(f"Файл: {file}")
            print(f"Предсказанный класс: {predicted_class}")
            print(f"Уверенность: {confidence:.4f}")
            
            # Показываем вероятности для всех классов
            print("Вероятности для всех классов:")
            for i, class_name in enumerate(label_encoder.classes_):
                print(f"  {class_name}: {probabilities[i]:.4f}")
            print("-" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Предсказание видов деревьев по спектральным данным')
    parser.add_argument('--file', type=str, help='Путь к Excel файлу для предсказания')
    parser.add_argument('--folder', type=str, help='Путь к папке с Excel файлами')
    
    args = parser.parse_args()
    
    # Загружаем модель и предобработчики
    print("Загрузка модели и предобработчиков...")
    model, scaler, label_encoder = load_model_and_preprocessors()
    
    if model is None:
        return
    
    print("Модель успешно загружена!")
    print(f"Поддерживаемые классы: {', '.join(label_encoder.classes_)}")
    
    if args.file:
        # Предсказание для одного файла
        print(f"\nОбработка файла: {args.file}")
        result = predict_single_file(args.file, model, scaler, label_encoder)
        
        if result:
            predicted_class, confidence, probabilities = result
            print("-" * 60)
            print(f"Предсказанный класс: {predicted_class}")
            print(f"Уверенность: {confidence:.4f}")
            
            print("\nВероятности для всех классов:")
            for i, class_name in enumerate(label_encoder.classes_):
                print(f"  {class_name}: {probabilities[i]:.4f}")
    
    elif args.folder:
        # Предсказание для папки
        predict_folder(args.folder, model, scaler, label_encoder)
    
    else:
        print("Использование:")
        print("  python predict.py --file путь/к/файлу.xlsx")
        print("  python predict.py --folder путь/к/папке")
        print("\nПример:")
        print("  python predict.py --file береза/beresa_001x.xlsx")
        print("  python predict.py --folder береза/")

if __name__ == "__main__":
    main() 