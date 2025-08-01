# Классификация спектров деревьев с использованием нейронных сетей

Эта программа выполняет классификацию спектральных данных различных видов деревьев с использованием глубокого обучения.

## Поддерживаемые виды деревьев

- Береза
- Дуб  
- Ель
- Клен
- Липа
- Осина
- Сосна

## 📊 Результаты исследования (1D-AlexNet)

### ✅ Достигнутые показатели:
- **Точность без шума**: 85.95%
- **Точность при 20% шуме**: 84.86% (потеря всего 1.09%)
- **1000 реализаций** гауссовского шума с нулевым средним
- **Обработано**: 603 спектра из 7 видов растительности

### 🌿 Лучшие результаты по видам:
- **Береза**: 93.3% (вероятность правильной классификации)
- **Осина**: 92.9%
- **Дуб**: 83.3% (FPR = 0.000 - без ложных срабатываний!)

### 🚨 Минимальные ложные тревоги:
- **Дуб**: 0.000 (идеально!)
- **Ель**: 0.009
- **Береза**: 0.013

**📈 График анализа**: `1d_alexnet_noise_analysis.png`

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Структура данных

Программа ожидает следующую структуру папок:

```
Спектры/
├── береза/
│   ├── файл1.xlsx
│   ├── файл2.xlsx
│   └── ...
├── дуб/
│   ├── файл1.xlsx
│   └── ...
├── ель/
├── клен/
├── липа/
├── осина/
└── сосна/
```

Каждый Excel файл должен содержать спектральные данные, где:
- Первый столбец: длина волны (optional)
- Второй столбец: интенсивность спектра

## Запуск программы

### Тестирование данных (рекомендуется)
Перед обучением рекомендуется проверить корректность данных:
```bash
python test_data_loading.py
```

### Основная программа (1D-AlexNet согласно статье)
```bash
python3 main_1d_alexnet.py
```

### Альтернативные версии
```bash
# Scikit-learn версия (если TensorFlow недоступен)
python3 main_sklearn.py

# Базовая TensorFlow версия  
python3 main.py
```

### Предсказание на новых данных
После обучения модели можно использовать её для классификации новых спектров:

```bash
# Предсказание для одного файла
python predict.py --file береза/beresa_001x.xlsx

# Предсказание для всех файлов в папке
python predict.py --folder береза/
```

## Функциональность программы

### 🏆 Основная версия (1D-AlexNet):
1. **Загрузка спектральных данных**: 603 файла из 7 видов растительности
2. **1D-AlexNet архитектура**: Согласно научной статье
3. **1000 реализаций шума**: Гауссовский шум с нулевым средним
4. **Вероятности классификации**: По каждому виду растительности
5. **Анализ ложных тревог**: FPR для каждого класса
6. **Графический анализ**: Устойчивость к различным уровням шума

### 🔧 Альтернативные версии:
1. **Scikit-learn**: Глубокая нейросеть + Random Forest
2. **TensorFlow**: Базовая CNN архитектура
3. **Предобработка**: Автоматическая нормализация данных
4. **Визуализация**: Матрицы ошибок, важность признаков
5. **Сохранение**: Модели и предобработчики

## Файлы проекта

### Основные скрипты
- **`main_1d_alexnet.py`** - 🏆 **ОСНОВНОЙ СКРИПТ** - 1D-AlexNet с 1000 реализациями шума (согласно статье)
- `main_sklearn.py` - альтернативная реализация через scikit-learn  
- `main.py` - базовая версия с TensorFlow
- `predict.py` - скрипт для предсказания на новых данных
- `test_data_loading.py` - тестирование корректности данных
- `requirements.txt` - зависимости проекта

### Выходные файлы (создаются после обучения)

#### 1D-AlexNet версия:
- **`1d_alexnet_noise_analysis.png`** - анализ устойчивости к шуму с 1000 реализациями
- `scaler.pkl` - нормализатор данных
- `label_encoder.pkl` - кодировщик меток классов

#### Scikit-learn версия:
- `tree_classification_neural_network.pkl` - обученная нейросеть
- `tree_classification_random_forest.pkl` - модель Random Forest
- `confusion_matrix_*.png` - матрицы ошибок  
- `feature_importance_*.png` - важность признаков

#### TensorFlow версия:
- `tree_classification_model.h5` - обученная модель
- `training_history.png` - графики обучения

## Архитектура модели

Нейронная сеть включает:
- Входной слой с размерностью, равной длине спектра
- 4 полносвязных слоя с ReLU активацией (512, 256, 128, 64 нейрона)
- Dropout слои для предотвращения переобучения
- Выходной слой с softmax активацией для классификации

## Метрики качества

Программа выводит:
- Точность классификации
- Отчет о классификации (precision, recall, f1-score)
- Матрицу ошибок
- Коэффициент ложных срабатываний (FPR) для каждого класса

## Тестирование с шумом

Модель тестируется с различными уровнями шума (0%, 1%, 5%, 10%, 15%, 20%) для оценки устойчивости к помехам в реальных условиях.

## Требования к системе

- Python 3.7+
- Рекомендуется наличие GPU для ускорения обучения
- Минимум 4 ГБ RAM
- Свободное место на диске: ~1 ГБ

## Примечания

- Все спектры автоматически обрезаются до минимальной длины среди всех загруженных файлов
- Программа автоматически обнаруживает и использует GPU при наличии
- Используется стратифицированное разделение данных (80% обучение, 20% тестирование)
- Применяется ранняя остановка обучения для предотвращения переобучения 