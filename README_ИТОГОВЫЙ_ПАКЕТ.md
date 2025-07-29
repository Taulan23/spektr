# ИТОГОВЫЙ ПАКЕТ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ ДЕРЕВЬЕВ

## 📦 СОДЕРЖИМОЕ ПАКЕТА

### 🎯 ДЛЯ 20 ВИДОВ ДЕРЕВЬЕВ (Extra Trees)

#### Основные файлы:
1. **`extratrees_20_species_clean_results.py`** - основной скрипт классификации
2. **`parameters_20_species_extratrees.txt`** - параметры для воспроизведения результатов

#### Матрицы ошибок:
3. **`confusion_matrix_20_species_1percent.png`** - матрица для 1% шума
4. **`confusion_matrix_20_species_5percent.png`** - матрица для 5% шума  
5. **`confusion_matrix_20_species_10percent.png`** - матрица для 10% шума

### 🌿 ДЛЯ 7 ВИДОВ ДЕРЕВЬЕВ (1D-AlexNet)

#### Основные файлы:
6. **`alexnet_7_species_no_noise_final.py`** - основной скрипт классификации

#### Результаты:
7. **`alexnet_7_species_improved_confusion_matrix_20250728_173602.png`** - матрица ошибок
8. **`alexnet_7_species_improved_training_history_20250728_173602.png`** - график обучения
9. **`alexnet_7_species_80_20_confusion_matrix_20250728_175213.png`** - матрица (80/20 split)
10. **`alexnet_7_species_80_20_training_history_20250728_175213.png`** - график (80/20 split)

## 🚀 КАК ИСПОЛЬЗОВАТЬ

### Для 20 видов:
```bash
python3 extratrees_20_species_clean_results.py
```

### Для 7 видов (требует TensorFlow):
```bash
# Активировать виртуальное окружение
source alexnet_env_311/bin/activate

# Запустить скрипт
python alexnet_7_species_no_noise_final.py
```

## 📊 РЕЗУЛЬТАТЫ

### 20 видов (Extra Trees):
- **0% шум**: 95.26%
- **1% шум**: 95.26%
- **5% шум**: 95.44%
- **10% шум**: 94.56%

### 7 видов (1D-AlexNet):
- **Точность**: ~83-85%
- **Проблемные виды**: липа, сосна
- **Лучшие виды**: береза, дуб, осина

## 🔧 ТРЕБОВАНИЯ

### Для 20 видов:
- Python 3.x
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- openpyxl

### Для 7 видов:
- Python 3.11
- TensorFlow 2.15.0
- Все вышеперечисленные библиотеки

## 📝 ВОСПРОИЗВОДИМОСТЬ

Все результаты воспроизводимы благодаря:
- Фиксированным random_state (42)
- Сохраненным параметрам моделей
- Документированным процессам предобработки

---
*Пакет создан: 29 июля 2025* 