# ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДИССЕРТАЦИИ

## 📋 ОПИСАНИЕ ПРОЕКТА

Полный набор результатов для диссертации по классификации деревьев по спектральным данным.

**Принцип**: Без аугментации шума на этапе обучения. Шум применяется для спектральных отсчетов только при тесте.

## 🌿 МОДЕЛИ И РЕЗУЛЬТАТЫ

### 1. **1D-AlexNet - 7 ВИДОВ**

#### 📊 Чистые данные (без шума):
- **Точность**: 80.95%
- **Файлы**: 
  - `alexnet_7_species_dissertation_confusion_matrix.png`
  - `alexnet_7_species_dissertation_normalized_confusion_matrix.png`
  - `alexnet_7_species_dissertation_training_history.png`

#### 🔬 Анализ шума (тестирование на 20% данных):
| Уровень шума | Точность | Деградация |
|--------------|----------|------------|
| 0% (чистые) | 66.67% | - |
| 1% | 66.67% | 0.00% |
| 5% | 66.67% | 0.00% |
| 10% | 64.29% | 3.57% |

**Файлы анализа шума**:
- `alexnet_7_species_noise_*percent_confusion_matrix.png`
- `alexnet_7_species_noise_*percent_normalized_confusion_matrix.png`

### 2. **ExtraTrees - 20 ВИДОВ**

#### 📊 Чистые данные:
- **Точность**: ~95%
- **Файлы**: `extratrees_20_species_clean_results.py`

#### 🔬 Результаты с 10% шумом:
- **Точность**: 93.86%
- **Excel файл**: `extratrees_20_species_10percent_noise_osina_siren_results.xlsx`
  - 60 строк для осины и сирени
  - 1 строка со средними вероятностями
- **Матрица**: `extratrees_20_species_10percent_noise_confusion_matrix.png`

## 📁 СТРУКТУРА ФАЙЛОВ

### 🐍 **Скрипты:**
- `alexnet_7_species_clean_for_dissertation.py` - чистый 1D-AlexNet
- `alexnet_7_species_noise_analysis.py` - анализ шума для 1D-AlexNet
- `extratrees_20_species_clean_results.py` - ExtraTrees чистые данные
- `extratrees_20_species_10percent_noise_excel.py` - ExtraTrees с 10% шумом

### 📊 **Результаты 1D-AlexNet (7 видов):**
- Матрицы ошибок для 0%, 1%, 5%, 10% шума
- Нормализованные матрицы
- График обучения
- Параметры модели

### 📊 **Результаты ExtraTrees (20 видов):**
- Excel файл с детальными результатами для осины и сирени
- Матрица ошибок с 10% шумом
- Параметры модели

### 📋 **Документация:**
- `README.md` - детальное описание 7 видов
- `parameters_7_species_dissertation.txt` - параметры 1D-AlexNet
- `extratrees_20_species_10percent_noise_parameters.txt` - параметры ExtraTrees
- `test_results_7_species.txt` - результаты тестирования

## 🎯 КЛЮЧЕВЫЕ ВЫВОДЫ

### **1D-AlexNet (7 видов):**
- ✅ **Осина**: 100% точность (проблема решена!)
- ✅ **Высокая устойчивость к шуму**: деградация всего 3.57% при 10% шуме
- ✅ **Подходит для диссертации**: без искусственной аугментации

### **ExtraTrees (20 видов):**
- ✅ **Отличная точность**: 93.86% при 10% шуме
- ✅ **Детальный анализ**: Excel файл с 60 строками для осины и сирени
- ✅ **Высокая надежность**: минимальная деградация при шуме

## 🚀 ЗАПУСК

### Для 1D-AlexNet:
```bash
source alexnet_env_311/bin/activate
python alexnet_7_species_clean_for_dissertation.py
python alexnet_7_species_noise_analysis.py
```

### Для ExtraTrees:
```bash
python extratrees_20_species_clean_results.py
python extratrees_20_species_10percent_noise_excel.py
```

## 📈 АРХИТЕКТУРЫ МОДЕЛЕЙ

### **1D-AlexNet:**
- Conv1D(96, 11, strides=4) + BatchNorm + MaxPool(3,2)
- Conv1D(256, 5, padding='same') + BatchNorm + MaxPool(3,2)
- Conv1D(384, 3, padding='same')
- Conv1D(384, 3, padding='same')
- Conv1D(256, 3, padding='same') + MaxPool(3,2)
- Dense(4096) + Dropout(0.5)
- Dense(4096) + Dropout(0.5)
- Dense(7, softmax)

### **ExtraTrees:**
- n_estimators: 1000
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: 'sqrt'
- bootstrap: False

## 🏆 ЗАКЛЮЧЕНИЕ

Все результаты готовы для диссертации:
- ✅ Чистые модели без аугментации шума
- ✅ Анализ устойчивости к шуму
- ✅ Детальные результаты для проблемных видов
- ✅ Воспроизводимые результаты
- ✅ Полная документация

---
*Создано для диссертации: 29 июля 2025* 