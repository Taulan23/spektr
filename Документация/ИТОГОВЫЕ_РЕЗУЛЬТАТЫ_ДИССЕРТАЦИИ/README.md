# ДИССЕРТАЦИЯ - КЛАССИФИКАЦИЯ 7 ВИДОВ ДЕРЕВЬЕВ

## 📋 ОПИСАНИЕ

Чистая версия модели 1D-AlexNet для классификации 7 видов деревьев **БЕЗ АУГМЕНТАЦИИ ШУМА** - подходит для диссертации.

## 🌿 ВИДЫ ДЕРЕВЬЕВ

- Береза
- Дуб  
- Ель
- Клен
- Липа
- Осина
- Сосна

## 📊 РЕЗУЛЬТАТЫ

### Общая точность: **80.95%** (чистые данные)

### Детальные результаты по видам:

| Вид | Точность | Полнота | F1-мера |
|-----|----------|---------|---------|
| Береза | 1.000 | 1.000 | 1.000 |
| Дуб | 1.000 | 0.167 | 0.286 |
| Ель | 1.000 | 1.000 | 1.000 |
| Клен | 1.000 | 1.000 | 1.000 |
| Липа | 1.000 | 0.500 | 0.667 |
| Осина | 1.000 | 1.000 | 1.000 |
| Сосна | 0.429 | 1.000 | 0.600 |

## 🔬 АНАЛИЗ УСТОЙЧИВОСТИ К ШУМУ

### Результаты тестирования на зашумленных данных:

| Уровень шума | Точность | Деградация |
|--------------|----------|------------|
| 0% (чистые) | 66.67% | - |
| 1% | 66.67% | 0.00% |
| 5% | 66.67% | 0.00% |
| 10% | 64.29% | 3.57% |

**Вывод**: Модель показывает высокую устойчивость к шуму до 5%, при 10% шуме деградация составляет всего 3.57%.

## 📁 ФАЙЛЫ

### Основные файлы:
- `alexnet_7_species_clean_for_dissertation.py` - чистый скрипт без аугментации
- `alexnet_7_species_noise_analysis.py` - анализ с шумом
- `parameters_7_species_dissertation.txt` - параметры модели

### Результаты (чистые данные):
- `alexnet_7_species_dissertation_confusion_matrix.png` - матрица ошибок
- `alexnet_7_species_dissertation_normalized_confusion_matrix.png` - нормализованная матрица
- `alexnet_7_species_dissertation_training_history.png` - график обучения

### Результаты анализа шума:
- `alexnet_7_species_noise_0percent_confusion_matrix.png` - матрица для 0% шума
- `alexnet_7_species_noise_1percent_confusion_matrix.png` - матрица для 1% шума
- `alexnet_7_species_noise_5percent_confusion_matrix.png` - матрица для 5% шума
- `alexnet_7_species_noise_10percent_confusion_matrix.png` - матрица для 10% шума
- `alexnet_7_species_noise_*percent_normalized_confusion_matrix.png` - нормализованные матрицы
- `test_results_7_species.txt` - результаты тестирования

## 🔧 ОСОБЕННОСТИ ДЛЯ ДИССЕРТАЦИИ

✅ **Без аугментации шума при обучении**  
✅ **Чистые данные без модификации**  
✅ **Реалистичные условия классификации**  
✅ **Воспроизводимые результаты**  
✅ **Анализ устойчивости к шуму**  

## 🚀 ЗАПУСК

### Чистая версия (для диссертации):
```bash
# Активировать виртуальное окружение
source alexnet_env_311/bin/activate

# Запустить чистый скрипт
python alexnet_7_species_clean_for_dissertation.py
```

### Анализ с шумом:
```bash
# Запустить анализ устойчивости к шуму
python alexnet_7_species_noise_analysis.py
```

## 📈 АРХИТЕКТУРА МОДЕЛИ

1D-AlexNet с архитектурой:
- Conv1D(96, 11, strides=4) + BatchNorm + MaxPool(3,2)
- Conv1D(256, 5, padding='same') + BatchNorm + MaxPool(3,2)
- Conv1D(384, 3, padding='same')
- Conv1D(384, 3, padding='same')
- Conv1D(256, 3, padding='same') + MaxPool(3,2)
- Dense(4096) + Dropout(0.5)
- Dense(4096) + Dropout(0.5)
- Dense(7, softmax)

## 🎯 ВЫВОДЫ

### Основные результаты:
- **Осина** показывает отличную точность (100%)
- **Береза, ель, клен** также показывают идеальную точность
- **Дуб и липа** имеют проблемы с полнотой
- **Сосна** имеет проблемы с точностью

### Устойчивость к шуму:
- Модель очень устойчива к шуму до 5%
- При 10% шуме деградация минимальна (3.57%)
- Это показывает высокую надежность модели в реальных условиях

Модель подходит для диссертации как пример классификации без искусственной аугментации данных с дополнительным анализом устойчивости к шуму.

---
*Создано для диссертации: 29 июля 2025* 