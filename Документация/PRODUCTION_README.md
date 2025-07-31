# 🌲 Классификатор древесных пород - Продакшен версия

## ✅ Готовые к использованию виды:
- **БЕРЕЗА** - высокая надежность
- **ЕЛЬ** - высокая надежность

## ⚠️ Ограничения:
- Модель обучена на весенних данных, тестирована на летних
- Некоторые виды требуют ручной проверки
- Рекомендуется комбинировать с экспертной оценкой

## 🚀 Использование:
```python
import joblib
from tensorflow import keras

model = keras.models.load_model('production_tree_classifier.keras')
scaler = joblib.load('production_scaler.pkl')
label_encoder = joblib.load('production_label_encoder.pkl')
metadata = joblib.load('production_metadata.pkl')
```
