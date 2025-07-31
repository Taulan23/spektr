import joblib
import numpy as np

# Загружаем модель
model = joblib.load("./extra_trees_20_species_model_20250724_110036.pkl")

print("Классы в модели:")
print(model.classes_)
print(f"Количество классов: {len(model.classes_)}")

# Создаем тестовые данные
test_data = np.random.rand(1, 44)  # 44 признака
probabilities = model.predict_proba(test_data)
print(f"Форма вероятностей: {probabilities.shape}")
print(f"Количество классов в вероятностях: {probabilities.shape[1]}")

# Проверяем наши названия классов
tree_types = [
    'береза', 'дуб', 'ель', 'ель_голубая', 'ива', 'каштан', 'клен', 
    'клен_ам', 'липа', 'лиственница', 'орех', 'осина', 'рябина', 
    'сирень', 'сосна', 'тополь_бальзамический', 'тополь_черный', 
    'туя', 'черемуха', 'ясень'
]

print(f"\nНаши названия классов: {len(tree_types)}")
for i, name in enumerate(tree_types):
    print(f"{i}: {name}") 