import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_noise_implementation():
    """Анализ реализации шума в нашем коде"""
    print("=== АНАЛИЗ РЕАЛИЗАЦИИ ШУМА ===")
    
    # Создаем тестовый спектр
    np.random.seed(42)
    test_spectrum = np.random.normal(0, 1, 1000)  # Нормализованный спектр
    print(f"Тестовый спектр: mean={np.mean(test_spectrum):.4f}, std={np.std(test_spectrum):.4f}")
    
    print("\n=== ТЕКУЩАЯ РЕАЛИЗАЦИЯ ===")
    print("В нашем коде используется:")
    print("noise = np.random.normal(0, noise_level, data.shape)")
    print("где noise_level - это процент от std(data)")
    
    # Текущая реализация
    noise_level = 0.1  # 10%
    std_dev = noise_level / 100.0 * np.std(test_spectrum)
    noise_current = np.random.normal(0, std_dev, test_spectrum.shape)
    spectrum_with_current = test_spectrum + noise_current
    
    print(f"\nТекущая реализация (10% шума):")
    print(f"  • std_dev = {noise_level/100.0} * {np.std(test_spectrum):.4f} = {std_dev:.6f}")
    print(f"  • std шума: {np.std(noise_current):.6f}")
    print(f"  • std спектра с шумом: {np.std(spectrum_with_current):.6f}")
    
    print("\n=== ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ ===")
    print("Должно быть:")
    print("noise = np.random.normal(0, noise_level * std(data), data.shape)")
    print("где noise_level - это коэффициент (0.1 для 10%)")
    
    # Правильная реализация
    noise_correct = np.random.normal(0, noise_level * np.std(test_spectrum), test_spectrum.shape)
    spectrum_with_correct = test_spectrum + noise_correct
    
    print(f"\nПравильная реализация (10% шума):")
    print(f"  • std шума = {noise_level} * {np.std(test_spectrum):.4f} = {noise_level * np.std(test_spectrum):.6f}")
    print(f"  • std шума: {np.std(noise_correct):.6f}")
    print(f"  • std спектра с шумом: {np.std(spectrum_with_correct):.6f}")
    
    print("\n=== СРАВНЕНИЕ ===")
    current_std = np.std(noise_current)
    correct_std = np.std(noise_correct)
    ratio = current_std / correct_std
    
    print(f"Текущий std шума: {current_std:.6f}")
    print(f"Правильный std шума: {correct_std:.6f}")
    print(f"Отношение: {ratio:.2f}x")
    print(f"Проблема: Текущий шум в {ratio:.1f} раз меньше правильного!")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Оригинальный спектр
    axes[0, 0].plot(test_spectrum[:100])
    axes[0, 0].set_title('Оригинальный спектр')
    axes[0, 0].set_ylabel('Амплитуда')
    
    # Текущий шум
    axes[0, 1].plot(noise_current[:100])
    axes[0, 1].set_title(f'Текущий шум (std={current_std:.4f})')
    axes[0, 1].set_ylabel('Амплитуда')
    
    # Правильный шум
    axes[1, 0].plot(noise_correct[:100])
    axes[1, 0].set_title(f'Правильный шум (std={correct_std:.4f})')
    axes[1, 0].set_ylabel('Амплитуда')
    
    # Сравнение спектров
    axes[1, 1].plot(test_spectrum[:100], label='Оригинал', alpha=0.7)
    axes[1, 1].plot(spectrum_with_current[:100], label='Текущий шум', alpha=0.7)
    axes[1, 1].plot(spectrum_with_correct[:100], label='Правильный шум', alpha=0.7)
    axes[1, 1].set_title('Сравнение спектров')
    axes[1, 1].set_ylabel('Амплитуда')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/Анализ_реализации_шума.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ График сохранен: ФИНАЛЬНЫЕ_РЕЗУЛЬТАТЫ/Анализ_реализации_шума.png")
    
    return current_std, correct_std, ratio

def fix_noise_implementation():
    """Показывает правильную реализацию шума"""
    print("\n=== ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ ШУМА ===")
    
    correct_code = '''
def add_gaussian_noise_correct(data, noise_level):
    """Правильное добавление гауссовского шума"""
    if noise_level == 0:
        return data
    
    # noise_level - это коэффициент (0.1 для 10%)
    std_dev = noise_level * np.std(data)
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

# Использование:
# Для 10% шума: add_gaussian_noise_correct(data, 0.1)
# Для 5% шума: add_gaussian_noise_correct(data, 0.05)
# Для 1% шума: add_gaussian_noise_correct(data, 0.01)
'''
    
    print(correct_code)
    
    print("=== ОБЪЯСНЕНИЕ ПРОБЛЕМЫ ===")
    print("1. В текущем коде: std_dev = noise_level / 100.0 * np.std(data)")
    print("2. Это означает, что для 10% шума: std_dev = 0.1 / 100.0 * std = 0.001 * std")
    print("3. Правильно должно быть: std_dev = 0.1 * std")
    print("4. Текущий шум в 100 раз меньше правильного!")
    
    return correct_code

if __name__ == "__main__":
    # Анализируем текущую реализацию
    current_std, correct_std, ratio = analyze_noise_implementation()
    
    # Показываем правильную реализацию
    fix_noise_implementation()
    
    print(f"\n=== ВЫВОД ===")
    print(f"❌ Проблема найдена: шум в {ratio:.1f} раз меньше правильного")
    print(f"✅ Решение: изменить реализацию шума")
    print(f"✅ Это объясняет низкую точность при шуме") 