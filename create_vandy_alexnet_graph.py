import matplotlib.pyplot as plt
import numpy as np

# Данные для графика
classes = ['береза', 'дуб', 'ель', 'клен', 'липа', 'осина', 'сосна']
accuracy_no_noise = [100, 100, 100, 100, 93.3, 100, 100]
accuracy_with_noise = [100, 100, 100, 100, 93.3, 100, 100]
confidence_with_noise = [100, 100, 100, 100, 99.9, 99.3, 100]

# Создаем график
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# График 1: Точность без шума и с шумом
x = np.arange(len(classes))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracy_no_noise, width, label='Без шума', color='skyblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, accuracy_with_noise, width, label='С 1.1% шумом', color='lightcoral', alpha=0.8)

ax1.set_xlabel('Классы деревьев', fontsize=12, fontweight='bold')
ax1.set_ylabel('Точность (%)', fontsize=12, fontweight='bold')
ax1.set_title('Точность классификации Vandy AlexNet', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(classes, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(85, 105)

# Добавляем значения на столбцы
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height}%', ha='center', va='bottom', fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height}%', ha='center', va='bottom', fontweight='bold')

# График 2: Уверенность модели с шумом
bars3 = ax2.bar(classes, confidence_with_noise, color='lightgreen', alpha=0.8)

ax2.set_xlabel('Классы деревьев', fontsize=12, fontweight='bold')
ax2.set_ylabel('Средняя уверенность (%)', fontsize=12, fontweight='bold')
ax2.set_title('Уверенность модели с 1.1% шумом', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(95, 102)

# Добавляем значения на столбцы
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('vandy_alexnet_results_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print('График сохранен как vandy_alexnet_results_comparison.png') 