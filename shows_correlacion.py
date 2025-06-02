import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Crear un DataFrame de ejemplo
data = pd.DataFrame({
    'Variable X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Variable Y': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
})

# Calcular la correlación
correlation = data.corr()

# Visualizar con un mapa de calor
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de calor de la correlación')
plt.show()