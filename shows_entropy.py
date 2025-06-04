import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy


def compute_entropy(data, n_bins=100):
    hist, _ = np.histogram(data, bins=n_bins, density=True)
    return entropy(hist, base=2)

def mostrar_entropia(datos):
    """ Muestra un gráfico de barras con los valores y la entropía calculada """
    valores, conteo = np.unique(datos, return_counts=True)
    plt.bar(valores, conteo, color=['red', 'blue', 'green', 'yellow'])
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.title(f"Entropía = {compute_entropy(datos,len(datos)):.2f}")
    plt.show()

# 🔹 Ejemplo de datos
datos_baja_entropia = [1, 1, 1, 1]  # Entropía mínima (todos iguales)
datos_alta_entropia = [1, 2, 3, 4]  # Entropía máxima (todos diferentes)
datos_intermedia_entropia = [1, 1, 2, 3]  # Entropía intermedia

# 🔹 Cálculo de entropías
print("Entropía baja:", compute_entropy(datos_baja_entropia, n_bins=len(datos_baja_entropia)))
print("Entropía alta:", compute_entropy(datos_alta_entropia, n_bins=len(datos_alta_entropia)))
print("Entropía intermedia:", compute_entropy(datos_intermedia_entropia, n_bins=len(datos_intermedia_entropia)))

# 🔹 Visualización
mostrar_entropia(datos_baja_entropia)
mostrar_entropia(datos_alta_entropia)
mostrar_entropia(datos_intermedia_entropia)

n_bins = 4
uniform_distribution = np.ones(n_bins) / n_bins  # Probabilidades iguales
max_entropy = entropy(uniform_distribution,base=2)
print(max_entropy)
