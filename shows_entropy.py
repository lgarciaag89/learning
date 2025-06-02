import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


def create_scatter_plot(data, x_col, y_col,corr_value, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[x_col], y=data[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # Calcular la regresión lineal
    slope, intercept, _, _, _ = linregress(data[x_col], data[y_col])
    line_x = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    line_y = slope * line_x + intercept

    # Graficar la línea de tendencia
    plt.plot(line_x, line_y, color='red', linestyle='dashed', label=f'Correlación = {corr_value:.2f}')

    plt.grid()
    plt.show()

path = "data/TR_starPep_AB_training.fasta_AAC_class.csv"
target = 'Class'
data = pd.read_csv(path)
print("Original data shape:", data.shape)

data[target] = data[target].map({'ABP': 1, 'NoNABP': 0})

correlations = data.corr()

# 1. Encontrar la variable más correlacionada con la clase y su valor
most_correlated_with_class = correlations[target].drop(target).idxmax()
most_correlated_value = correlations[target][most_correlated_with_class]

correlations = correlations.drop(columns=[target],index=[target])
# Aplicar la máscara triangular superior
upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
# Desenrollar la matriz y ordenar los valores
sorted_correlations = upper.unstack().dropna().sort_values(ascending=False)

# Seleccionar los pares de mayor y menor correlación
most_correlated_pair = sorted_correlations.index[0]
most_correlated_value_pair = sorted_correlations.iloc[0]

least_correlated_pair = sorted_correlations.index[-1]
least_correlated_value = sorted_correlations.iloc[-1]

create_scatter_plot(data, most_correlated_with_class, target, most_correlated_value,
                    f'Scatter plot: {most_correlated_with_class} vs {target}=({most_correlated_value})')
create_scatter_plot(data, most_correlated_pair[0], most_correlated_pair[1],most_correlated_value_pair,
                    f'Scatter plot: {most_correlated_pair[0]} - {most_correlated_pair[1]}={most_correlated_value_pair}')
create_scatter_plot(data, least_correlated_pair[0], least_correlated_pair[1],least_correlated_value,
                    f'Scatter plot: {least_correlated_pair[0]} - {least_correlated_pair[1]}={least_correlated_value}')

np.random.seed(42)
horas_estudio = np.random.normal(5, 2, 100)
puntuacion_examen = horas_estudio * 10 + np.random.normal(0, 5, 100)

# Crear DataFrame
data = pd.DataFrame({'Horas de Estudio': horas_estudio, 'Puntuación Examen': puntuacion_examen})

# Calcular la correlación
correlacion = data.corr().iloc[0, 1]
create_scatter_plot(data, 'Horas de Estudio', 'Puntuación Examen', correlacion,
                    f'Scatter plot: Horas de Estudio vs Puntuación Examen={correlacion:.2f}')
print(f"Correlación entre horas de estudio y puntuación en examen: {correlacion:.2f}")
