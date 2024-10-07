import pandas as pd
import numpy as np
import math

# Función para calcular la entropía


def calcular_entropia(columna):
    elementos, cuentas = np.unique(columna, return_counts=True)
    entropia = 0
    for cuenta in cuentas:
        probabilidad = cuenta / sum(cuentas)
        entropia -= probabilidad * math.log2(probabilidad)
    return entropia

# Función para calcular la ganancia de información


def calcular_ganancia_informacion(df, atributo, objetivo):
    entropia_total = calcular_entropia(df[objetivo])

    # Calcular la entropía ponderada del atributo
    valores_atributo, cuentas_atributo = np.unique(
        df[atributo], return_counts=True)
    entropia_atributo = 0
    for i, valor in enumerate(valores_atributo):
        subset = df[df[atributo] == valor]
        entropia_atributo += (cuentas_atributo[i] / sum(
            cuentas_atributo)) * calcular_entropia(subset[objetivo])

    ganancia_informacion = entropia_total - entropia_atributo
    return ganancia_informacion


# Cargar el dataset CSV
df = pd.read_csv('personas.csv')

# Especifica la columna de clase objetivo y los atributos para analizar
columna_objetivo = 'talla_ropa'
atributos = ['altura', 'peso', 'talla']

# Calcular la entropía total del dataset
entropia_total = calcular_entropia(df[columna_objetivo])
print(f"Entropía total de '{columna_objetivo}': {entropia_total}\n")

# Calcular la ganancia de información para cada atributo
for atributo in atributos:
    ganancia = calcular_ganancia_informacion(df, atributo, columna_objetivo)
    print(f"Ganancia de información para '{atributo}': {ganancia}")
