"""
PASO 1 - Una sola neurona artificial (la pieza básica de una ANN)
==================================================================

Antes de construir una red entera, veamos UNA neurona artificial.

Una neurona artificial hace solo tres cosas:

    1. Multiplica cada entrada por un "peso" (lo importante que es esa entrada).
    2. Suma todo y le añade un "sesgo" (bias): un número que desplaza el resultado.
    3. Pasa esa suma por una "función de activación" que la convierte
       en la salida final (aquí usamos la sigmoide, que da un número entre 0 y 1).

            entrada1 --(peso1)--\
                                 \
            entrada2 --(peso2)----[ suma + sesgo ]--> sigmoide --> salida (0..1)

IMPORTANTE: la salida es un número CONTINUO (0.73, 0.12...) y se calcula
de una sola vez. No hay "tiempo". Esta es la gran diferencia con la SNN
que veremos en el paso 3.

Aquí vamos a enseñarle a la neurona la puerta lógica AND:
    0 AND 0 = 0,   0 AND 1 = 0,   1 AND 0 = 0,   1 AND 1 = 1

Ejecuta:  python paso1_neurona_artificial.py
"""

import numpy as np


def sigmoide(z):
    """
    Función de activación sigmoide:  1 / (1 + e^(-z))
    - Si z es muy negativo -> salida cercana a 0
    - Si z = 0             -> salida 0.5
    - Si z es muy positivo -> salida cercana a 1
    """
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Datos: las 4 combinaciones posibles de dos entradas y la respuesta correcta
# ---------------------------------------------------------------------------
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)
respuestas_correctas = np.array([0, 0, 0, 1], dtype=float)

# ---------------------------------------------------------------------------
# Parámetros de la neurona. Empiezan al azar: la neurona "no sabe nada".
# ---------------------------------------------------------------------------
generador = np.random.default_rng(42)       # semilla fija -> resultados repetibles
pesos = generador.normal(0, 0.5, size=2)    # un peso por cada entrada
sesgo = 0.0
tasa_aprendizaje = 0.5                       # cuánto corregimos en cada paso

print("Pesos iniciales:", pesos, " sesgo:", sesgo)

# ---------------------------------------------------------------------------
# Entrenamiento (descenso por gradiente)
# ---------------------------------------------------------------------------
# Idea: mirar el error de la neurona y mover los pesos un poquito
# en la dirección que reduce ese error. Repetirlo muchas veces.
for epoca in range(2000):             # una "época" = pasar una vez por todos los datos
    for x, objetivo in zip(entradas, respuestas_correctas):
        # --- 1) Hacia delante: calcular la salida ---
        z = np.dot(pesos, x) + sesgo   # np.dot = peso1*x1 + peso2*x2
        salida = sigmoide(z)

        # --- 2) ¿Cuánto nos hemos equivocado? ---
        error = salida - objetivo      # positivo si nos pasamos, negativo si nos quedamos cortos

        # --- 3) Corregir los pesos ---
        # Con la función de pérdida "entropía cruzada" + sigmoide, la
        # derivada del error respecto a z es simplemente (salida - objetivo).
        # Y la derivada respecto a cada peso es esa cantidad * su entrada.
        pesos = pesos - tasa_aprendizaje * error * x
        sesgo = sesgo - tasa_aprendizaje * error

    if epoca % 500 == 0:
        # Calculamos el error medio en todos los ejemplos para ver el progreso
        salidas = sigmoide(entradas @ pesos + sesgo)   # @ = multiplicación de matrices
        error_medio = np.mean(np.abs(salidas - respuestas_correctas))
        print(f"Época {epoca:4d}  error medio = {error_medio:.4f}")

print("\nPesos aprendidos:", np.round(pesos, 2), " sesgo:", round(sesgo, 2))
print("\nResultado final:")
for x, objetivo in zip(entradas, respuestas_correctas):
    salida = sigmoide(np.dot(pesos, x) + sesgo)
    print(f"  {int(x[0])} AND {int(x[1])} -> salida = {salida:.3f}  "
          f"(redondeado {int(round(salida))}, correcto {int(objetivo)})")

print("""
Fíjate: la neurona ha aprendido pesos positivos grandes y un sesgo muy
negativo. Solo cuando AMBAS entradas son 1 la suma supera el sesgo y
la salida se acerca a 1.

Una sola neurona NO puede aprender cosas más complicadas (por ejemplo XOR,
o reconocer dígitos). Para eso juntamos muchas neuronas en capas: eso es
una red neuronal artificial (ANN). -> Paso 2.
""")
