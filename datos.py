"""
datos.py
========
Datos compartidos por la ANN y la SNN.

Para comparar las dos redes de forma justa, las dos van a aprender
EXACTAMENTE la misma tarea: reconocer 4 "dibujos" de 5x5 píxeles
(los números 0, 1, 2 y 3).

Cada dibujo es una cuadrícula de 5 filas x 5 columnas:
    1 = píxel encendido (negro)
    0 = píxel apagado  (blanco)

Como las redes neuronales trabajan con listas de números, "aplanamos"
cada dibujo en una lista de 25 números (5 * 5 = 25).
"""

import numpy as np  # numpy: la librería básica de Python para trabajar con números y matrices

# ---------------------------------------------------------------------------
# 1) Los dibujos "limpios" (sin ruido)
# ---------------------------------------------------------------------------
# Esto es un diccionario: asocia un nombre (clave) con un valor.
# Aquí la clave es el nombre del dígito y el valor es su dibujo 5x5.
DIBUJOS = {
    "0": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
    ],
    "1": [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ],
    "2": [
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    "3": [
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
    ],
}

# Lista con los nombres de las clases, en orden: ["0", "1", "2", "3"]
NOMBRES_CLASES = list(DIBUJOS.keys())
NUM_CLASES = len(NOMBRES_CLASES)  # 4
NUM_PIXELES = 25                  # 5 x 5


def dibujos_limpios():
    """
    Devuelve dos cosas:
      X -> matriz de forma (4, 25): una fila por dibujo, 25 píxeles por fila
      y -> vector de forma (4,)   : la clase de cada fila (0, 1, 2, 3)
    """
    X = []
    y = []
    # enumerate() nos da a la vez la posición (indice) y el valor (nombre)
    for indice, nombre in enumerate(NOMBRES_CLASES):
        imagen = np.array(DIBUJOS[nombre])   # lista de listas -> matriz numpy 5x5
        X.append(imagen.flatten())           # flatten(): 5x5 -> 25 números seguidos
        y.append(indice)
    return np.array(X, dtype=float), np.array(y)


def agregar_ruido(X, prob_cambio, generador):
    """
    Cambia al azar algunos píxeles (0 -> 1 o 1 -> 0).

    prob_cambio: probabilidad de que cada píxel se invierta (ej. 0.1 = 10 %).
    generador:   generador de números aleatorios de numpy (para que los
                 resultados sean repetibles si usamos la misma "semilla").
    """
    # generador.random(forma) crea números aleatorios entre 0 y 1.
    # Comparamos con prob_cambio para obtener True/False en cada píxel.
    mascara = generador.random(X.shape) < prob_cambio
    X_ruidoso = X.copy()
    # Donde la máscara es True, invertimos el píxel: 1 - 0 = 1, 1 - 1 = 0
    X_ruidoso[mascara] = 1 - X_ruidoso[mascara]
    return X_ruidoso


def crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=0):
    """
    Crea un conjunto de datos con muchas copias "ruidosas" de cada dibujo.
    Así la red no puede simplemente memorizar 4 dibujos: tiene que
    aprender a reconocerlos aunque vengan un poco estropeados.
    """
    generador = np.random.default_rng(semilla)
    X_base, y_base = dibujos_limpios()

    # np.repeat repite cada fila 'ejemplos_por_clase' veces
    X = np.repeat(X_base, ejemplos_por_clase, axis=0)
    y = np.repeat(y_base, ejemplos_por_clase)
    X = agregar_ruido(X, prob_cambio, generador)

    # Mezclamos el orden para que no vengan todos los "0" juntos, etc.
    orden = generador.permutation(len(y))
    return X[orden], y[orden]


def mostrar_dibujo(fila_de_25):
    """Imprime un dibujo 5x5 en la terminal usando '#' y '.'."""
    imagen = np.array(fila_de_25).reshape(5, 5)  # 25 números -> 5x5
    for fila in imagen:
        print(" ".join("#" if p > 0.5 else "." for p in fila))


# Este bloque solo se ejecuta si lanzas este archivo directamente:
#     python datos.py
# Si otro archivo hace "import datos", este bloque NO se ejecuta.
if __name__ == "__main__":
    X, y = dibujos_limpios()
    for fila, clase in zip(X, y):
        print(f"Dígito {NOMBRES_CLASES[clase]}:")
        mostrar_dibujo(fila)
        print()

    print("Ejemplo de un '3' con ruido:")
    gen = np.random.default_rng(1)
    mostrar_dibujo(agregar_ruido(X[3:4], 0.1, gen)[0])
