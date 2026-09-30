"""
PASO 2 - Red Neuronal Artificial (ANN) desde cero
==================================================

Ahora juntamos muchas neuronas como las del paso 1 en CAPAS:

    Capa de entrada      Capa oculta          Capa de salida
    (25 píxeles)         (16 neuronas)        (4 neuronas: "0","1","2","3")

      x1  ─┐
      x2  ─┼──(pesos W1)──>  h1..h16  ──(pesos W2)──>  s0 s1 s2 s3
      ...  │
      x25 ─┘

- Cada neurona oculta recibe los 25 píxeles, hace "suma ponderada + sesgo"
  y aplica una activación (sigmoide).
- Cada neurona de salida recibe las 16 neuronas ocultas y da una
  "puntuación" para su dígito. Con la función softmax convertimos las
  4 puntuaciones en probabilidades que suman 1.
- La respuesta de la red es el dígito con mayor probabilidad.

¿Cómo aprende? Con BACKPROPAGATION (retropropagación del error):
    1. Hacia delante: calculamos la salida.
    2. Medimos el error (función de pérdida).
    3. Hacia atrás: usando derivadas (regla de la cadena) calculamos
       cuánto ha contribuido CADA peso al error.
    4. Movemos cada peso un poquito en la dirección que reduce el error.

Todo esto funciona porque la sigmoide y la softmax son funciones SUAVES
(se pueden derivar). Recuerda esto: en la SNN no será así.

Ejecuta:  python paso2_ann.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import datos


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    Convierte una lista de puntuaciones en probabilidades (positivas, suman 1).
    Restamos el máximo antes de exp() solo para evitar números gigantes;
    no cambia el resultado.
    """
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def a_one_hot(y, num_clases):
    """
    Convierte las etiquetas en formato "one-hot":
        clase 2 -> [0, 0, 1, 0]
    Así podemos comparar directamente con las 4 salidas de la red.
    """
    one_hot = np.zeros((len(y), num_clases))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


class RedNeuronalArtificial:
    """
    Una ANN de 2 capas (una oculta + una de salida).

    Una "clase" en Python es una plantilla para crear objetos que guardan
    datos (aquí: los pesos) y funciones que trabajan con esos datos
    (aquí: predecir y entrenar). 'self' es el propio objeto.
    """

    def __init__(self, num_entradas, num_ocultas, num_salidas, semilla=0):
        # __init__ se ejecuta al crear la red: RedNeuronalArtificial(25, 16, 4)
        gen = np.random.default_rng(semilla)

        # W1: matriz de pesos entrada -> oculta, forma (25, 16)
        #     W1[i, j] = peso de la conexión del píxel i a la neurona oculta j
        self.W1 = gen.normal(0, 0.5, size=(num_entradas, num_ocultas))
        self.b1 = np.zeros(num_ocultas)            # un sesgo por neurona oculta

        # W2: matriz de pesos oculta -> salida, forma (16, 4)
        self.W2 = gen.normal(0, 0.5, size=(num_ocultas, num_salidas))
        self.b2 = np.zeros(num_salidas)

    def hacia_delante(self, X):
        """
        Propagación hacia delante (forward pass).
        X tiene forma (N, 25): N ejemplos a la vez. Numpy calcula todos
        de golpe con multiplicaciones de matrices (@).
        """
        z1 = X @ self.W1 + self.b1     # (N,25) @ (25,16) -> (N,16)
        h = sigmoide(z1)               # activaciones de la capa oculta
        z2 = h @ self.W2 + self.b2     # (N,16) @ (16,4)  -> (N,4)
        p = softmax(z2)                # probabilidades de cada dígito
        return h, p                    # devolvemos h porque lo necesitamos al entrenar

    def predecir(self, X):
        _, p = self.hacia_delante(X)
        return np.argmax(p, axis=1)    # argmax: posición del valor más alto de cada fila

    def entrenar(self, X, y, epocas=200, tasa_aprendizaje=0.5, tam_lote=16,
                 semilla=0, mostrar=True):
        """
        Entrena la red con descenso por gradiente en mini-lotes.
        Devuelve la lista de pérdidas por época (para dibujarla).
        """
        gen = np.random.default_rng(semilla)
        Y = a_one_hot(y, self.W2.shape[1])
        historial_perdida = []

        for epoca in range(epocas):
            # Mezclamos el orden de los ejemplos en cada época
            orden = gen.permutation(len(X))

            for inicio in range(0, len(X), tam_lote):
                indices = orden[inicio:inicio + tam_lote]
                Xl, Yl = X[indices], Y[indices]
                n = len(Xl)

                # ---------- 1) HACIA DELANTE ----------
                h, p = self.hacia_delante(Xl)

                # ---------- 2) y 3) HACIA ATRÁS (backpropagation) ----------
                # Error en la capa de salida. Con softmax + entropía cruzada,
                # la derivada es simplemente (predicción - objetivo).
                delta2 = (p - Yl) / n                    # forma (n, 4)

                # Gradientes de W2 y b2
                grad_W2 = h.T @ delta2                   # (16,n) @ (n,4) -> (16,4)
                grad_b2 = delta2.sum(axis=0)

                # Llevamos el error hacia atrás, a la capa oculta.
                # Multiplicamos por la derivada de la sigmoide: h * (1 - h)
                delta1 = (delta2 @ self.W2.T) * h * (1 - h)   # (n,16)

                grad_W1 = Xl.T @ delta1                  # (25,n) @ (n,16) -> (25,16)
                grad_b1 = delta1.sum(axis=0)

                # ---------- 4) ACTUALIZAR PESOS ----------
                self.W2 -= tasa_aprendizaje * grad_W2
                self.b2 -= tasa_aprendizaje * grad_b2
                self.W1 -= tasa_aprendizaje * grad_W1
                self.b1 -= tasa_aprendizaje * grad_b1

            # Pérdida (entropía cruzada) sobre todos los datos, para vigilar el progreso.
            # Es pequeña cuando la red da probabilidad alta a la clase correcta.
            _, p = self.hacia_delante(X)
            perdida = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
            historial_perdida.append(perdida)

            if mostrar and (epoca % 20 == 0 or epoca == epocas - 1):
                precision = np.mean(self.predecir(X) == y)
                print(f"Época {epoca:3d}  pérdida = {perdida:.4f}  "
                      f"precisión = {precision * 100:.1f} %")

        return historial_perdida


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Datos de entrenamiento y de prueba (distinta semilla = ejemplos distintos)
    X_ent, y_ent = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=0)
    X_prueba, y_prueba = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=1)

    print(f"Entrenando con {len(X_ent)} ejemplos...\n")
    red = RedNeuronalArtificial(num_entradas=25, num_ocultas=16, num_salidas=4)
    perdidas = red.entrenar(X_ent, y_ent, epocas=100)

    precision = np.mean(red.predecir(X_prueba) == y_prueba)
    print(f"\nPrecisión con ejemplos NUEVOS (prueba): {precision * 100:.1f} %")

    # Veamos qué "piensa" la red de un ejemplo concreto
    ejemplo = X_prueba[0:1]                 # [0:1] para mantener forma (1, 25)
    _, probs = red.hacia_delante(ejemplo)
    print("\nEjemplo de prueba:")
    datos.mostrar_dibujo(ejemplo[0])
    for nombre, prob in zip(datos.NOMBRES_CLASES, probs[0]):
        barra = "█" * int(prob * 30)
        print(f"  '{nombre}': {prob:.3f} {barra}")
    print(f"  -> correcto: '{datos.NOMBRES_CLASES[y_prueba[0]]}'")

    # Gráfica de la pérdida
    plt.figure(figsize=(6, 4))
    plt.plot(perdidas)
    plt.xlabel("Época")
    plt.ylabel("Pérdida (entropía cruzada)")
    plt.title("ANN: la pérdida baja mientras aprende")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("figuras", exist_ok=True)   # crea la carpeta si no existe
    plt.savefig("figuras/ann_perdida.png", dpi=120)
    print("\nGráfica guardada en figuras/ann_perdida.png")
    plt.show()
