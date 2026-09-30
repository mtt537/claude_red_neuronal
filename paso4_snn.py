"""
PASO 4 - Red Neuronal de Pulsos (SNN) desde cero
=================================================

Ahora conectamos neuronas LIF (paso 3) para resolver LA MISMA tarea que
la ANN del paso 2: reconocer los dígitos 0, 1, 2 y 3 dibujados en 5x5.

    25 neuronas de entrada               4 neuronas LIF de salida
    (una por píxel)                      (una por dígito)

    píxel 1  |  |   | |    --(pesos W)-->    "0"  ─┐
    píxel 2    |  |    |                     "1"   ├─ se inhiben entre sí
    ...                                      "2"   │  (el que dispara "calla"
    píxel 25  | |   |   |                    "3"  ─┘   a los demás)

Hay TRES ideas nuevas respecto a la ANN:

1) CODIFICAR la imagen en pulsos (codificación por tasa)
   Una SNN no entiende números como 0 o 1: solo entiende pulsos en el
   tiempo. Así que mostramos la imagen durante 100 ms y:
     - cada píxel ENCENDIDO emite pulsos al azar (unos 100 por segundo)
     - cada píxel APAGADO se queda en silencio
   (Es parecido a cómo la retina manda pulsos al cerebro.)

2) SIMULAR EN EL TIEMPO
   Avanzamos milisegundo a milisegundo. En cada paso, cada pulso de
   entrada "empuja" el potencial V de cada neurona de salida una cantidad
   igual a su peso (en mV). Pesos positivos excitan, negativos inhiben.
   Leemos la respuesta CONTANDO qué neurona de salida ha disparado más.

3) APRENDER CON STDP (y no con backpropagation)
   Un pulso es un salto brusco (0 -> 1): no se puede derivar, así que la
   backpropagation normal no funciona directamente. Usamos una regla
   inspirada en el cerebro, STDP ("Spike-Timing-Dependent Plasticity"):

       "Si la entrada disparó JUSTO ANTES que la salida,
        esa conexión probablemente ayudó -> se refuerza."

   Para saber qué entradas dispararon "justo antes", cada entrada lleva
   una TRAZA: un número que sube +1 con cada pulso y se va apagando con
   el tiempo (como la huella de un pulso reciente).

   Como queremos que aprenda una tarea concreta, añadimos un "profesor"
   (versión supervisada de STDP):
     - Durante el entrenamiento, la neurona CORRECTA recibe un empujón
       extra para que dispare -> sus conexiones activas se REFUERZAN.
     - Si una neurona INCORRECTA dispara -> sus conexiones activas
       se DEBILITAN (castigo).

   Toda la información que usa la regla es LOCAL: cada conexión solo
   mira su propia entrada y su propia salida. En backpropagation, en
   cambio, el error tiene que viajar hacia atrás por toda la red.

Ejecuta:  python paso4_snn.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import datos


def codificar_en_pulsos(imagen, duracion, frecuencia_max, dt, generador):
    """
    Convierte una imagen (25 valores entre 0 y 1) en trenes de pulsos.

    Devuelve una matriz de forma (num_pasos, 25) llena de True/False:
        pulsos[t, i] == True  ->  el píxel i emite un pulso en el paso t

    frecuencia_max: pulsos por segundo de un píxel totalmente encendido.
    """
    num_pasos = int(duracion / dt)
    # Probabilidad de disparar en un paso de dt ms.
    # Ej: 100 pulsos/s * 1 ms = 100 * 0.001 = 0.1 -> 10 % de probabilidad cada ms
    prob = imagen * frecuencia_max * (dt / 1000.0)
    # Para cada instante y cada píxel "tiramos un dado" (número entre 0 y 1)
    return generador.random((num_pasos, len(imagen))) < prob


class RedNeuronalPulsos:
    """Una SNN de una capa: 25 entradas -> 4 neuronas LIF de salida."""

    def __init__(self, num_entradas, num_salidas, semilla=0):
        self.gen = np.random.default_rng(semilla)

        # --- Parámetros de las neuronas LIF (los mismos del paso 3) ---
        self.dt = 1.0            # ms
        self.tau = 20.0          # ms, constante de fuga
        self.v_reposo = -65.0    # mV
        self.v_umbral = -50.0    # mV
        self.v_reinicio = -70.0  # mV
        self.refractario = 2.0   # ms

        # --- Parámetros de la simulación ---
        self.duracion = 100.0        # ms que mostramos cada imagen
        self.frecuencia_max = 100.0  # pulsos/s de un píxel encendido
        self.inhibicion = 10.0       # mV que baja a las demás cuando una dispara

        # --- Parámetros del aprendizaje (STDP) ---
        self.tau_traza = 20.0    # ms, lo que tarda en apagarse la "huella" de un pulso
        self.a_refuerzo = 0.005  # cuánto se refuerza una conexión
        self.a_castigo = 0.005   # cuánto se debilita una conexión
        self.empujon_profesor = 20.0  # mV extra para la neurona correcta al entrenar
        self.peso_max = 4.0      # los pesos se mantienen entre -peso_max y +peso_max

        # --- Pesos: W[i, j] = cuánto sube V de la salida j con un pulso de la entrada i ---
        # Empiezan pequeños y al azar: la red no sabe nada.
        self.W = self.gen.uniform(0.0, 1.0, size=(num_entradas, num_salidas))

    def simular(self, imagen, clase_correcta=None, aprender=False, guardar=False):
        """
        Presenta UNA imagen a la red durante 'self.duracion' ms.

        - Si aprender=True, usa el profesor y aplica STDP (necesita clase_correcta).
        - Devuelve cuántos pulsos ha emitido cada neurona de salida.
        - Si guardar=True, devuelve también todo lo que ha pasado
          (para poder dibujarlo).
        """
        num_salidas = self.W.shape[1]
        pulsos_entrada = codificar_en_pulsos(imagen, self.duracion,
                                             self.frecuencia_max, self.dt, self.gen)
        num_pasos = len(pulsos_entrada)

        # Estado de las neuronas de salida (un valor por neurona -> vectores)
        v = np.full(num_salidas, self.v_reposo)   # potencial de membrana
        descanso = np.zeros(num_salidas)          # ms de periodo refractario restantes
        traza = np.zeros(self.W.shape[0])         # huella de pulsos recientes de cada entrada
        conteo = np.zeros(num_salidas, dtype=int)

        # Corriente del profesor: solo para la neurona correcta y solo al aprender
        corriente_profesor = np.zeros(num_salidas)
        if aprender:
            corriente_profesor[clase_correcta] = self.empujon_profesor

        if guardar:
            historial_v = np.zeros((num_pasos, num_salidas))
            historial_pulsos = np.zeros((num_pasos, num_salidas), dtype=bool)

        # ============ BUCLE DEL TIEMPO: un paso = 1 ms ============
        for t in range(num_pasos):
            entrada_t = pulsos_entrada[t]       # qué píxeles disparan ahora (True/False x 25)

            # 1) Actualizar las trazas: se apagan un poco y suben +1 donde hay pulso
            traza *= np.exp(-self.dt / self.tau_traza)
            traza[entrada_t] += 1.0

            # 2) Fuga (igual que en el paso 3) + corriente del profesor
            v += (-(v - self.v_reposo) + corriente_profesor) * (self.dt / self.tau)

            # 3) Cada pulso de entrada suma su peso a V.
            #    W[entrada_t] son las filas de los píxeles que han disparado;
            #    sumándolas obtenemos el empujón total para cada salida.
            v += self.W[entrada_t].sum(axis=0)

            # 4) Las neuronas en periodo refractario no cambian
            en_descanso = descanso > 0
            v[en_descanso] = self.v_reinicio
            descanso[en_descanso] -= self.dt

            # 5) ¿Quién llega al umbral?
            disparan = v >= self.v_umbral       # vector True/False x 4
            if disparan.any():
                conteo[disparan] += 1
                v[disparan] = self.v_reinicio
                descanso[disparan] = self.refractario

                # Inhibición lateral: cada pulso baja el potencial de las DEMÁS
                num_disparos = disparan.sum()
                v[~disparan] -= self.inhibicion * num_disparos   # ~ significa "NO"

                # 6) APRENDIZAJE STDP (solo si estamos entrenando)
                if aprender:
                    for j in np.where(disparan)[0]:
                        if j == clase_correcta:
                            # Entrada activa justo antes de la salida correcta -> reforzar
                            self.W[:, j] += self.a_refuerzo * traza
                        else:
                            # La neurona equivocada ha disparado -> debilitar
                            self.W[:, j] -= self.a_castigo * traza
                    # Los pesos no pueden crecer sin límite
                    np.clip(self.W, -self.peso_max, self.peso_max, out=self.W)

            if guardar:
                historial_v[t] = v
                historial_pulsos[t] = disparan

        if guardar:
            return conteo, pulsos_entrada, historial_v, historial_pulsos
        return conteo

    def predecir(self, X):
        """Para cada imagen: la neurona que más pulsos ha emitido es la respuesta."""
        predicciones = []
        for imagen in X:
            conteo = self.simular(imagen)
            predicciones.append(np.argmax(conteo))
        return np.array(predicciones)

    def entrenar(self, X, y, epocas=5, mostrar=True):
        historial_precision = []
        for epoca in range(epocas):
            orden = self.gen.permutation(len(X))
            for i in orden:
                self.simular(X[i], clase_correcta=y[i], aprender=True)

            precision = np.mean(self.predecir(X) == y)
            historial_precision.append(precision)
            if mostrar:
                print(f"Época {epoca}  precisión (entrenamiento) = {precision * 100:.1f} %")
        return historial_precision


def dibujar_actividad(red, imagen, clase, archivo):
    """Dibuja los pulsos de entrada, el potencial y los pulsos de salida."""
    conteo, p_entrada, hist_v, hist_p = red.simular(imagen, guardar=True)
    t = np.arange(len(p_entrada)) * red.dt

    fig, ejes = plt.subplots(3, 1, figsize=(8, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})

    # (a) Raster de entrada: un punto por cada pulso de cada píxel
    tiempos, pixeles = np.where(p_entrada)
    ejes[0].scatter(t[tiempos], pixeles, s=4, color="black")
    ejes[0].set_ylabel("píxel de entrada")
    ejes[0].set_title(f"Entrada: dígito '{datos.NOMBRES_CLASES[clase]}' "
                      f"convertido en pulsos")

    # (b) Potencial de membrana de las 4 neuronas de salida
    for j in range(hist_v.shape[1]):
        ejes[1].plot(t, hist_v[:, j], label=f"neurona '{datos.NOMBRES_CLASES[j]}'")
    ejes[1].axhline(red.v_umbral, color="red", linestyle="--", linewidth=1)
    ejes[1].set_ylabel("V (mV)")
    ejes[1].legend(loc="upper right", fontsize=8)
    ejes[1].set_title("Potencial de las neuronas de salida")

    # (c) Raster de salida
    tiempos, neuronas = np.where(hist_p)
    ejes[2].scatter(t[tiempos], neuronas, s=30, marker="|", color="C3")
    ejes[2].set_yticks(range(len(datos.NOMBRES_CLASES)))
    ejes[2].set_yticklabels(datos.NOMBRES_CLASES)
    ejes[2].set_ylabel("salida")
    ejes[2].set_xlabel("Tiempo (ms)")
    ejes[2].set_title(f"Pulsos de salida por neurona: {conteo.tolist()}  ->  "
                      f"respuesta '{datos.NOMBRES_CLASES[np.argmax(conteo)]}'")

    plt.tight_layout()
    plt.savefig(archivo, dpi=120)
    print(f"Gráfica guardada en {archivo}")


def dibujar_pesos(red, archivo):
    """Cada neurona de salida tiene 25 pesos: los dibujamos como una imagen 5x5."""
    fig, ejes = plt.subplots(1, red.W.shape[1], figsize=(10, 3))
    limite = np.abs(red.W).max()
    for j, eje in enumerate(ejes):
        im = eje.imshow(red.W[:, j].reshape(5, 5), cmap="bwr",
                        vmin=-limite, vmax=limite)
        eje.set_title(f"pesos hacia '{datos.NOMBRES_CLASES[j]}'")
        eje.axis("off")
    fig.colorbar(im, ax=ejes, shrink=0.8, label="peso (mV)")
    plt.savefig(archivo, dpi=120)
    print(f"Gráfica guardada en {archivo}")


if __name__ == "__main__":
    X_ent, y_ent = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=0)
    X_prueba, y_prueba = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=1)

    red = RedNeuronalPulsos(num_entradas=25, num_salidas=4)

    print("Antes de entrenar (la red responde casi al azar):")
    print(f"  precisión = {np.mean(red.predecir(X_prueba) == y_prueba) * 100:.1f} %\n")

    print(f"Entrenando con {len(X_ent)} ejemplos (esto tarda un poco: "
          f"hay que simular cada milisegundo)...\n")
    red.entrenar(X_ent, y_ent, epocas=5)

    precision = np.mean(red.predecir(X_prueba) == y_prueba)
    print(f"\nPrecisión con ejemplos NUEVOS (prueba): {precision * 100:.1f} %\n")

    os.makedirs("figuras", exist_ok=True)
    dibujar_actividad(red, X_prueba[0], y_prueba[0], "figuras/snn_actividad.png")
    dibujar_pesos(red, "figuras/snn_pesos.png")

    print("""
Qué observar en las gráficas:
  * snn_actividad.png: arriba, la imagen convertida en pulsos al azar.
    En medio, V de cada neurona de salida sube con los pulsos y cae por
    la fuga. Abajo, la neurona correcta dispara mucho más que las demás.
  * snn_pesos.png: los pesos de cada neurona de salida "dibujan" su
    dígito (rojo = excita, azul = inhibe). ¡La red ha aprendido
    plantillas solo con la regla local STDP!
""")
    plt.show()
