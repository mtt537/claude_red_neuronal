"""
PASO 3 - Una sola neurona de pulsos (la pieza básica de una SNN)
=================================================================

Las neuronas de una SNN (Spiking Neural Network) se parecen más a las
neuronas del cerebro. El modelo más usado es la neurona LIF
("Leaky Integrate-and-Fire" = "Integra y Dispara con Fugas").

Imagínala como un CUBO DE AGUA CON UN AGUJERO:

    - El nivel del agua es el "potencial de membrana" V (en milivoltios).
    - La corriente de entrada I es el agua que entra al cubo  -> INTEGRA
    - El agujero hace que el agua se vaya escapando poco a poco
      y el nivel vuelve a su valor de reposo              -> FUGAS (Leaky)
    - Si el nivel llega a un UMBRAL, el cubo se vacía de golpe
      y la neurona emite un PULSO (spike)                 -> DISPARA (Fire)
    - Tras disparar, la neurona descansa un momento sin poder
      disparar otra vez (periodo refractario).

La ecuación (solo para curiosos):

    tau * dV/dt = -(V - V_reposo) + R*I
      ^^^^^^^^^^    ^^^^^^^^^^^^^^   ^^^
      velocidad     fuga hacia        entrada
                    el reposo

Como el ordenador no puede calcular en tiempo continuo, avanzamos en
pasitos de tiempo dt (1 milisegundo) - esto se llama "método de Euler":

    V_nuevo = V + (dt / tau) * ( -(V - V_reposo) + R*I )

DIFERENCIAS CLAVE con la neurona artificial del paso 1:
    * La neurona LIF tiene MEMORIA (V depende de lo que pasó antes).
    * Su salida NO es un número continuo: es 0 (nada) o 1 (pulso)
      en cada instante de tiempo.
    * La información está en CUÁNDO y CUÁNTAS VECES dispara.

Ejecuta:  python paso3_neurona_lif.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt


class NeuronaLIF:
    """Una neurona Leaky Integrate-and-Fire."""

    def __init__(self,
                 tau=20.0,          # ms: cuánto tarda en "vaciarse" el cubo (fuga)
                 v_reposo=-65.0,    # mV: nivel de reposo
                 v_umbral=-50.0,    # mV: si V llega aquí -> ¡pulso!
                 v_reinicio=-70.0,  # mV: valor de V justo después del pulso
                 refractario=2.0,   # ms: tiempo de descanso tras un pulso
                 dt=1.0):           # ms: tamaño de cada paso de tiempo
        self.tau = tau
        self.v_reposo = v_reposo
        self.v_umbral = v_umbral
        self.v_reinicio = v_reinicio
        self.refractario = refractario
        self.dt = dt

        self.v = v_reposo              # la neurona empieza en reposo
        self.tiempo_descanso = 0.0     # ms que le quedan de periodo refractario

    def paso(self, entrada):
        """
        Avanza la neurona UN paso de tiempo.
        'entrada' es R*I, en mV (cuánto "empuja" la corriente).
        Devuelve True si la neurona ha disparado en este paso.
        """
        # Si está en periodo refractario, no hace nada
        if self.tiempo_descanso > 0:
            self.tiempo_descanso -= self.dt
            return False

        # 1) Integrar con fugas (método de Euler)
        dv = (-(self.v - self.v_reposo) + entrada) * (self.dt / self.tau)
        self.v += dv

        # 2) ¿Ha llegado al umbral?
        if self.v >= self.v_umbral:
            self.v = self.v_reinicio                 # se "vacía el cubo"
            self.tiempo_descanso = self.refractario  # descansa
            return True                              # ¡PULSO!
        return False


def simular(entrada_constante, duracion=200.0, dt=1.0):
    """
    Simula una neurona LIF con una entrada constante.
    Devuelve: tiempos, potencial en cada instante, y tiempos de los pulsos.
    """
    neurona = NeuronaLIF(dt=dt)
    num_pasos = int(duracion / dt)
    tiempos = np.arange(num_pasos) * dt
    potenciales = np.zeros(num_pasos)
    pulsos = []

    for i in range(num_pasos):
        # La entrada solo está "encendida" entre t = 20 ms y t = 180 ms
        entrada = entrada_constante if 20 <= tiempos[i] < 180 else 0.0
        disparo = neurona.paso(entrada)
        potenciales[i] = neurona.v
        if disparo:
            pulsos.append(tiempos[i])

    return tiempos, potenciales, pulsos


if __name__ == "__main__":
    # Probamos tres intensidades de entrada.
    # Con v_reposo = -65 y v_umbral = -50, necesitamos más de 15 mV para
    # llegar al umbral: por debajo de eso la neurona NUNCA dispara.
    intensidades = [12.0, 20.0, 35.0]

    fig, ejes = plt.subplots(len(intensidades), 1, figsize=(8, 7), sharex=True)
    for eje, intensidad in zip(ejes, intensidades):
        t, v, pulsos = simular(intensidad)
        print(f"Entrada = {intensidad:4.1f} mV -> {len(pulsos):2d} pulsos en 160 ms"
              f"  (≈ {len(pulsos) / 0.16:5.1f} pulsos/segundo)")

        eje.plot(t, v, label="potencial V")
        eje.axhline(-50, color="red", linestyle="--", linewidth=1, label="umbral")
        # Dibujamos cada pulso como una rayita vertical arriba
        for tp in pulsos:
            eje.vlines(tp, -50, -40, color="black")
        eje.set_ylabel("V (mV)")
        eje.set_title(f"Entrada = {intensidad} mV  ->  {len(pulsos)} pulsos")
        eje.set_ylim(-75, -38)
        eje.grid(alpha=0.3)
    ejes[0].legend(loc="upper right")
    ejes[-1].set_xlabel("Tiempo (ms)")
    plt.tight_layout()
    os.makedirs("figuras", exist_ok=True)
    plt.savefig("figuras/lif_neurona.png", dpi=120)
    print("\nGráfica guardada en figuras/lif_neurona.png")

    print("""
Qué observar:
  * Con entrada débil (12 mV) V sube pero se queda por debajo del umbral:
    las fugas ganan y NO hay ningún pulso. (Una neurona artificial sigmoide
    siempre da alguna salida; esta puede quedarse totalmente callada.)
  * Cuanto más fuerte la entrada, MÁS pulsos por segundo. Así una SNN puede
    representar "cuánto" de algo hay: con la FRECUENCIA de pulsos
    ("codificación por tasa").
  * Cuando se apaga la entrada, V vuelve solo al reposo (la fuga).

Siguiente: conectar muchas neuronas LIF en una red que aprenda -> Paso 4.
""")
    plt.show()
