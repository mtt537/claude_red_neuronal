"""
PASO 5 - Comparación ANN vs SNN en la misma tarea
==================================================

Entrenamos las dos redes con los MISMOS datos y las comparamos en:

    1. Precisión con distintos niveles de ruido en las imágenes.
    2. Tiempo de cálculo en tu ordenador.
    3. Cantidad de "trabajo" que hace cada red para UNA imagen:
         - ANN: multiplicaciones (cada peso x su entrada, SIEMPRE todas)
         - SNN: sumas (solo cuando llega un pulso -> "basada en eventos")

Ejecuta:  python paso5_comparacion.py
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import datos
from paso2_ann import RedNeuronalArtificial
from paso4_snn import RedNeuronalPulsos, codificar_en_pulsos


def cronometrar(funcion, *argumentos):
    """Ejecuta funcion(*argumentos) y devuelve (resultado, segundos que tardó)."""
    inicio = time.perf_counter()
    resultado = funcion(*argumentos)
    return resultado, time.perf_counter() - inicio


if __name__ == "__main__":
    X_ent, y_ent = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=0)

    # ------------------------------------------------------------------
    # 1) Entrenar las dos redes
    # ------------------------------------------------------------------
    # 'lambda: ...' crea una mini-función sin nombre. La usamos para pasarle
    # a cronometrar() "lo que hay que ejecutar" sin ejecutarlo todavía.
    print("Entrenando la ANN...")
    ann = RedNeuronalArtificial(num_entradas=25, num_ocultas=16, num_salidas=4)
    _, t_ann = cronometrar(lambda: ann.entrenar(X_ent, y_ent, epocas=100, mostrar=False))

    print("Entrenando la SNN...")
    snn = RedNeuronalPulsos(num_entradas=25, num_salidas=4)
    _, t_snn = cronometrar(lambda: snn.entrenar(X_ent, y_ent, epocas=5, mostrar=False))

    print(f"\nTiempo de entrenamiento:  ANN = {t_ann:.2f} s   SNN = {t_snn:.2f} s")

    # ------------------------------------------------------------------
    # 2) Precisión con más y más ruido
    # ------------------------------------------------------------------
    niveles_ruido = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    prec_ann, prec_snn = [], []
    print("\nRuido   | ANN      | SNN")
    print("--------+----------+---------")
    for ruido in niveles_ruido:
        X_p, y_p = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=ruido, semilla=123)
        a = np.mean(ann.predecir(X_p) == y_p)
        s = np.mean(snn.predecir(X_p) == y_p)
        prec_ann.append(a)
        prec_snn.append(s)
        print(f"{ruido * 100:5.0f} % | {a * 100:6.1f} % | {s * 100:6.1f} %")

    # ------------------------------------------------------------------
    # 3) ¿Cuánto trabajo hace cada red por imagen?
    # ------------------------------------------------------------------
    X_p, _ = datos.crear_dataset(ejemplos_por_clase=50, prob_cambio=0.08, semilla=7)

    # ANN: cada peso se multiplica por su entrada, SIEMPRE, aunque la entrada sea 0
    mult_ann = ann.W1.size + ann.W2.size            # 25*16 + 16*4 = 464

    # SNN: cada pulso de entrada provoca 1 suma en cada neurona de salida.
    # Los píxeles apagados NO generan pulsos -> NO cuestan nada.
    gen = np.random.default_rng(0)
    pulsos_por_imagen = [codificar_en_pulsos(img, snn.duracion, snn.frecuencia_max,
                                             snn.dt, gen).sum() for img in X_p]
    sumas_snn = np.mean(pulsos_por_imagen) * snn.W.shape[1]

    _, t_pred_ann = cronometrar(ann.predecir, X_p)
    _, t_pred_snn = cronometrar(snn.predecir, X_p)

    print(f"""
Trabajo por imagen:
  ANN: {mult_ann} multiplicaciones + sumas (siempre las mismas), en 1 solo paso
  SNN: ~{sumas_snn:.0f} sumas (sin multiplicar), repartidas en {int(snn.duracion)} pasos de tiempo
       (~{np.mean(pulsos_por_imagen):.0f} pulsos de entrada por imagen)

Tiempo para clasificar {len(X_p)} imágenes en TU ordenador:
  ANN: {t_pred_ann * 1000:.1f} ms     SNN: {t_pred_snn * 1000:.1f} ms

¡Ojo! En un ordenador normal la SNN es MÁS LENTA, porque hay que simular
cada milisegundo con un bucle. Su ventaja de energía aparece en chips
"neuromórficos" (Intel Loihi, SpiNNaker, ...) donde cada neurona es
hardware real que solo gasta energía cuando le llega un pulso.
""")

    plt.figure(figsize=(6, 4))
    ruido_pct = [r * 100 for r in niveles_ruido]
    plt.plot(ruido_pct, [p * 100 for p in prec_ann], "o-", label="ANN (backpropagation)")
    plt.plot(ruido_pct, [p * 100 for p in prec_snn], "s-", label="SNN (STDP + LIF)")
    plt.axhline(25, color="gray", linestyle=":", label="azar (25 %)")
    plt.xlabel("Píxeles cambiados al azar (%)")
    plt.ylabel("Precisión (%)")
    plt.title("ANN vs SNN: reconocer dígitos con ruido")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("figuras", exist_ok=True)
    plt.savefig("figuras/comparacion.png", dpi=120)
    print("Gráfica guardada en figuras/comparacion.png")
    plt.show()
