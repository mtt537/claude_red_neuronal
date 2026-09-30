# Redes neuronales desde cero: ANN vs SNN (en Python)

Proyecto para **aprender** cómo funcionan dos tipos de redes neuronales y en
qué se diferencian, programándolas desde cero (solo con `numpy` para los
cálculos y `matplotlib` para las gráficas, sin librerías de IA).

- **ANN** (*Artificial Neural Network*): la red neuronal "clásica" que usa casi
  todo el deep learning actual.
- **SNN** (*Spiking Neural Network*): red de pulsos, más parecida al cerebro.
  Las neuronas se comunican con pulsos eléctricos a lo largo del tiempo.

Las dos redes aprenden **la misma tarea**: reconocer los dígitos 0, 1, 2 y 3
dibujados en una cuadrícula de 5x5 píxeles, aunque tengan algo de "ruido"
(píxeles cambiados al azar).

## Instalación

```bash
# (opcional pero recomendado) crear un entorno virtual
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Orden de lectura (¡lee el código, está muy comentado!)

Abre cada archivo, lee los comentarios de arriba a abajo y después ejecútalo.

| Paso | Archivo | Qué aprendes |
|------|---------|--------------|
| 0 | `datos.py` | Cómo se representan las imágenes como listas de números |
| 1 | `paso1_neurona_artificial.py` | Una sola neurona artificial aprende la puerta AND |
| 2 | `paso2_ann.py` | Una ANN completa con *backpropagation* reconoce dígitos |
| 3 | `paso3_neurona_lif.py` | Una sola neurona de pulsos (LIF) y cómo dispara |
| 4 | `paso4_snn.py` | Una SNN completa que aprende con STDP reconoce dígitos |
| 5 | `paso5_comparacion.py` | Las dos redes cara a cara |

```bash
python datos.py
python paso1_neurona_artificial.py
python paso2_ann.py
python paso3_neurona_lif.py
python paso4_snn.py
python paso5_comparacion.py
```

Ejecútalos **desde esta carpeta**. Las gráficas se muestran en pantalla y se
guardan en `figuras/`.

## La idea en una imagen

```
ANN:  entrada (números)  ──►  suma ponderada ──► activación ──►  salida (número)
      [0, 1, 1, 0, ...]       todo en UN solo paso                [0.02, 0.95, ...]

SNN:  entrada (pulsos)   ──►  el potencial V sube, se va "fugando" ──► ¿V ≥ umbral? ──► PULSO
      | |  |   || |           se simula milisegundo a milisegundo         (y V se reinicia)
```

## Diferencias principales

| | ANN | SNN |
|---|---|---|
| **Qué se transmite** | Números continuos (0.73, -1.2...) | Pulsos: 0 ó 1 en cada instante |
| **Tiempo** | No existe: se calcula todo de una vez | Fundamental: se simula paso a paso |
| **Memoria de la neurona** | Ninguna (misma entrada → misma salida) | Sí: el potencial V acumula el pasado |
| **Dónde está la información** | En el valor de cada salida | En **cuántos** pulsos y **cuándo** |
| **Cómo aprende (aquí)** | Backpropagation (necesita derivadas) | STDP: regla local "dispararon juntas → se refuerzan" |
| **¿Se puede derivar?** | Sí, funciones suaves | No directamente: un pulso es un salto brusco |
| **Resultados** | Deterministas (siempre igual) | Algo aleatorios (los pulsos de entrada son al azar) |
| **Operaciones** | Multiplicaciones, siempre todas | Solo sumas, y solo cuando llega un pulso |
| **En un PC normal** | Muy rápida | Lenta (hay que simular cada milisegundo) |
| **En hardware neuromórfico** | — | Muy eficiente en energía (Loihi, SpiNNaker...) |
| **Parecido al cerebro** | Inspirada, pero lejano | Mucho más cercano |

## Resultados de ejemplo (`paso5_comparacion.py`)

```
Ruido   | ANN      | SNN
--------+----------+---------
    0 % |  100.0 % |   99.0 %
   10 % |   96.0 % |   93.0 %
   20 % |   87.5 % |   79.0 %
```

La ANN gana un poco en precisión: *backpropagation* es un método de
aprendizaje muy potente. La SNN, con una regla mucho más simple y local
(cada conexión solo "ve" su entrada y su salida), llega bastante cerca.
Los números de la SNN cambian un poco en cada ejecución porque los pulsos de
entrada se generan al azar.

## Glosario rápido

- **Peso**: número que dice lo fuerte que es una conexión entre dos neuronas.
- **Sesgo (bias)**: número que se suma para desplazar la activación.
- **Función de activación**: transforma la suma de una neurona (ej. sigmoide).
- **Época**: una pasada completa por todos los datos de entrenamiento.
- **Tasa de aprendizaje**: cuánto se cambian los pesos en cada corrección.
- **Backpropagation**: calcular hacia atrás cuánto contribuye cada peso al error.
- **Potencial de membrana (V)**: "carga" de una neurona de pulsos.
- **Umbral**: valor de V a partir del cual la neurona dispara.
- **Periodo refractario**: tiempo de descanso tras un pulso.
- **Codificación por tasa**: representar un valor con la frecuencia de pulsos.
- **Inhibición lateral**: cuando una neurona dispara, "calla" a sus vecinas.
- **STDP**: si la entrada dispara justo antes que la salida, la conexión se refuerza.

## Ideas para seguir practicando

1. En `paso2_ann.py`, cambia `num_ocultas` (16) por 4 o por 64. ¿Qué pasa?
2. En `paso2_ann.py`, prueba `tasa_aprendizaje=5.0` o `0.01`.
3. En `paso3_neurona_lif.py`, cambia `tau` a 5 o a 50 ms. ¿Dispara más o menos?
4. En `paso4_snn.py`, pon `self.inhibicion = 0.0`. ¿Empeora la SNN?
5. En `paso4_snn.py`, cambia `self.duracion` a 20 ms o a 300 ms. Menos tiempo =
   respuesta más rápida pero con menos pulsos para decidir.
6. Añade un quinto dígito en `datos.py` (¡y cambia `num_salidas` a 5!).
