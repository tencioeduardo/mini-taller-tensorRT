# Mini-taller: LiteRT en sistemas embebidos

**Despliegue de aprendizaje automático en sistemas embebidos: el caso de LiteRT**

Mini-taller del curso **Taller de Sistemas Embebidos**, Escuela de Ingeniería Electrónica, Instituto Tecnológico de Costa Rica.

Presentado por **Eduardo Tencio Solano**.

## ¿De qué trata el mini-taller?

Un modelo de aprendizaje automático puede funcionar muy bien en un servidor y fallar al llevarlo a un dispositivo: no cabe en la memoria, tarda segundos por cada imagen o consume demasiada energía. Esta es la brecha entre la nube y el borde (*edge*).

**LiteRT** (*Lite Runtime*, antes TensorFlow Lite) es el motor de Google AI Edge que cierra esa brecha. Ejecuta modelos ya entrenados directamente en el dispositivo, en tres pasos:

```text
1. Convertir          2. Comprimir            3. Ejecutar
PyTorch, TensorFlow,  Cuantización:           En CPU, GPU o NPU,
JAX o Keras           modelos más pequeños    según el hardware
al formato .tflite    y livianos              disponible
```

El mini-taller recorre ese flujo de lo conceptual a lo práctico:

- Por qué procesar en el borde y qué es LiteRT.
- Cómo se ejecuta un modelo: la API `CompiledModel` (moderna) y la API `Interpreter` (clásica).
- Los aceleradores disponibles: CPU, GPU y NPU.
- Compresión por cuantización (FP32, FP16 e INT8) y sus compromisos de tamaño, velocidad y precisión.
- Cómo conseguir un modelo: convertir uno propio o descargar uno ya convertido.
- Qué medir antes de decidir: tamaño, memoria, latencia y precisión.

## Contenido del repositorio

```text
mini-taller-liteRT/
├── README.md
├── presentacion/
│   └── LiteRT_Mini_Taller.pdf
├── demostracion/
│   ├── Demostracion.md
│   ├── entrenar_rps.ipynb
│   ├── requirements.txt
│   ├── setup.sh
│   ├── verificar_entorno.py
│   ├── rps_common.py
│   ├── evaluar.py
│   ├── litert_rps.py
│   └── docs/
└── tutorial/                (en desarrollo)
```

### `presentacion/`

Contiene la presentación teórica del mini-taller en PDF (`LiteRT_Mini_Taller.pdf`, 20 diapositivas), organizada en cinco bloques:

1. **El desafío del borde:** por qué un buen modelo falla fuera del servidor y qué es LiteRT.
2. **¿Qué hace y cómo trabaja?:** el camino de un modelo, las APIs y los aceleradores.
3. **Compresión y modelos:** cuantización (PTQ y QAT) y dónde conseguir modelos listos.
4. **Evaluación:** datos reales de la literatura y el compromiso entre portabilidad y rendimiento.
5. **Conclusiones y demo:** resumen y transición a la práctica en Linux.

La última diapositiva reúne las fuentes consultadas.

### `demostracion/`

Contiene la demostración técnica que sigue a la presentación. Un mismo clasificador de **Piedra, Papel o Tijera** se convierte a cuatro formatos (FP32, FP16, INT8 dinámico e INT8 completo), se mide en una laptop Linux con las dos APIs de LiteRT y finalmente se ejecuta en vivo con la cámara web.

- `Demostracion.md` → guía completa: preparación, pasos de la ejecución en clase, funcionamiento y solución de problemas.
- `entrenar_rps.ipynb` → cuaderno de Google Colab que entrena el modelo, lo convierte a `.tflite` y exporta el set de prueba.
- `setup.sh` y `requirements.txt` → preparan el entorno de Python y validan los archivos.
- `verificar_entorno.py` → registra las versiones y las capacidades del equipo.
- `rps_common.py` → código compartido por los demás scripts.
- `evaluar.py` → mide tamaño, memoria, latencia y precisión de cada modelo.
- `litert_rps.py` → clasificación en vivo con la cámara y OpenCV.
- `docs/` → documentación complementaria de la demostración.

> Los modelos (`modelos/`) y el set de prueba (`datos/`) **no están en el repositorio**: se generan ejecutando `entrenar_rps.ipynb` en Colab. Los pasos están en `Demostracion.md`.

### `tutorial/`

Reservada para el tutorial práctico guiado, en el que los participantes ejecutan el flujo de LiteRT paso a paso. Actualmente en desarrollo.

## Cómo seguir el mini-taller

1. Revisar la **presentación** (`presentacion/`) para conocer los conceptos.
2. Seguir la **demostración** (`demostracion/Demostracion.md`) para ver el flujo completo funcionando.
3. Realizar el **tutorial** (`tutorial/`) para practicarlo (cuando esté disponible).

## Requisitos generales

- Linux x86-64 y Python 3.10 o superior (para la demostración).
- Una cámara web (opcional: la demostración incluye un modo con imagen fija).
- Una cuenta de Google, para generar los modelos en Colab.