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
└── tutorial/
    ├── Tutorial.md
    ├── setup.sh
    ├── comun.py
    ├── 01_descargar.py
    ├── 02_inspeccionar.py
    ├── 03_inferencia.py
    ├── 04_cuantizar.sh
    ├── 05_benchmark.py
    ├── 06_precision.py
    └── 07_resumen.py
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

Contiene el tutorial práctico guiado, que cada participante realiza por su cuenta en Linux. Mientras la demostración parte de un modelo entrenado por nosotros, el tutorial parte de un modelo descargado del LiteRT Community Hub en Hugging Face (MobileNet-v2, 1000 clases de ImageNet) y profundiza en las etapas de optimización y evaluación:

Paso	Archivo	Qué se hace
0	setup.sh	Prepara el entorno virtual con versiones fijas y registra el hardware del equipo.
1	01_descargar.py	Descarga del Hub las versiones FP32 e INT8 estática, las etiquetas y 100 imágenes de prueba.
2	02_inspeccionar.py	Inspecciona firmas, tensores (formato NCHW), cuantización y operadores de cada modelo.
3	03_inferencia.py	Primera inferencia con la API CompiledModel sobre una foto tomada por el participante.
4	04_cuantizar.sh	Cuantización dinámica propia con LiteRT Quantizer (litert-quantizer).
5	05_benchmark.py	Mide latencia (con 1 y varios hilos) y RAM de las tres variantes.
6	06_precision.py	Mide Top-1, Top-5 y concordancia con FP32 de cada variante.
7	07_resumen.py	Reúne todo en una tabla comparativa medida en el propio equipo.
Tutorial.md → guía completa: conceptos de cada paso, código embebido, salidas esperadas, preparación del entregable y solución de problemas.
comun.py → código compartido por los pasos en Python.

Cada paso guarda su salida de consola en resultados/paso_XX.txt. El entregable es un PDF con esas salidas, de los pasos 0 a 7; Tutorial.md explica cómo armarlo.

Los modelos, las imágenes y los resultados no están en el repositorio: se descargan y generan al seguir el tutorial. Solo los pasos 0 y 1 requieren Internet.

## Cómo seguir el mini-taller

1. Revisar la **presentación** (`presentacion/`) para conocer los conceptos.
2. Seguir la **demostración** (`demostracion/Demostracion.md`) para ver el flujo completo funcionando.
3. Realizar el **tutorial** (`tutorial/`) para practicarlo y generar el entregable.

## Requisitos generales

- Linux x86-64 y Python 3.10 o superior (para la demostración).
- Una cámara web (opcional: la demostración incluye un modo con imagen fija).
- Una cuenta de Google, para generar los modelos en Colab.
- Para el tutorial: conexión a Internet en los pasos 0 y 1, unos 400 MB libres en disco y un teléfono o cámara para tomar una foto.