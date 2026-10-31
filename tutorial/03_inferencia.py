#!/usr/bin/env python3
"""
PASO 3 - Primera inferencia con la API CompiledModel de LiteRT.

Se clasifica una foto tomada por usted con el modelo FP32, escribiendo a mano
cada etapa del ciclo de CompiledModel:

  from_file -> create_input_buffers / create_output_buffers
            -> write -> run_by_index -> read -> softmax -> Top-5

Uso:
  python 03_inferencia.py --imagen mi_foto.jpg
"""
import argparse
import os
import re
import time

import numpy as np

from comun import (MODELO_FP32, Registro, cargar_etiquetas, capturar_logs_nativos,
                   exigir_modelo, fallar, ok, preprocesar, seccion, softmax)


def resumir_logs(texto):
    """Conserva solo las líneas de los registros nativos que explican el hardware."""
    claves = ("RegisterAccelerator", "registered", "could not be loaded",
              "XNNPACK", "Flatbuffer model")
    vistas = set()
    for linea in texto.splitlines():
        if any(c in linea for c in claves):
            linea = linea.split("] ", 1)[-1] if "] " in linea else linea
            linea = re.sub(r"^INFO: |ptr=0x[0-9a-fA-F]+, ", "", linea)
            linea = linea.split("(/")[0].strip()      # recorta rutas largas
            if linea not in vistas:
                vistas.add(linea)
                print(f"    {linea}")


def main(ruta_imagen):
    exigir_modelo(MODELO_FP32, 1)
    if not os.path.isfile(ruta_imagen):
        fallar(f"No existe la imagen '{ruta_imagen}'. Tome una foto (ver tutorial) "
               "y pase su ruta con --imagen.")
    etiquetas = cargar_etiquetas()

    from ai_edge_litert.compiled_model import CompiledModel

    # ---------------------------------------------------------------- 3.1
    seccion("3.1 Crear el CompiledModel (compilación para el hardware)")
    logs = []
    t0 = time.perf_counter()
    with capturar_logs_nativos(logs):
        modelo = CompiledModel.from_file(MODELO_FP32)   # por defecto: CPU
    ms_compilar = (time.perf_counter() - t0) * 1000
    print(f"Modelo: {MODELO_FP32}")
    print(f"Tiempo de carga y compilación: {ms_compilar:.1f} ms (se hace UNA sola vez)")
    print("Mensajes del runtime (aceleradores que LiteRT intentó registrar):")
    resumir_logs(logs[0])
    print(f"¿Todo el grafo quedó en el acelerador?: {modelo.is_fully_accelerated()}")

    # ---------------------------------------------------------------- 3.2
    seccion("3.2 Firmas y búferes de entrada/salida")
    firmas = modelo.get_signature_list()
    nombre_firma = list(firmas)[0]
    print(f"Firmas disponibles: {firmas}")
    for nombre, det in modelo.get_input_tensor_details(nombre_firma).items():
        print(f"Entrada '{nombre}': tensor {det['name']}, tipo {det['dtype']}, forma {det['shape']}")
    entradas = modelo.create_input_buffers(0)     # 0 = primera firma
    salidas = modelo.create_output_buffers(0)
    req_in = modelo.get_input_buffer_requirements(0, 0)
    req_out = modelo.get_output_buffer_requirements(0, 0)
    n_salidas = req_out["buffer_size"] // np.dtype(np.float32).itemsize
    print(f"Búfer de entrada : {req_in['buffer_size']:,} bytes "
          f"(= 1x3x224x224 valores float32 de 4 bytes)")
    print(f"Búfer de salida  : {req_out['buffer_size']:,} bytes (= {n_salidas} valores float32)")

    # ---------------------------------------------------------------- 3.3
    seccion("3.3 Preprocesamiento de la imagen")
    from PIL import Image
    with Image.open(ruta_imagen) as img:
        print(f"Imagen original: {ruta_imagen}  ({img.size[0]}x{img.size[1]} px, {img.mode})")
    x = preprocesar(ruta_imagen)
    print(f"Tensor de entrada: forma={x.shape}  tipo={x.dtype}")
    print(f"Valores normalizados: mín={x.min():.2f}  máx={x.max():.2f}  media={x.mean():.3f}")
    print("Formato NCHW: (lote=1, canales=3, alto=224, ancho=224), como en PyTorch.")

    # ---------------------------------------------------------------- 3.4
    seccion("3.4 Inferencia: write -> run_by_index -> read")
    tiempos = []
    for _ in range(3):
        t0 = time.perf_counter()
        entradas[0].write(x)
        modelo.run_by_index(0, entradas, salidas)
        logits = salidas[0].read(n_salidas, np.float32)
        tiempos.append((time.perf_counter() - t0) * 1000)
    print(f"Tiempo de inferencia: 1.ª = {tiempos[0]:.1f} ms, 2.ª = {tiempos[1]:.1f} ms, "
          f"3.ª = {tiempos[2]:.1f} ms")
    print(f"Salida cruda (logits): {n_salidas} valores, rango [{logits.min():.2f}, "
          f"{logits.max():.2f}], suma {logits.sum():.2f}")
    print("Como la suma no es 1, la salida son logits: se aplica softmax.")

    # ---------------------------------------------------------------- 3.5
    seccion("3.5 Resultado: Top-5")
    probs = softmax(logits)
    for pos, clase in enumerate(np.argsort(probs)[::-1][:5], start=1):
        barra = "#" * int(round(probs[clase] * 30))
        print(f"  {pos}. [{clase:3d}] {etiquetas[clase][:38]:38s} {probs[clase] * 100:6.2f} % {barra}")
    print()
    ok("Paso 3 completado: inferencia con CompiledModel.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagen", required=True, help="ruta de su foto (JPG o PNG)")
    args = p.parse_args()
    with Registro(3, "Inferencia con CompiledModel sobre una foto propia"):
        main(args.imagen)
