#!/usr/bin/env python3
"""
PASO 2 - Inspeccionar un modelo .tflite antes de usarlo.

Para cada modelo muestra:
  * Tamaño y cabecera del FlatBuffer.
  * Firmas (signatures): nombres de entrada y salida con los que se invoca.
  * Tensores de entrada y salida: forma, tipo de dato y parámetros de cuantización.
  * Operadores del grafo y tipos de dato de todos sus tensores.
  * En cuántos segmentos queda repartido el grafo al delegarlo a XNNPACK (CPU).

Uso:
  python 02_inspeccionar.py                      # FP32 e INT8 del Hub
  python 02_inspeccionar.py modelos/otro.tflite  # cualquier modelo
"""
import argparse
import os
from collections import Counter

import numpy as np

from comun import (MODELO_FP32, MODELO_INT8, Registro, capturar_logs_nativos,
                   fallar, ok, seccion, tamano_mb, validar_tflite)


def describir_tensor(det):
    escala, cero = det["quantization"]
    cuant = (f"escala={escala:.6f}, punto_cero={cero}" if escala
             else "sin cuantizar")
    forma = "x".join(str(d) for d in det["shape"])
    return f"{det['name']}\n        forma=({forma})  tipo={np.dtype(det['dtype']).name}  [{cuant}]"


def inspeccionar(ruta):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.interpreter import Interpreter

    seccion(f"Modelo: {os.path.basename(ruta)}")
    valido, detalle = validar_tflite(ruta)
    if not valido:
        fallar(f"{ruta}: {detalle}")
    print(f"Tamaño en disco : {tamano_mb(ruta):.2f} MB ({os.path.getsize(ruta):,} bytes)")
    print(f"FlatBuffer      : {detalle}")

    logs = []
    with capturar_logs_nativos(logs):
        # 1) Grafo tal como está en el archivo -> tensores y su cuantización
        sin_delegar = Interpreter(model_path=ruta)
        # 2) Modelo preparado para CPU: allocate_tensors reparte el grafo en XNNPACK
        delegado = Interpreter(model_path=ruta)
        delegado.allocate_tensors()
        # 3) API moderna: CompiledModel
        compilado = CompiledModel.from_file(ruta)

    firmas = compilado.get_signature_list()
    print(f"Firmas          : {firmas}")

    entrada = sin_delegar.get_input_details()[0]
    salida = sin_delegar.get_output_details()[0]
    print(f"Entrada         : {describir_tensor(entrada)}")
    print(f"Salida          : {describir_tensor(salida)}")
    req = compilado.get_input_buffer_requirements(0, 0)
    print(f"Búfer de entrada: {req['buffer_size']:,} bytes (reservado por CompiledModel)")

    ops = [o for o in delegado._get_ops_details() if o["op_name"] != "DELEGATE"]
    segmentos = sum(1 for o in delegado._get_ops_details() if o["op_name"] == "DELEGATE")
    conteo_ops = Counter(o["op_name"] for o in ops)
    print(f"Operadores      : {len(ops)} en total")
    for nombre, n in conteo_ops.most_common():
        print(f"        {nombre:20s} {n:3d}")

    tipos = Counter(np.dtype(t["dtype"]).name for t in sin_delegar.get_tensor_details())
    print("Tensores por tipo de dato: " + ", ".join(f"{k}={v}" for k, v in tipos.most_common()))
    print(f"Segmentos XNNPACK: {segmentos}  (1 = todo el grafo corre en un solo bloque acelerado)")
    print(f"CompiledModel totalmente acelerado: {compilado.is_fully_accelerated()}")
    return {"entrada": entrada, "salida": salida}


def main(rutas):
    resultados = {r: inspeccionar(r) for r in rutas}
    if len(rutas) > 1:
        seccion("Comparación de entradas")
        for ruta, info in resultados.items():
            e = info["entrada"]
            print(f"  {os.path.basename(ruta):40s} entrada {np.dtype(e['dtype']).name:7s} "
                  f"forma {[int(d) for d in e['shape']]}")
    print()
    ok("Inspección completada.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspecciona modelos .tflite")
    p.add_argument("modelos", nargs="*", default=[MODELO_FP32, MODELO_INT8])
    p.add_argument("--sin-registro", action="store_true",
                   help="no guardar en resultados/paso_02.txt (lo usa el paso 4)")
    args = p.parse_args()
    with Registro(2, "Inspección de los modelos", activo=not args.sin_registro):
        main(args.modelos)
