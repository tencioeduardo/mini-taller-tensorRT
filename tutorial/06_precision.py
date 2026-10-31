#!/usr/bin/env python3
"""
PASO 6 - Medir cuánta precisión se pierde al cuantizar.

Clasifica el lote de 100 imágenes etiquetadas con las tres variantes y reporta:
  * Precisión Top-1 y Top-5 frente a la clase verdadera.
  * Concordancia con FP32: en cuántas imágenes la variante predice lo mismo
    que el modelo original (no necesita etiquetas; es la métrica clave al
    validar una cuantización).
  * Las imágenes en las que cada variante cambió su predicción.

Uso:
  python 06_precision.py
"""
import csv
import json
import os

import numpy as np

from comun import (ARCHIVO_VERDAD, DIR_LOTE, DIR_RESULTADOS, VARIANTES,
                   Clasificador, Registro, cargar_etiquetas, exigir_modelo,
                   fallar, ok, preprocesar, seccion)


def cargar_lote():
    if not os.path.isfile(ARCHIVO_VERDAD):
        fallar(f"No existe '{ARCHIVO_VERDAD}'. Ejecute primero el paso 1.")
    with open(ARCHIVO_VERDAD, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    faltan = [r["archivo"] for r in filas
              if not os.path.isfile(os.path.join(DIR_LOTE, r["archivo"]))]
    if faltan:
        fallar(f"Faltan {len(faltan)} imágenes del lote. Ejecute de nuevo el paso 1.")
    return [(r["archivo"], int(r["clase"])) for r in filas]


def main():
    for _, _, ruta in VARIANTES:
        exigir_modelo(ruta, 4 if "dynamic" in ruta else 1)
    etiquetas = cargar_etiquetas()
    lote = cargar_lote()

    seccion("Preprocesamiento del lote")
    tensores = [preprocesar(os.path.join(DIR_LOTE, a)) for a, _ in lote]
    verdad = np.array([c for _, c in lote])
    print(f"{len(tensores)} imágenes listas (1x3x224x224 float32 cada una).")
    print("Nota: son imágenes de muestra de ImageNet y algunas pueden haberse usado en el")
    print("entrenamiento; la precisión absoluta puede ser optimista. Lo importante aquí es")
    print("la COMPARACIÓN entre variantes, que usan exactamente las mismas imágenes.")

    hilos = min(4, os.cpu_count() or 1)
    resultados, top1_fp32, prob_fp32 = {}, None, None
    for corto, descripcion, ruta in VARIANTES:
        seccion(f"{corto}: {descripcion}")
        clf = Clasificador(ruta, hilos=hilos)
        probs = np.stack([clf.probabilidades(x) for x in tensores])
        top5 = np.argsort(probs, axis=1)[:, ::-1][:, :5]
        top1 = top5[:, 0]
        acierto1 = float(np.mean(top1 == verdad))
        acierto5 = float(np.mean([v in fila for v, fila in zip(verdad, top5)]))
        if top1_fp32 is None:
            top1_fp32, prob_fp32 = top1, probs
        concordancia = float(np.mean(top1 == top1_fp32))
        dif_conf = float(np.mean(np.abs(probs[np.arange(len(top1)), top1_fp32]
                                        - prob_fp32[np.arange(len(top1)), top1_fp32])))
        print(f"  Top-1: {acierto1 * 100:5.1f} %   Top-5: {acierto5 * 100:5.1f} %   "
              f"Concordancia con FP32: {concordancia * 100:5.1f} %")
        print(f"  Diferencia media de confianza en la clase elegida por FP32: "
              f"{dif_conf * 100:.2f} puntos porcentuales")
        cambios = np.flatnonzero(top1 != top1_fp32)
        if corto != "FP32":
            if len(cambios) == 0:
                print("  No cambió ninguna predicción respecto a FP32.")
            else:
                print(f"  Predicciones distintas a FP32 ({len(cambios)}), primeras 5:")
                for i in cambios[:5]:
                    print(f"    {lote[i][0][:30]:30s} FP32: {etiquetas[top1_fp32[i]][:18]:18s} "
                          f"-> {etiquetas[top1[i]][:18]}")
        resultados[corto] = {"descripcion": descripcion, "top1": acierto1,
                             "top5": acierto5, "concordancia_fp32": concordancia,
                             "cambios": int(len(cambios))}

    seccion("Tabla de precisión")
    print(f"  {'Variante':8s} {'Top-1':>8s} {'Top-5':>8s} {'Concordancia FP32':>19s}")
    for corto, r in resultados.items():
        print(f"  {corto:8s} {r['top1'] * 100:7.1f}% {r['top5'] * 100:7.1f}% "
              f"{r['concordancia_fp32'] * 100:18.1f}%")

    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    with open(os.path.join(DIR_RESULTADOS, "precision.json"), "w", encoding="utf-8") as f:
        json.dump({"imagenes": len(lote), "variantes": resultados}, f, indent=2)
    print()
    ok("Paso 6 completado: resultados guardados en resultados/precision.json")


if __name__ == "__main__":
    with Registro(6, "Precisión de las tres variantes"):
        main()
