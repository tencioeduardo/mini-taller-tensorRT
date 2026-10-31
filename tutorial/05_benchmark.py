#!/usr/bin/env python3
"""
PASO 5 - Medir latencia y memoria de las tres variantes en SU equipo.

Para cada variante (FP32, dinámica propia, INT8 del Hub) y cada número de hilos:
  * Tiempo de carga y compilación del CompiledModel.
  * Latencia de inferencia: se descartan las primeras ejecuciones (calentamiento)
    y se reportan mínimo, mediana y percentil 90 de N repeticiones.
  * Si todo el grafo quedó en el acelerador de CPU (XNNPACK).
  * RAM adicional en ejecución, medida en un proceso aparte para cada modelo.

Uso:
  python 05_benchmark.py                     # 100 repeticiones
  python 05_benchmark.py --repeticiones 300  # medición más estable
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from comun import (DIR_LOTE, DIR_RESULTADOS, VARIANTES,
                   Clasificador, Registro, exigir_modelo, fallar, ok,
                   preprocesar, seccion, tamano_mb)


def entrada_de_prueba():
    """Usa la primera imagen del lote (el contenido no cambia la latencia)."""
    ruta = os.path.join(DIR_LOTE, "n01440764_tench.JPEG")
    if not os.path.isfile(ruta):
        fallar(f"No existe '{ruta}'. Ejecute primero el paso 1.")
    return preprocesar(ruta)


def medir_ram(ruta):
    """Ejecuta este mismo script en un proceso nuevo para medir la RAM de un solo modelo."""
    salida = subprocess.run([sys.executable, __file__, "--medir-ram", ruta],
                            capture_output=True, text=True)
    for linea in salida.stdout.splitlines():
        if linea.startswith("{"):
            return json.loads(linea)
    fallar(f"No se pudo medir la RAM de {ruta}:\n{salida.stderr[-500:]}")


def pico_ram_mb():
    """Pico de memoria residente del proceso (VmHWM en /proc/self/status)."""
    with open("/proc/self/status", encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("VmHWM:"):
                return int(linea.split()[1]) / 1024          # kB -> MB
    return 0.0


def proceso_medir_ram(ruta):
    """(Proceso hijo) Pico de RAM antes y después de cargar el modelo y ejecutarlo."""
    x = entrada_de_prueba()
    base = pico_ram_mb()
    clf = Clasificador(ruta, hilos=1)
    for _ in range(5):
        clf.logits(x)
    pico = pico_ram_mb()
    print(json.dumps({"base_mb": base, "pico_mb": pico, "modelo_mb": pico - base}))


def medir_latencia(ruta, hilos, x, calentamiento, repeticiones):
    clf = Clasificador(ruta, hilos=hilos)
    for _ in range(calentamiento):
        clf.logits(x)
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        clf.logits(x)
        tiempos.append((time.perf_counter() - t0) * 1000)
    t = np.array(tiempos)
    return {"hilos": hilos, "compilacion_ms": clf.ms_compilacion,
            "min_ms": float(t.min()), "mediana_ms": float(np.median(t)),
            "p90_ms": float(np.percentile(t, 90)),
            "acelerado": bool(clf.totalmente_acelerado)}


def main(args):
    for _, _, ruta in VARIANTES:
        exigir_modelo(ruta, 4 if "dynamic" in ruta else 1)
    x = entrada_de_prueba()
    n_cpu = os.cpu_count() or 1
    lista_hilos = sorted({1, min(4, n_cpu)})

    seccion("Configuración")
    print(f"Núcleos lógicos: {n_cpu} | hilos a probar: {lista_hilos}")
    print(f"Calentamiento: {args.calentamiento} ejecuciones | medición: {args.repeticiones} ejecuciones")
    print("Consejo: cierre otros programas y conecte el cargador si es una laptop.")

    resultados = {}
    for corto, descripcion, ruta in VARIANTES:
        seccion(f"{corto}: {descripcion}")
        print(f"Archivo: {ruta} ({tamano_mb(ruta):.2f} MB)")
        filas = []
        for hilos in lista_hilos:
            r = medir_latencia(ruta, hilos, x, args.calentamiento, args.repeticiones)
            filas.append(r)
            print(f"  hilos={hilos}: compilación {r['compilacion_ms']:6.1f} ms | "
                  f"latencia mín {r['min_ms']:6.2f}  mediana {r['mediana_ms']:6.2f}  "
                  f"p90 {r['p90_ms']:6.2f} ms")
        ram = medir_ram(ruta)
        print(f"  Totalmente acelerado en XNNPACK: {'sí' if filas[0]['acelerado'] else 'NO'}")
        print(f"  RAM: proceso base {ram['base_mb']:.0f} MB -> pico {ram['pico_mb']:.0f} MB "
              f"(el modelo agrega {ram['modelo_mb']:.1f} MB)")
        resultados[corto] = {"descripcion": descripcion, "archivo": ruta,
                             "tamano_mb": tamano_mb(ruta), "latencias": filas,
                             "ram_modelo_mb": ram["modelo_mb"]}

    seccion("Tabla de latencia (mediana, ms)")
    print(f"  {'Variante':8s} " + " ".join(f"{f'{h} hilo(s)':>12s}" for h in lista_hilos)
          + f" {'vs FP32 (1 hilo)':>18s}")
    ref = resultados["FP32"]["latencias"][0]["mediana_ms"]
    for corto, r in resultados.items():
        meds = [f["mediana_ms"] for f in r["latencias"]]
        rel = ref / meds[0]
        if corto == "FP32":
            texto_rel = "referencia"
        elif rel >= 1:
            texto_rel = f"{rel:.2f}x más rápido"
        else:
            texto_rel = f"{1 / rel:.2f}x más lento"
        print(f"  {corto:8s} " + " ".join(f"{m:12.2f}" for m in meds) + f" {texto_rel:>18s}")

    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    with open(os.path.join(DIR_RESULTADOS, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump({"n_cpu": n_cpu, "repeticiones": args.repeticiones,
                   "variantes": resultados}, f, indent=2)
    print()
    ok("Paso 5 completado: mediciones guardadas en resultados/benchmark.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeticiones", type=int, default=100)
    p.add_argument("--calentamiento", type=int, default=10)
    p.add_argument("--medir-ram", metavar="MODELO", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.medir_ram:
        proceso_medir_ram(args.medir_ram)
    else:
        with Registro(5, "Latencia y memoria de las tres variantes"):
            main(args)
