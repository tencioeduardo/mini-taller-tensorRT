#!/usr/bin/env python3
"""
PASO 7 - Resumen final: la tabla comparativa medida en SU equipo.

Reúne los resultados de los pasos 4, 5 y 6 en una sola tabla (la versión
medida de la diapositiva "Evaluación: tamaño, latencia y precisión") y
verifica que existan todos los registros que van en el PDF de entrega.

Uso:
  python 07_resumen.py
"""
import json
import os

from comun import DIR_RESULTADOS, Registro, fallar, ok, seccion


def leer_json(nombre, paso):
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    if not os.path.isfile(ruta):
        fallar(f"No existe '{ruta}'. Ejecute primero el paso {paso}.")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def modelo_cpu():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.lower().startswith(("model name", "hardware")):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return "desconocido"


def main():
    bench = leer_json("benchmark.json", 5)
    prec = leer_json("precision.json", 6)
    vb, vp = bench["variantes"], prec["variantes"]
    hilos = [f["hilos"] for f in vb["FP32"]["latencias"]]
    h_max = hilos[-1]

    seccion("Equipo")
    print(f"CPU: {modelo_cpu()} | núcleos lógicos: {bench['n_cpu']}")

    seccion("Tabla comparativa final")
    col_hmax = f"Lat {h_max}h"
    enc = (f"  {'Variante':8s} {'Archivo':>8s} {'Reduc.':>7s} {'Lat 1h':>8s} {col_hmax:>8s} "
           f"{'RAM+':>7s} {'Top-1':>6s} {'Concord.':>8s} {'XNNPACK':>8s}")
    print(enc)
    print("  " + "-" * (len(enc) - 2))
    tam_ref = vb["FP32"]["tamano_mb"]
    for corto in vb:
        b, p = vb[corto], vp[corto]
        lat = [f["mediana_ms"] for f in b["latencias"]]
        print(f"  {corto:8s} {b['tamano_mb']:5.2f} MB {100 * (1 - b['tamano_mb'] / tam_ref):6.1f}% "
              f"{lat[0]:8.2f} {lat[-1]:8.2f} {b['ram_modelo_mb']:4.0f} MB "
              f"{p['top1'] * 100:5.0f}% {p['concordancia_fp32'] * 100:7.0f}% "
              f"{'completo' if b['latencias'][0]['acelerado'] else 'parcial':>8s}")
    print(f"\n  Archivo = tamaño en disco. Lat 1h / {col_hmax} = latencia mediana en ms con 1 y"
          f" {h_max} hilos.")
    print("  RAM+ = memoria que agrega el modelo cargado y ejecutándose.")
    print(f"  Top-1 y Concord. (concordancia con FP32) sobre {prec['imagenes']} imágenes del lote.")

    seccion("Observaciones calculadas a partir de sus mediciones")
    lat1 = {k: v["latencias"][0]["mediana_ms"] for k, v in vb.items()}
    for corto in ("DIN-W8", "INT8"):
        r_tam = tam_ref / vb[corto]["tamano_mb"]
        r_ram = vb["FP32"]["ram_modelo_mb"] / max(vb[corto]["ram_modelo_mb"], 1e-6)
        r_lat = lat1["FP32"] / lat1[corto]
        vel = (f"{r_lat:.2f}x más rápida" if r_lat >= 1 else f"{1 / r_lat:.2f}x más lenta")
        print(f"  * {corto} frente a FP32: archivo {r_tam:.1f}x más pequeño, "
              f"RAM extra {r_ram:.1f}x menor,")
        print(f"    inferencia {vel} (1 hilo), concordancia "
              f"{vp[corto]['concordancia_fp32'] * 100:.0f} %, "
              f"grafo {'completo' if vb[corto]['latencias'][0]['acelerado'] else 'partido'} en XNNPACK.")
    if h_max > 1:
        acel = lat1["FP32"] / vb["FP32"]["latencias"][-1]["mediana_ms"]
        print(f"  * Pasar de 1 a {h_max} hilos aceleró FP32 {acel:.2f}x.")

    seccion("Registros para el PDF de entrega")
    faltantes = 0
    for paso in range(8):
        ruta = os.path.join(DIR_RESULTADOS, f"paso_{paso:02d}.txt")
        existe = os.path.isfile(ruta) or paso == 7
        faltantes += not existe
        print(f"  [{'✓' if existe else '✗'}] {ruta}")
    print()
    if faltantes:
        print(f"[!] Faltan {faltantes} registros: ejecute los pasos marcados con ✗.")
    else:
        ok("Paso 7 completado: todos los registros están listos para el PDF.")


if __name__ == "__main__":
    with Registro(7, "Resumen comparativo"):
        main()
