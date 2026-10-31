#!/usr/bin/env bash
# =============================================================================
# PASO 4 - Cuantización dinámica propia con LiteRT Quantizer (línea de comandos)
#
# Receta dynamic_wi8_afp32:
#   wi8   -> pesos (weights) en INT8
#   afp32 -> activaciones en FP32 (se calculan en flotante durante la inferencia)
# No requiere datos de calibración: solo el modelo FP32.
# =============================================================================

ENTRADA="modelos/mobilenet_v2.tflite"
SALIDA="modelos/mobilenet_v2_dynamic_wi8_afp32.tflite"
RECETA="dynamic_wi8_afp32"
REGISTRO="resultados/paso_04.txt"

# ---- Verificaciones previas ------------------------------------------------
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[✗] El entorno virtual no está activo. Ejecute: source .venv/bin/activate"
    exit 1
fi
if ! command -v litert-quantizer >/dev/null 2>&1; then
    echo "[✗] No se encontró el comando litert-quantizer. Vuelva a ejecutar ./setup.sh"
    exit 1
fi
if [ ! -f "$ENTRADA" ]; then
    echo "[✗] No existe $ENTRADA. Ejecute primero el paso 1 (python 01_descargar.py)."
    exit 1
fi
# Un .tflite válido tiene el identificador "TFL3" en los bytes 4-7
if [ "$(head -c 8 "$ENTRADA" | tail -c 4)" != "TFL3" ]; then
    echo "[✗] $ENTRADA no es un modelo .tflite válido. Bórrelo y repita el paso 1."
    exit 1
fi
mkdir -p resultados

{
    echo "======================================================================"
    echo "  PASO 4: Cuantización dinámica con litert-quantizer"
    echo "  $(date '+%Y-%m-%d %H:%M:%S') | equipo: $(hostname)"
    echo "======================================================================"
    echo
    echo "--- 4.1 Comando ejecutado -------------------------------------------"
    echo "litert-quantizer --model_file=$ENTRADA \\"
    echo "                 --recipe=$RECETA --output_dir=modelos --overwrite_outputs"
    echo
    echo "--- 4.2 Salida de la herramienta ------------------------------------"
    rm -f "$SALIDA"
    INICIO=$(date +%s%N)
    litert-quantizer --model_file="$ENTRADA" --recipe="$RECETA" \
                     --output_dir=modelos --overwrite_outputs 2>&1 \
        | grep -vE '^(INFO|WARNING): \[|^└|Applying Transformations'
    FIN=$(date +%s%N)
    if [ ! -f "$SALIDA" ]; then
        echo "[✗] La herramienta no generó $SALIDA. Revise los mensajes anteriores."
        exit 1
    fi
    echo "Tiempo total del comando (incluye arranque de Python): $(( (FIN - INICIO) / 1000000 )) ms"
    echo
    echo "--- 4.3 Comparación de tamaños --------------------------------------"
    python - "$ENTRADA" "$SALIDA" <<'EOF'
import os, sys
a, b = (os.path.getsize(p) for p in sys.argv[1:3])
print(f"  Original FP32        : {a/1e6:6.2f} MB  ({a:,} bytes)")
print(f"  Dinámica wi8/afp32   : {b/1e6:6.2f} MB  ({b:,} bytes)")
print(f"  Reducción            : {100*(1-b/a):.1f} %  ({a/b:.2f}x más pequeño)")
EOF
    echo
    echo "--- 4.4 Inspección del modelo cuantizado ----------------------------"
    python 02_inspeccionar.py --sin-registro "$SALIDA"
    echo
    echo "[✓] Paso 4 completado: modelo cuantizado creado."
} 2>&1 | tee "$REGISTRO"

echo
echo "[i] Salida guardada en $REGISTRO"
