#!/usr/bin/env bash
# Prepara el entorno de la demo LiteRT. NO descarga el modelo a ciegas:
# valida que rps_model.tflite sea un FlatBuffer TFLite real.

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

is_valid_tflite() {
    # Un .tflite válido tiene el identificador "TFL3" en los bytes 4-7
    [ -s "$1" ] && [ "$(head -c 8 "$1" 2>/dev/null | tail -c 4 2>/dev/null)" = "TFL3" ]
}

echo "======================================================"
echo "  Configuración de entorno - LiteRT"
echo "======================================================"

# 1. Paquetes del sistema
if ! dpkg -s python3-venv >/dev/null 2>&1 || ! dpkg -s python3-pip >/dev/null 2>&1; then
    echo "[!] Instalando python3-venv y python3-pip..."
    sudo apt update && sudo apt install -y python3-venv python3-pip || exit 1
fi

# 2. Entorno virtual
if [ ! -d "$VENV_DIR" ]; then
    echo "[+] Creando entorno virtual '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR" || exit 1
else
    echo "[✓] Entorno virtual detectado."
fi
"$PIP_BIN" install --upgrade pip >/dev/null 2>&1

# 3. Dependencias de Python (Pillow ya no es necesario)
echo "[+] Verificando dependencias de Python..."
declare -A PACKAGES=(
    ["ai-edge-litert"]="ai_edge_litert"
    ["opencv-python"]="cv2"
    ["numpy"]="numpy"
)
for pkg in "${!PACKAGES[@]}"; do
    if ! "$PYTHON_BIN" -c "import ${PACKAGES[$pkg]}" >/dev/null 2>&1; then
        echo "   [-] Instalando $pkg..."
        "$PIP_BIN" install "$pkg" || exit 1
    else
        echo "   [✓] $pkg"
    fi
done

# 4. Validación del modelo
echo "[+] Verificando recursos..."
if is_valid_tflite rps_model.tflite; then
    echo "   [✓] rps_model.tflite es un modelo TFLite válido."
else
    if [ -e rps_model.tflite ]; then
        echo "   [✗] rps_model.tflite existe pero NO es un modelo válido:"
        echo "       $(head -c 120 rps_model.tflite | tr '\n' ' ')"
        echo "       Se renombra a rps_model.tflite.invalido"
        mv -f rps_model.tflite rps_model.tflite.invalido
    fi
    echo "   [!] Falta rps_model.tflite. Genérelo con entrenar_rps_colab.py (Google Colab)"
    echo "       o expórtelo desde Teachable Machine, y cópielo a esta carpeta."
fi

if [ -f labels.txt ]; then
    echo "   [✓] labels.txt ($(wc -l < labels.txt) líneas):"
    sed 's/^/       /' labels.txt | head -5
else
    echo "   [!] labels.txt no existe (se usará Piedra/Papel/Tijera por defecto)."
fi

[ -f test_hand.jpg ] || echo "   [i] test_hand.jpg: se crea con la tecla 's' durante la demo."

echo "======================================================"
echo "  Siguiente paso: source .venv/bin/activate && python litert_rps.py"
echo "======================================================"
