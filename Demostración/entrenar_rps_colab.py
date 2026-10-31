%pip install -q importlib-resources

# Entrena un clasificador Piedra/Papel/Tijera y lo exporta a .tflite (LiteRT).
# Ejecutar en Google Colab (GPU opcional): pegar en una celda o subir y correr
#   !python entrenar_rps_colab.py
# Salida: rps_model.tflite y labels.txt  (entrada 150x150x3 float32 en [0,1], salida softmax de 3 clases)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

IMG, BATCH, EPOCHS = 150, 64, 12

(ds_train, ds_test), info = tfds.load(
    "rock_paper_scissors", split=["train", "test"], as_supervised=True, with_info=True)
print("Orden de clases en el dataset:", info.features["label"].names)  # rock, paper, scissors


def prep_eval(img, lbl):
    img = tf.image.resize(img, (IMG, IMG))
    return tf.cast(img, tf.float32) / 255.0, lbl


def prep_train(img, lbl):
    img = tf.image.random_flip_left_right(img)
    crop = tf.random.uniform([], 220, 301, dtype=tf.int32)            # zoom aleatorio
    img = tf.image.random_crop(img, tf.stack([crop, crop, 3]))
    img = tf.cast(tf.image.resize(img, (IMG, IMG)), tf.float32) / 255.0
    img = tf.image.random_brightness(img, 0.25)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    return tf.clip_by_value(img, 0.0, 1.0), lbl


AUTOTUNE = tf.data.AUTOTUNE
train = ds_train.shuffle(1000).map(prep_train, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
test = ds_test.map(prep_eval, AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

L = tf.keras.layers
model = tf.keras.Sequential([
    L.Input((IMG, IMG, 3)),
    L.Conv2D(32, 3, activation="relu"), L.MaxPooling2D(),
    L.Conv2D(64, 3, activation="relu"), L.MaxPooling2D(),
    L.Conv2D(128, 3, activation="relu"), L.MaxPooling2D(),
    L.Conv2D(128, 3, activation="relu"), L.MaxPooling2D(),
    L.Flatten(), L.Dropout(0.5),
    L.Dense(128, activation="relu"),
    L.Dense(3, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=EPOCHS, validation_data=test)

# --- Conversión a TFLite (cuantización dinámica: entrada/salida siguen siendo float32) ---
model.export("rps_saved_model")
converter = tf.lite.TFLiteConverter.from_saved_model("rps_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
assert tflite_bytes[4:8] == b"TFL3", "El archivo generado no es un FlatBuffer TFLite"

with open("rps_model.tflite", "wb") as f:
    f.write(tflite_bytes)
with open("labels.txt", "w", encoding="utf-8") as f:
    f.write("Piedra\nPapel\nTijera\n")   # mismo orden que el dataset: rock, paper, scissors

# --- Verificación rápida del .tflite con el intérprete ---
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
i, o = interp.get_input_details()[0], interp.get_output_details()[0]
print("Entrada:", i["shape"], i["dtype"].__name__, "| Salida:", o["shape"])

correct = total = 0
for x, y in test.unbatch().take(300):
    interp.set_tensor(i["index"], x[None].numpy())
    interp.invoke()
    correct += int(np.argmax(interp.get_tensor(o["index"])) == int(y))
    total += 1
print(f"Precisión del .tflite en {total} imágenes de prueba: {correct / total:.1%}")

try:
    from google.colab import files
    files.download("rps_model.tflite")
    files.download("labels.txt")
except ImportError:
    print("Copie rps_model.tflite y labels.txt a la carpeta de la demo.")