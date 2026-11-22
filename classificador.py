"""
Classificador de imagens de folhas de tomate (Inteligência Artificial).

Usa transferência de aprendizado com a rede MobileNetV2, pré-treinada no ImageNet.
Só a última camada é treinada, então funciona bem mesmo em um computador comum.

Estrutura esperada da pasta de dados (ex.: subconjunto do dataset PlantVillage):

    dados/
      saudavel/            (Tomato___healthy)
      requeima/            (Tomato___Late_blight)
      pinta_preta/         (Tomato___Early_blight)
      mancha_bacteriana/   (Tomato___Bacterial_spot)

As pastas com os nomes originais do PlantVillage também são aceitas.
"""
import json
import os

TAMANHO = (224, 224)
ARQ_MODELO = "modelo_folhas.keras"
ARQ_CLASSES = "classes.json"

# nomes originais do PlantVillage -> nomes usados pelo sistema especialista
MAPA_PLANTVILLAGE = {
    "Tomato___healthy": "saudavel",
    "Tomato___Late_blight": "requeima",
    "Tomato___Early_blight": "pinta_preta",
    "Tomato___Bacterial_spot": "mancha_bacteriana",
}


def treinar(pasta_dados, epocas=5):
    import tensorflow as tf
    from tensorflow import keras

    treino = keras.utils.image_dataset_from_directory(
        pasta_dados, validation_split=0.2, subset="training", seed=42,
        image_size=TAMANHO, batch_size=32)
    validacao = keras.utils.image_dataset_from_directory(
        pasta_dados, validation_split=0.2, subset="validation", seed=42,
        image_size=TAMANHO, batch_size=32)

    classes = [MAPA_PLANTVILLAGE.get(c, c) for c in treino.class_names]
    print("Classes encontradas:", classes)

    treino = treino.prefetch(tf.data.AUTOTUNE)
    validacao = validacao.prefetch(tf.data.AUTOTUNE)

    base = keras.applications.MobileNetV2(
        input_shape=TAMANHO + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # congela a rede pré-treinada

    modelo = keras.Sequential([
        keras.Input(shape=TAMANHO + (3,)),
        keras.layers.RandomFlip("horizontal"),       # aumento de dados
        keras.layers.RandomRotation(0.1),
        keras.layers.Rescaling(1 / 127.5, offset=-1),  # pré-processamento da MobileNetV2
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(classes), activation="softmax"),
    ])
    modelo.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    modelo.fit(treino, validation_data=validacao, epochs=epocas)

    _, acuracia = modelo.evaluate(validacao)
    print(f"Acurácia na validação: {acuracia:.1%}")

    modelo.save(ARQ_MODELO)
    with open(ARQ_CLASSES, "w", encoding="utf-8") as f:
        json.dump(classes, f)
    print(f"Modelo salvo em {ARQ_MODELO}")


def classificar(caminho_imagem):
    """Retorna (classe, confiança, {classe: probabilidade})."""
    if not os.path.exists(ARQ_MODELO):
        raise FileNotFoundError("Modelo não encontrado. Rode primeiro: python main.py treinar --dados <pasta>")

    import numpy as np
    from tensorflow import keras

    modelo = keras.models.load_model(ARQ_MODELO)
    with open(ARQ_CLASSES, encoding="utf-8") as f:
        classes = json.load(f)

    img = keras.utils.load_img(caminho_imagem, target_size=TAMANHO)
    x = np.expand_dims(keras.utils.img_to_array(img), 0)
    probs = modelo.predict(x, verbose=0)[0]

    i = int(np.argmax(probs))
    return classes[i], float(probs[i]), {c: float(p) for c, p in zip(classes, probs)}
