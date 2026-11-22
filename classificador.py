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
PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))
ARQ_MODELO = os.path.join(PASTA_PROJETO, "modelo_folhas.keras")
ARQ_CLASSES = os.path.join(PASTA_PROJETO, "classes.json")

# nomes originais do PlantVillage -> nomes usados pelo sistema especialista
MAPA_PLANTVILLAGE = {
    "Tomato___healthy": "saudavel",
    "Tomato___Late_blight": "requeima",
    "Tomato___Early_blight": "pinta_preta",
    "Tomato___Bacterial_spot": "mancha_bacteriana",
}

_modelo = None
_classes = None


def modelo_existe():
    return os.path.exists(ARQ_MODELO) and os.path.exists(ARQ_CLASSES)


def treinar(pasta_dados, epocas=5):
    """Treina o modelo e devolve a acurácia na validação."""
    global _modelo, _classes
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
        keras.layers.RandomFlip("horizontal"),         # aumento de dados
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

    _modelo, _classes = modelo, classes
    return float(acuracia)


def carregar():
    """Carrega o modelo do disco uma única vez."""
    global _modelo, _classes
    if _modelo is None:
        if not modelo_existe():
            raise FileNotFoundError(
                "Modelo não encontrado. Treine a IA primeiro "
                "(aba Métricas > Treinar IA, ou: python main.py treinar --dados <pasta>).")
        from tensorflow import keras
        _modelo = keras.models.load_model(ARQ_MODELO)
        with open(ARQ_CLASSES, encoding="utf-8") as f:
            _classes = json.load(f)
    return _modelo, _classes


def classes_do_modelo():
    return carregar()[1]


def _ler_imagem(caminho):
    import numpy as np
    from PIL import Image
    img = Image.open(caminho).convert("RGB").resize(TAMANHO)
    return np.asarray(img, dtype="float32")


def classificar_lote(caminhos, tamanho_lote=32, progresso=None):
    """Classifica várias imagens. Devolve lista de (classe, confiança, {classe: prob})."""
    import numpy as np
    modelo, classes = carregar()
    resultados = []
    for inicio in range(0, len(caminhos), tamanho_lote):
        parte = caminhos[inicio:inicio + tamanho_lote]
        x = np.stack([_ler_imagem(c) for c in parte])
        probs = modelo.predict(x, verbose=0)
        for p in probs:
            i = int(np.argmax(p))
            resultados.append((classes[i], float(p[i]),
                               {c: float(v) for c, v in zip(classes, p)}))
        if progresso:
            progresso(len(resultados), len(caminhos))
    return resultados


def classificar(caminho_imagem):
    """Classifica uma imagem. Devolve (classe, confiança, {classe: prob})."""
    return classificar_lote([caminho_imagem])[0]
