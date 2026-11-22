"""
Separa as imagens do PlantVillage em dados/ (treino, 80%) e teste/ (20%).

Uso:
  python preparar_dados.py CAMINHO_DO_PLANTVILLAGE [--limite 500]

CAMINHO_DO_PLANTVILLAGE é a pasta que contém as subpastas Tomato___healthy,
Tomato___Late_blight etc. (no repositório do GitHub: PlantVillage-Dataset/raw/color).
--limite define o máximo de imagens por classe (menos imagens = treino mais rápido).
"""
import argparse
import os
import random
import shutil

from classificador import MAPA_PLANTVILLAGE, PASTA_PROJETO

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp"}


def preparar(origem, limite=500, proporcao_teste=0.2, semente=42):
    random.seed(semente)
    for pasta_original, classe in MAPA_PLANTVILLAGE.items():
        dir_origem = os.path.join(origem, pasta_original)
        if not os.path.isdir(dir_origem):
            print(f"AVISO: pasta não encontrada: {dir_origem}")
            continue
        imagens = [f for f in os.listdir(dir_origem) if os.path.splitext(f)[1].lower() in EXTENSOES]
        random.shuffle(imagens)
        imagens = imagens[:limite]
        n_teste = int(len(imagens) * proporcao_teste)

        for destino, lista in (("teste", imagens[:n_teste]), ("dados", imagens[n_teste:])):
            dir_destino = os.path.join(PASTA_PROJETO, destino, classe)
            os.makedirs(dir_destino, exist_ok=True)
            for arq in lista:
                shutil.copy2(os.path.join(dir_origem, arq), dir_destino)
        print(f"{classe:<20} treino: {len(imagens) - n_teste:>5}   teste: {n_teste:>5}")
    print("\nPronto! Pastas criadas em:", os.path.join(PASTA_PROJETO, "dados"),
          "e", os.path.join(PASTA_PROJETO, "teste"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("origem")
    p.add_argument("--limite", type=int, default=500)
    a = p.parse_args()
    preparar(a.origem, a.limite)
