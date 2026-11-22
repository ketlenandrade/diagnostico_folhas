"""
Avaliação do classificador: calcula métricas em uma pasta de teste.

Estrutura da pasta de teste (imagens que NÃO foram usadas no treino):

    teste/
      saudavel/
      requeima/
      pinta_preta/
      mancha_bacteriana/

Métricas calculadas:
  - Acurácia ........ acertos / total
  - Precisão ........ das imagens que a IA disse ser da classe X, quantas eram mesmo X
  - Recall .......... das imagens que eram da classe X, quantas a IA acertou
  - F1-score ........ média harmônica entre precisão e recall
  - Matriz de confusão (linhas = classe real, colunas = classe prevista)
"""
import os

from classificador import MAPA_PLANTVILLAGE, PASTA_PROJETO

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagens(pasta):
    """Devolve (caminhos, classes_reais) a partir das subpastas."""
    caminhos, reais = [], []
    for sub in sorted(os.listdir(pasta)):
        dir_classe = os.path.join(pasta, sub)
        if not os.path.isdir(dir_classe):
            continue
        classe = MAPA_PLANTVILLAGE.get(sub, sub)
        for arq in sorted(os.listdir(dir_classe)):
            if os.path.splitext(arq)[1].lower() in EXTENSOES:
                caminhos.append(os.path.join(dir_classe, arq))
                reais.append(classe)
    return caminhos, reais


def calcular_metricas(reais, previstos, classes):
    """Calcula as métricas a partir das listas de classes reais e previstas."""
    classes = list(classes) + sorted(set(reais) - set(classes))
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    matriz = [[0] * n for _ in range(n)]
    for r, p in zip(reais, previstos):
        matriz[idx[r]][idx[p]] += 1

    total = len(reais)
    acertos = sum(matriz[i][i] for i in range(n))

    por_classe = {}
    for i, c in enumerate(classes):
        vp = matriz[i][i]                                   # verdadeiros positivos
        fp = sum(matriz[j][i] for j in range(n)) - vp       # falsos positivos
        fn = sum(matriz[i]) - vp                            # falsos negativos
        precisao = vp / (vp + fp) if vp + fp else 0.0
        recall = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        por_classe[c] = {"precisao": precisao, "recall": recall, "f1": f1,
                         "suporte": sum(matriz[i])}

    com_amostras = [m for m in por_classe.values() if m["suporte"] > 0]
    k = len(com_amostras) or 1
    macro = {
        "precisao": sum(m["precisao"] for m in com_amostras) / k,
        "recall": sum(m["recall"] for m in com_amostras) / k,
        "f1": sum(m["f1"] for m in com_amostras) / k,
    }
    return {
        "classes": classes,
        "matriz": matriz,
        "total": total,
        "acertos": acertos,
        "acuracia": acertos / total if total else 0.0,
        "por_classe": por_classe,
        "macro": macro,
    }


def avaliar(pasta_teste, progresso=None, funcao_classificar=None):
    """Classifica todas as imagens da pasta e calcula as métricas."""
    if funcao_classificar is None:
        from classificador import classificar_lote as funcao_classificar
        from classificador import classes_do_modelo
        classes = classes_do_modelo()
    else:
        classes = None

    caminhos, reais = listar_imagens(pasta_teste)
    if not caminhos:
        raise ValueError("Nenhuma imagem encontrada. A pasta deve ter uma subpasta por classe.")

    resultados = funcao_classificar(caminhos, progresso=progresso)
    previstos = [r[0] for r in resultados]
    if classes is None:
        classes = sorted(set(reais) | set(previstos))

    metricas = calcular_metricas(reais, previstos, classes)
    metricas["erros"] = [(c, r, p) for c, r, p in zip(caminhos, reais, previstos) if r != p]
    return metricas


def relatorio_texto(m):
    linhas = [
        "RELATÓRIO DE MÉTRICAS DO CLASSIFICADOR",
        "=" * 62,
        f"Imagens avaliadas: {m['total']}   Acertos: {m['acertos']}",
        f"Acurácia: {m['acuracia']:.2%}",
        "",
        f"{'Classe':<20}{'Precisão':>10}{'Recall':>10}{'F1':>10}{'Suporte':>10}",
        "-" * 62,
    ]
    for c in m["classes"]:
        pc = m["por_classe"][c]
        linhas.append(f"{c:<20}{pc['precisao']:>10.2%}{pc['recall']:>10.2%}"
                      f"{pc['f1']:>10.2%}{pc['suporte']:>10}")
    mc = m["macro"]
    linhas += [
        "-" * 62,
        f"{'Média (macro)':<20}{mc['precisao']:>10.2%}{mc['recall']:>10.2%}{mc['f1']:>10.2%}",
        "",
        "Matriz de confusão (linhas = real, colunas = previsto):",
        " " * 20 + "".join(f"{c[:9]:>10}" for c in m["classes"]),
    ]
    for c, linha in zip(m["classes"], m["matriz"]):
        linhas.append(f"{c:<20}" + "".join(f"{v:>10}" for v in linha))
    if m.get("erros"):
        linhas += ["", f"Imagens classificadas errado ({len(m['erros'])}):"]
        for caminho, r, p in m["erros"][:50]:
            nome = os.path.join(os.path.basename(os.path.dirname(caminho)), os.path.basename(caminho))
            linhas.append(f"  {nome}: real={r}, previsto={p}")
    return "\n".join(linhas)


def figura_matriz(m, escuro=False, fundo=None):
    """Cria a figura (matplotlib) da matriz de confusão, no tema claro ou escuro."""
    from matplotlib.figure import Figure
    classes, matriz = m["classes"], m["matriz"]
    fundo = fundo or ("#1C1D21" if escuro else "#FFFFFF")
    texto = "#E5E7EB" if escuro else "#1F2937"
    fig = Figure(figsize=(4.5, 4.2), dpi=100, facecolor=fundo)
    ax = fig.add_subplot(111)
    ax.set_facecolor(fundo)
    im = ax.imshow(matriz, cmap="Greens")
    barra = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    barra.ax.tick_params(colors=texto, labelsize=8)
    barra.outline.set_visible(False)
    ax.set_xticks(range(len(classes)), labels=classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), labels=classes, fontsize=8)
    ax.tick_params(colors=texto, length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xlabel("Classe prevista", color=texto, fontsize=9)
    ax.set_ylabel("Classe real", color=texto, fontsize=9)
    maior = max(max(l) for l in matriz) or 1
    for i, linha in enumerate(matriz):
        for j, v in enumerate(linha):
            ax.text(j, i, v, ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if v > maior / 2 else "#1F2937")
    fig.tight_layout()
    return fig


def salvar_resultados(m, pasta_saida=PASTA_PROJETO):
    """Salva relatorio_metricas.txt e matriz_confusao.png. Devolve os caminhos."""
    txt = os.path.join(pasta_saida, "relatorio_metricas.txt")
    png = os.path.join(pasta_saida, "matriz_confusao.png")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(relatorio_texto(m))
    figura_matriz(m).savefig(png)
    return txt, png
