"""
Sistema de diagnóstico de doenças em folhas de tomate
IA (classificação de imagens) + Sistema Especialista (regras)

Uso:
  python main.py                                     # abre a interface gráfica
  python main.py treinar --dados pasta_com_imagens   # treina a IA
  python main.py avaliar --teste pasta_de_teste      # calcula as métricas
  python main.py diagnosticar foto.jpg               # diagnóstico no terminal
  python main.py demo                                # testa o sistema especialista sem a IA
"""
import argparse

from sistema_especialista import PERGUNTAS, MotorInferencia, explicar

CLASSES = ["saudavel", "requeima", "pinta_preta", "mancha_bacteriana"]


def perguntar_sim_nao(texto):
    while True:
        r = input(f"{texto} (s/n): ").strip().lower()
        if r in ("s", "sim"):
            return True
        if r in ("n", "nao", "não"):
            return False
        print("Responda com s ou n.")


def coletar_fatos(classe, confianca):
    """Junta o resultado da IA com as respostas do usuário."""
    fatos = {"classe_ia": classe, "confianca": confianca}
    for fato, pergunta in PERGUNTAS.get(classe, []):
        fatos[fato] = perguntar_sim_nao(pergunta)
    return fatos


def diagnosticar(classe, confianca):
    fatos = coletar_fatos(classe, confianca)
    resultado, regras = MotorInferencia().inferir(fatos)
    print()
    print(explicar(resultado, regras))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="comando")

    t = sub.add_parser("treinar", help="treina o classificador de imagens")
    t.add_argument("--dados", required=True)
    t.add_argument("--epocas", type=int, default=5)

    d = sub.add_parser("diagnosticar", help="diagnostica a partir de uma foto")
    d.add_argument("imagem")

    sub.add_parser("demo", help="simula o resultado da IA para testar as regras")

    a = sub.add_parser("avaliar", help="calcula as métricas em uma pasta de teste")
    a.add_argument("--teste", required=True)

    sub.add_parser("interface", help="abre a interface gráfica (padrão)")

    args = p.parse_args()

    # Sem argumentos (ex.: botão Executar do VS Code): abre a interface gráfica
    if args.comando in (None, "interface"):
        from interface import iniciar
        iniciar()

    elif args.comando == "avaliar":
        from avaliacao import avaliar, relatorio_texto, salvar_resultados
        m = avaliar(args.teste, progresso=lambda f, t: print(f"\r{f}/{t} imagens", end=""))
        print("\n")
        print(relatorio_texto(m))
        txt, png = salvar_resultados(m)
        print(f"\nSalvo em: {txt} e {png}")

    elif args.comando == "treinar":
        from classificador import treinar
        treinar(args.dados, args.epocas)

    elif args.comando == "diagnosticar":
        from classificador import classificar
        classe, conf, probs = classificar(args.imagem)
        print("\nResultado da IA (classificação da imagem):")
        for c, pr in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"  {c:<18} {pr:6.1%}")
        print(f"\nClasse prevista: {classe} ({conf:.1%})\n")
        diagnosticar(classe, conf)

    elif args.comando == "demo":
        print("Modo demonstração: você informa o que a IA teria respondido.\n")
        for i, c in enumerate(CLASSES, 1):
            print(f"  {i}. {c}")
        classe = CLASSES[int(input("Classe prevista pela IA (1-4): ")) - 1]
        conf = float(input("Confiança da IA (0 a 100): ")) / 100
        print()
        diagnosticar(classe, conf)


if __name__ == "__main__":
    main()
