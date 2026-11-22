"""
Sistema de diagnóstico de doenças em folhas de tomate
IA (classificação de imagens) + Sistema Especialista (regras)

Uso:
  python main.py treinar --dados pasta_com_imagens   # treina a IA
  python main.py diagnosticar foto.jpg               # diagnóstico completo
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

    args = p.parse_args()

    # Sem argumentos (ex.: botão Executar do VS Code): mostra um menu
    if args.comando is None:
        print("=== Diagnóstico de doenças em folhas de tomate ===")
        print("  1. Demonstração (testar as regras sem a IA)")
        print("  2. Diagnosticar uma foto")
        print("  3. Treinar a IA")
        opcao = input("Escolha (1-3): ").strip()
        if opcao == "2":
            args.comando, args.imagem = "diagnosticar", input("Caminho da foto: ").strip().strip('"')
        elif opcao == "3":
            args.comando, args.dados = "treinar", input("Pasta com as imagens: ").strip().strip('"')
            args.epocas = 5
        else:
            args.comando = "demo"

    if args.comando == "treinar":
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
