"""
Sistema especialista para diagnóstico de doenças em folhas de tomate.

Componentes clássicos de um sistema especialista:
  - Base de fatos ........ o que se sabe sobre o caso (resultado da IA + respostas do usuário)
  - Base de conhecimento . regras SE ... ENTÃO ... escritas a partir do conhecimento de um agrônomo
  - Motor de inferência .. encadeamento para frente (forward chaining)
  - Módulo de explicação . mostra quais regras levaram à conclusão
"""


# ---------------------------------------------------------------------------
# Funções auxiliares para escrever condições
# ---------------------------------------------------------------------------
def maior_igual(limite):
    return lambda v: v >= limite


def menor(limite):
    return lambda v: v < limite


def entre(minimo, maximo):
    return lambda v: minimo <= v < maximo


def um_de(*valores):
    return lambda v: v in valores


# ---------------------------------------------------------------------------
# Estrutura de uma regra
# ---------------------------------------------------------------------------
class Regra:
    def __init__(self, nome, se, entao, porque):
        self.nome = nome          # identificador da regra
        self.se = se              # condições: {fato: valor esperado ou função}
        self.entao = entao        # conclusões: {fato: valor}
        self.porque = porque      # texto usado na explicação

    def pode_disparar(self, fatos):
        for fato, esperado in self.se.items():
            if fato not in fatos:
                return False
            valor = fatos[fato]
            if callable(esperado):
                if not esperado(valor):
                    return False
            elif valor != esperado:
                return False
        return True


# ---------------------------------------------------------------------------
# Base de conhecimento
# ---------------------------------------------------------------------------
REGRAS = [
    # --- Avaliação da confiança da IA ------------------------------------
    Regra("R1", {"confianca": maior_igual(0.80)}, {"nivel_confianca": "alta"},
          "A IA teve confiança de 80% ou mais."),
    Regra("R2", {"confianca": entre(0.50, 0.80)}, {"nivel_confianca": "media"},
          "A IA teve confiança entre 50% e 80%."),
    Regra("R3", {"confianca": menor(0.50)}, {"nivel_confianca": "baixa"},
          "A IA teve confiança abaixo de 50%."),

    # --- Foto ruim / resultado incerto -----------------------------------
    Regra("R4", {"nivel_confianca": "baixa"},
          {"doenca": "indeterminada", "certeza": "inconclusivo"},
          "Com confiança baixa, a imagem não é suficiente para diagnosticar."),

    # --- Planta saudável -------------------------------------------------
    Regra("R5", {"classe_ia": "saudavel", "nivel_confianca": um_de("alta", "media")},
          {"doenca": "nenhuma", "certeza": "confirmado"},
          "A IA classificou a folha como saudável."),

    # --- Requeima (Phytophthora infestans) -------------------------------
    Regra("R6", {"classe_ia": "requeima", "nivel_confianca": "alta", "umidade_alta": True},
          {"doenca": "requeima", "certeza": "confirmado"},
          "Imagem indica requeima e a umidade alta favorece o fungo."),
    Regra("R7", {"classe_ia": "requeima", "nivel_confianca": "media", "mofo_branco_verso": True},
          {"doenca": "requeima", "certeza": "confirmado"},
          "Imagem sugere requeima e há mofo branco no verso da folha (sinal típico)."),
    Regra("R8", {"classe_ia": "requeima", "nivel_confianca": "alta", "umidade_alta": False},
          {"doenca": "requeima", "certeza": "provavel"},
          "Imagem indica requeima, mas o clima seco não é típico da doença."),

    # --- Pinta-preta (Alternaria solani) ---------------------------------
    Regra("R9", {"classe_ia": "pinta_preta", "nivel_confianca": um_de("alta", "media"),
                 "aneis_concentricos": True},
          {"doenca": "pinta_preta", "certeza": "confirmado"},
          "Imagem indica pinta-preta e as manchas têm anéis concêntricos (sinal típico)."),
    Regra("R10", {"classe_ia": "pinta_preta", "nivel_confianca": "alta",
                  "aneis_concentricos": False},
          {"doenca": "pinta_preta", "certeza": "provavel"},
          "Imagem indica pinta-preta, mas sem os anéis concêntricos típicos."),

    # --- Mancha bacteriana (Xanthomonas spp.) ----------------------------
    Regra("R11", {"classe_ia": "mancha_bacteriana", "nivel_confianca": um_de("alta", "media"),
                  "halo_amarelo": True},
          {"doenca": "mancha_bacteriana", "certeza": "confirmado"},
          "Imagem indica mancha bacteriana e as manchas têm halo amarelo (sinal típico)."),
    Regra("R12", {"classe_ia": "mancha_bacteriana", "nivel_confianca": "alta",
                  "halo_amarelo": False},
          {"doenca": "mancha_bacteriana", "certeza": "provavel"},
          "Imagem indica mancha bacteriana, mas sem halo amarelo."),

    # --- Recomendações ---------------------------------------------------
    Regra("R13", {"doenca": "nenhuma"},
          {"recomendacao": "Nenhuma ação necessária. Continue monitorando a lavoura semanalmente."},
          "Planta saudável não precisa de tratamento."),
    Regra("R14", {"doenca": "requeima"},
          {"recomendacao": "Remova e destrua as folhas atacadas, evite irrigar por cima das "
                           "folhas e aplique fungicida à base de cobre ou mancozebe."},
          "Tratamento indicado para requeima."),
    Regra("R15", {"doenca": "pinta_preta"},
          {"recomendacao": "Retire as folhas mais velhas e atacadas, faça rotação de culturas "
                           "e aplique fungicida (ex.: clorotalonil)."},
          "Tratamento indicado para pinta-preta."),
    Regra("R16", {"doenca": "mancha_bacteriana"},
          {"recomendacao": "Use sementes sadias, evite trabalhar com as plantas molhadas e "
                           "aplique produto cúprico. Fungicidas comuns não funcionam (é bactéria)."},
          "Tratamento indicado para mancha bacteriana."),
    Regra("R17", {"doenca": "indeterminada"},
          {"recomendacao": "Tire outra foto com boa luz, folha inteira e fundo neutro. "
                           "Se persistir, consulte um agrônomo."},
          "Sem diagnóstico, é preciso uma nova imagem ou um especialista."),

    # --- Alertas ---------------------------------------------------------
    Regra("R18", {"doenca": "requeima", "umidade_alta": True},
          {"alerta": "URGENTE: com umidade alta a requeima pode destruir a lavoura em poucos dias."},
          "Requeima se espalha muito rápido em clima úmido."),
    Regra("R19", {"certeza": "provavel"},
          {"alerta": "Diagnóstico provável, não confirmado. Recomenda-se confirmar com um agrônomo."},
          "Os sintomas observados não batem totalmente com a imagem."),
]


# ---------------------------------------------------------------------------
# Perguntas feitas ao usuário conforme a classe prevista pela IA
# ---------------------------------------------------------------------------
PERGUNTAS = {
    "requeima": [
        ("umidade_alta", "A umidade do ar está alta (chuva, neblina, orvalho frequente)?"),
        ("mofo_branco_verso", "Há um mofo esbranquiçado no verso da folha?"),
    ],
    "pinta_preta": [
        ("aneis_concentricos", "As manchas têm anéis concêntricos (como um alvo)?"),
    ],
    "mancha_bacteriana": [
        ("halo_amarelo", "As manchas são pequenas, escuras e com um halo amarelo em volta?"),
    ],
    "saudavel": [],
}

NOMES = {
    "nenhuma": "Nenhuma doença (planta saudável)",
    "requeima": "Requeima (Phytophthora infestans)",
    "pinta_preta": "Pinta-preta (Alternaria solani)",
    "mancha_bacteriana": "Mancha bacteriana (Xanthomonas spp.)",
    "indeterminada": "Não foi possível determinar",
}


# ---------------------------------------------------------------------------
# Motor de inferência: encadeamento para frente
# ---------------------------------------------------------------------------
class MotorInferencia:
    def __init__(self, regras=REGRAS):
        self.regras = regras

    def inferir(self, fatos_iniciais):
        fatos = dict(fatos_iniciais)
        disparadas = []
        mudou = True
        while mudou:                       # repete até não surgir fato novo
            mudou = False
            for regra in self.regras:
                if regra in disparadas or not regra.pode_disparar(fatos):
                    continue
                # só dispara se acrescentar algo novo à base de fatos
                novos = {f: v for f, v in regra.entao.items() if f not in fatos}
                if not novos:
                    continue
                fatos.update(novos)
                disparadas.append(regra)
                mudou = True

        # regra padrão caso nenhuma regra tenha chegado a um diagnóstico
        if "doenca" not in fatos:
            fatos["doenca"] = "indeterminada"
            fatos["certeza"] = "inconclusivo"
            fatos["recomendacao"] = ("Os sintomas não correspondem às regras conhecidas. "
                                     "Consulte um agrônomo.")
        return fatos, disparadas


def explicar(fatos, disparadas):
    """Módulo de explicação: monta o relatório em texto."""
    linhas = [
        "=" * 60,
        "DIAGNÓSTICO",
        "=" * 60,
        f"Doença ......: {NOMES.get(fatos['doenca'], fatos['doenca'])}",
        f"Certeza .....: {fatos.get('certeza', '-')}",
        f"Recomendação : {fatos.get('recomendacao', '-')}",
    ]
    if "alerta" in fatos:
        linhas.append(f"Alerta ......: {fatos['alerta']}")
    linhas += ["", "Por que cheguei a essa conclusão:"]
    for r in disparadas:
        linhas.append(f"  [{r.nome}] {r.porque}")
    return "\n".join(linhas)
