# Diagnóstico de Doenças em Folhas de Tomate
**Inteligência Artificial (classificação de imagens) + Sistema Especialista (regras)**

## Problema
Pequenos agricultores perdem parte da colheita por não identificarem a tempo as doenças das plantas.
O sistema recebe a foto de uma folha e informa a doença, o grau de certeza e o tratamento.

## Arquitetura
1. **IA – `classificador.py`**: rede MobileNetV2 (transferência de aprendizado) classifica a foto em
   `saudavel`, `requeima`, `pinta_preta` ou `mancha_bacteriana`, com nível de confiança.
2. **Sistema especialista – `sistema_especialista.py`**:
   - Base de fatos: resultado da IA + respostas do usuário
   - Base de conhecimento: 19 regras SE… ENTÃO…
   - Motor de inferência: encadeamento para frente (forward chaining)
   - Módulo de explicação: mostra as regras usadas
3. **Programa principal – `main.py`**: junta as duas partes.

## Instalação
```
pip install -r requirements.txt
```

## Uso
```
python main.py demo                      # testa as regras sem treinar a IA
python main.py treinar --dados dados/    # treina a IA
python main.py diagnosticar folha.jpg    # diagnóstico completo
```

## Dados para treino
Baixe o dataset **PlantVillage** (Kaggle) e copie para `dados/` apenas as pastas:
`Tomato___healthy`, `Tomato___Late_blight`, `Tomato___Early_blight`, `Tomato___Bacterial_spot`.
