# Diagnóstico de Doenças em Folhas de Tomate
**Inteligência Artificial (classificação de imagens) + Sistema Especialista (regras)**

## Problema
Pequenos agricultores perdem parte da colheita por não identificarem a tempo as doenças das plantas.
O sistema recebe a foto de uma folha e informa a doença, o grau de certeza e o tratamento.

## Arquitetura
| Arquivo | Função |
|---|---|
| `classificador.py` | **IA**: rede MobileNetV2 (transferência de aprendizado) classifica a foto em `saudavel`, `requeima`, `pinta_preta` ou `mancha_bacteriana` |
| `sistema_especialista.py` | **Sistema especialista**: base de fatos, 19 regras SE… ENTÃO…, motor de inferência (encadeamento para frente) e módulo de explicação |
| `avaliacao.py` | **Métricas**: acurácia, precisão, recall, F1-score e matriz de confusão |
| `interface.py` | **Interface gráfica moderna** (CustomTkinter, modo claro e escuro) |
| `main.py` | Programa principal |

## Instalação
```
pip install -r requirements.txt
```
(No Windows com Python da Microsoft Store, use `python3.11` ou o caminho completo do `python3.11.exe`.)

## Uso
```
python main.py                                 # abre a interface gráfica
python main.py treinar --dados dados/          # treina a IA
python main.py avaliar --teste teste/          # calcula as métricas
python main.py diagnosticar folha.jpg          # diagnóstico pelo terminal
python main.py demo                            # testa as regras sem a IA
```

### Interface gráfica
- **Aba Diagnóstico**: abra uma foto → veja a imagem e as probabilidades da IA → marque os sintomas → clique em *Diagnosticar*.
- **Aba Métricas**: *Treinar IA* (escolha a pasta `dados/`) e *Avaliar pasta de teste* (escolha `teste/`).
  Mostra acurácia, tabela por classe e matriz de confusão, e salva `relatorio_metricas.txt` e `matriz_confusao.png`.

## Dados
Baixe o dataset **PlantVillage** (`git clone https://github.com/spMohanty/PlantVillage-Dataset`) e rode:
```
python preparar_dados.py PlantVillage-Dataset/raw/color --limite 500
```
O script copia as 4 classes de tomate e separa 80% para `dados/` e 20% para `teste/`. Para fazer manualmente, copie as pastas
`Tomato___healthy`, `Tomato___Late_blight`, `Tomato___Early_blight` e `Tomato___Bacterial_spot`:

```
dados/   ~80% das imagens de cada classe (treino)
teste/   ~20% das imagens de cada classe (NUNCA usadas no treino)
```
Para um teste com fotos reais de campo, use o dataset **PlantDoc**.

## Métricas
- **Acurácia**: acertos ÷ total de imagens
- **Precisão**: das imagens que a IA disse ser da classe X, quantas eram realmente X
- **Recall**: das imagens que eram da classe X, quantas a IA acertou
- **F1-score**: média harmônica entre precisão e recall
- **Matriz de confusão**: linhas = classe real, colunas = classe prevista
