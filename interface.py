"""
Interface gráfica moderna (CustomTkinter) do sistema de diagnóstico de folhas de tomate.

Página "Diagnóstico": escolhe uma foto, mostra a imagem, as probabilidades da IA,
                      os sintomas (sistema especialista) e o diagnóstico.
Página "Métricas":    treina a IA e avalia uma pasta de teste (acurácia, precisão,
                      recall, F1 e matriz de confusão).
"""
import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

import avaliacao
import classificador
from sistema_especialista import NOMES, PERGUNTAS, MotorInferencia

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Paleta (claro, escuro)
FUNDO = ("#F3F5F7", "#111214")
LATERAL = ("#FFFFFF", "#18191C")
CARTAO = ("#FFFFFF", "#1C1D21")
BORDA = ("#E5E7EB", "#2A2C31")
AREA = ("#F3F4F6", "#25272C")
TEXTO = ("#111827", "#F3F4F6")
SUAVE = ("#6B7280", "#9CA3AF")
VERDE = ("#16A34A", "#22C55E")
VERDE_HOVER = ("#15803D", "#16A34A")
VERDE_FUNDO = ("#DCFCE7", "#14301F")
CINZA_BARRA = ("#D1D5DB", "#3A3D44")
ALERTA_FUNDO = ("#FEF3C7", "#33270C")
ALERTA_TEXTO = ("#92400E", "#FCD34D")

CERTEZA = {
    "confirmado": ("#16A34A", "✔  Confirmado"),
    "provavel": ("#D97706", "◐  Provável"),
    "inconclusivo": ("#DC2626", "✖  Inconclusivo"),
}
ROTULOS = {
    "saudavel": "Saudável",
    "requeima": "Requeima",
    "pinta_preta": "Pinta-preta",
    "mancha_bacteriana": "Mancha bacteriana",
}


def fonte(tamanho, peso="normal"):
    return ctk.CTkFont(family="Segoe UI", size=tamanho, weight=peso)


class Cartao(ctk.CTkFrame):
    """Painel com cantos arredondados e título discreto."""

    def __init__(self, master, titulo=None, **kw):
        super().__init__(master, fg_color=CARTAO, corner_radius=16,
                         border_width=1, border_color=BORDA, **kw)
        if titulo:
            ctk.CTkLabel(self, text=titulo.upper(), font=fonte(11, "bold"),
                         text_color=SUAVE).pack(anchor="w", padx=20, pady=(16, 6))


class App(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=FUNDO)
        self.title("FolhaScan — Diagnóstico de doenças em folhas de tomate")
        self.geometry("1280x820")
        self.minsize(1120, 740)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.resultado_ia = None
        self.respostas = {}
        self.metricas = None
        self._canvas = None
        self.status = ctk.StringVar(value="")

        self._montar_lateral()
        self.paginas = {
            "diag": self._pagina_diagnostico(),
            "met": self._pagina_metricas(),
        }
        self.mostrar("diag")
        self._atualizar_modelo()

    # ================================================================ lateral
    def _montar_lateral(self):
        lat = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=LATERAL)
        lat.grid(row=0, column=0, sticky="nsw")
        lat.pack_propagate(False)

        ctk.CTkLabel(lat, text="🍅  FolhaScan", font=fonte(24, "bold"),
                     text_color=TEXTO).pack(anchor="w", padx=24, pady=(30, 0))
        ctk.CTkLabel(lat, text="IA + Sistema Especialista", font=fonte(12),
                     text_color=SUAVE).pack(anchor="w", padx=26, pady=(2, 30))

        self.botoes_nav = {}
        for chave, texto in (("diag", "🔍    Diagnóstico"), ("met", "📊    Métricas")):
            b = ctk.CTkButton(lat, text=texto, anchor="w", height=44, corner_radius=12,
                              font=fonte(14), fg_color="transparent", text_color=TEXTO,
                              hover_color=AREA, command=lambda c=chave: self.mostrar(c))
            b.pack(fill="x", padx=14, pady=3)
            self.botoes_nav[chave] = b

        base = ctk.CTkFrame(lat, fg_color="transparent")
        base.pack(side="bottom", fill="x", padx=20, pady=22)
        ctk.CTkLabel(base, textvariable=self.status, font=fonte(11), text_color=SUAVE,
                     wraplength=200, justify="left", anchor="w").pack(fill="x", pady=(0, 12))
        self.lbl_modelo = ctk.CTkLabel(base, text="", font=fonte(12, "bold"), anchor="w")
        self.lbl_modelo.pack(fill="x", pady=(0, 12))
        self.sw_tema = ctk.CTkSwitch(base, text="Modo escuro", font=fonte(12),
                                     progress_color=VERDE, command=self._trocar_tema)
        self.sw_tema.select()
        self.sw_tema.pack(anchor="w")

    def mostrar(self, chave):
        for c, pg in self.paginas.items():
            pg.grid_forget()
            self.botoes_nav[c].configure(fg_color="transparent", text_color=TEXTO)
        self.paginas[chave].grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        self.botoes_nav[chave].configure(fg_color=VERDE_FUNDO, text_color=VERDE)

    def _trocar_tema(self):
        ctk.set_appearance_mode("dark" if self.sw_tema.get() else "light")
        if self.metricas:
            self._desenhar_matriz()

    def _atualizar_modelo(self):
        if classificador.modelo_existe():
            self.lbl_modelo.configure(text="●  Modelo treinado", text_color=VERDE)
        else:
            self.lbl_modelo.configure(text="●  Modelo não treinado", text_color="#DC2626")
            self.status.set("Treine a IA na página Métricas antes de diagnosticar.")

    def _cabecalho(self, pai, titulo, subtitulo):
        cab = ctk.CTkFrame(pai, fg_color="transparent")
        ctk.CTkLabel(cab, text=titulo, font=fonte(28, "bold"), text_color=TEXTO).pack(anchor="w")
        ctk.CTkLabel(cab, text=subtitulo, font=fonte(13), text_color=SUAVE).pack(anchor="w")
        return cab

    def _em_segundo_plano(self, tarefa, ao_terminar, msg):
        """Roda uma tarefa demorada sem travar a janela."""
        self.status.set(msg)
        self.configure(cursor="watch")

        def executar():
            try:
                r = tarefa()
                self.after(0, lambda: self._fim(ao_terminar, r, None))
            except Exception as e:  # noqa: BLE001
                self.after(0, lambda e=e: self._fim(ao_terminar, None, e))

        threading.Thread(target=executar, daemon=True).start()

    def _fim(self, ao_terminar, resultado, erro):
        self.configure(cursor="")
        if erro:
            self.status.set("Erro: " + str(erro))
            self.barra.stop()
            self.barra.configure(mode="determinate")
            self.barra.set(0)
            messagebox.showerror("Erro", str(erro))
        else:
            ao_terminar(resultado)

    # ============================================================ diagnóstico
    def _pagina_diagnostico(self):
        pg = ctk.CTkFrame(self, fg_color="transparent")
        pg.grid_columnconfigure(1, weight=1)
        pg.grid_rowconfigure(1, weight=1)
        self._cabecalho(pg, "Diagnóstico",
                        "Envie a foto de uma folha de tomate para identificar doenças."
                        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 18))

        # --- imagem
        esq = Cartao(pg, "Imagem da folha")
        esq.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        self.area_img = ctk.CTkLabel(esq, text="📷\n\nClique para escolher\numa foto da folha",
                                     width=360, height=360, fg_color=AREA, corner_radius=14,
                                     font=fonte(15), text_color=SUAVE, cursor="hand2")
        self.area_img.pack(padx=20, pady=(4, 10))
        self.area_img.bind("<Button-1>", lambda _e: self.abrir_imagem())
        self.lbl_arquivo = ctk.CTkLabel(esq, text="Formatos: JPG, PNG, BMP, WEBP",
                                        font=fonte(11), text_color=SUAVE, wraplength=350)
        self.lbl_arquivo.pack(padx=20)
        ctk.CTkButton(esq, text="Escolher imagem", height=44, corner_radius=12,
                      font=fonte(14, "bold"), fg_color=VERDE, hover_color=VERDE_HOVER,
                      command=self.abrir_imagem).pack(fill="x", padx=20, pady=(12, 20))

        # --- coluna da direita (rolável)
        dir_ = self.col_diag = ctk.CTkScrollableFrame(pg, fg_color="transparent")
        dir_.grid(row=1, column=1, sticky="nsew")

        c_ia = Cartao(dir_, "1 · Classificação da IA")
        c_ia.pack(fill="x", pady=(0, 14))
        self.lbl_top = ctk.CTkLabel(c_ia, text="Aguardando imagem…", font=fonte(22, "bold"),
                                    text_color=TEXTO)
        self.lbl_top.pack(anchor="w", padx=20)
        self.lbl_conf = ctk.CTkLabel(c_ia, text="A IA vai estimar a probabilidade de cada classe.",
                                     font=fonte(12), text_color=SUAVE)
        self.lbl_conf.pack(anchor="w", padx=20, pady=(0, 8))
        self.frm_barras = ctk.CTkFrame(c_ia, fg_color="transparent")
        self.frm_barras.pack(fill="x", padx=20, pady=(0, 16))

        c_sint = Cartao(dir_, "2 · Sintomas observados")
        c_sint.pack(fill="x", pady=(0, 14))
        self.frm_sint = ctk.CTkFrame(c_sint, fg_color="transparent")
        self.frm_sint.pack(fill="x", padx=20)
        ctk.CTkLabel(self.frm_sint, text="As perguntas aparecem depois da classificação.",
                     font=fonte(12), text_color=SUAVE).pack(anchor="w")
        self.btn_diag = ctk.CTkButton(c_sint, text="Diagnosticar  →", height=42, width=200,
                                      corner_radius=12, font=fonte(14, "bold"), fg_color=VERDE,
                                      hover_color=VERDE_HOVER, state="disabled",
                                      command=self.diagnosticar)
        self.btn_diag.pack(anchor="w", padx=20, pady=(12, 18))

        c_res = Cartao(dir_, "3 · Diagnóstico")
        c_res.pack(fill="x")
        self.frm_res = ctk.CTkFrame(c_res, fg_color="transparent")
        self.frm_res.pack(fill="x", padx=20, pady=(0, 18))
        self._quebraveis = []
        self.frm_res.bind("<Configure>", lambda e: self._ajustar_quebra(e.width))
        self._resultado_vazio()
        return pg

    def _ajustar_quebra(self, largura):
        """Faz o texto do diagnóstico quebrar conforme a largura do painel."""
        if largura == getattr(self, "_largura_res", None):
            return  # evita laço de redimensionamento
        self._largura_res = largura
        for lbl, recuo in self._quebraveis:
            if lbl.winfo_exists():
                lbl.configure(wraplength=max(200, largura - recuo))

    def _quebravel(self, lbl, recuo=10):
        self._quebraveis.append((lbl, recuo))
        lbl.configure(wraplength=max(200, getattr(self, "_largura_res", 600) - recuo))
        return lbl

    def _limpar(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _resultado_vazio(self):
        self._limpar(self.frm_res)
        ctk.CTkLabel(self.frm_res, text="O resultado aparecerá aqui.", font=fonte(12),
                     text_color=SUAVE).pack(anchor="w")

    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Escolha a foto da folha",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")])
        if not caminho:
            return
        img = Image.open(caminho)
        img.thumbnail((360, 360))
        foto = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.area_img.configure(image=foto, text="")
        self.lbl_arquivo.configure(text=caminho)
        self.btn_diag.configure(state="disabled")
        self.lbl_top.configure(text="Analisando…")
        self._resultado_vazio()
        self._em_segundo_plano(lambda: classificador.classificar(caminho),
                               self._mostrar_ia, "Classificando a imagem…")

    def _mostrar_ia(self, resultado):
        self.resultado_ia = resultado
        classe, conf, probs = resultado
        self.lbl_top.configure(text=ROTULOS.get(classe, classe))
        self.lbl_conf.configure(text=f"Classe mais provável  ·  confiança de {conf:.1%}")

        self._limpar(self.frm_barras)
        for c, p in sorted(probs.items(), key=lambda x: -x[1]):
            linha = ctk.CTkFrame(self.frm_barras, fg_color="transparent")
            linha.pack(fill="x", pady=4)
            destaque = c == classe
            ctk.CTkLabel(linha, text=ROTULOS.get(c, c), width=150, anchor="w",
                         font=fonte(13, "bold" if destaque else "normal"),
                         text_color=TEXTO).pack(side="left")
            barra = ctk.CTkProgressBar(linha, height=10, corner_radius=5, fg_color=CINZA_BARRA,
                                       progress_color=VERDE if destaque else SUAVE)
            barra.set(p)
            barra.pack(side="left", fill="x", expand=True, padx=10)
            ctk.CTkLabel(linha, text=f"{p:.1%}", width=60, anchor="e",
                         font=fonte(13, "bold" if destaque else "normal"),
                         text_color=TEXTO).pack(side="left")

        self._limpar(self.frm_sint)
        self.respostas = {}
        perguntas = PERGUNTAS.get(classe, [])
        if not perguntas:
            ctk.CTkLabel(self.frm_sint, text="Nenhuma pergunta necessária para esta classe.",
                         font=fonte(12), text_color=SUAVE).pack(anchor="w")
        for fato, texto in perguntas:
            sw = ctk.CTkSwitch(self.frm_sint, text=texto, font=fonte(13),
                               progress_color=VERDE, text_color=TEXTO)
            sw.pack(anchor="w", pady=5)
            self.respostas[fato] = sw

        self.btn_diag.configure(state="normal")
        self.status.set("Marque os sintomas observados e clique em Diagnosticar.")

    def diagnosticar(self):
        if not self.resultado_ia:
            return
        classe, conf, _ = self.resultado_ia
        fatos = {"classe_ia": classe, "confianca": conf}
        fatos.update({f: bool(sw.get()) for f, sw in self.respostas.items()})
        res, regras = MotorInferencia().inferir(fatos)

        self._limpar(self.frm_res)
        self._quebraveis = []
        cor, texto_badge = CERTEZA.get(res.get("certeza"), ("#6B7280", res.get("certeza", "-")))

        topo = ctk.CTkFrame(self.frm_res, fg_color="transparent")
        topo.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(topo, text=NOMES.get(res["doenca"], res["doenca"]), font=fonte(22, "bold"),
                     text_color=TEXTO).pack(side="left")
        ctk.CTkLabel(topo, text=texto_badge, font=fonte(12, "bold"), text_color="white",
                     fg_color=cor, corner_radius=10, height=28, width=150).pack(side="left", padx=14)

        ctk.CTkLabel(self.frm_res, text="RECOMENDAÇÃO", font=fonte(11, "bold"),
                     text_color=SUAVE).pack(anchor="w")
        self._quebravel(ctk.CTkLabel(self.frm_res, text=res.get("recomendacao", "-"), font=fonte(13),
                                     text_color=TEXTO, justify="left", anchor="w")
                        ).pack(anchor="w", pady=(2, 12))

        if "alerta" in res:
            caixa = ctk.CTkFrame(self.frm_res, fg_color=ALERTA_FUNDO, corner_radius=12)
            caixa.pack(fill="x", pady=(0, 12))
            self._quebravel(ctk.CTkLabel(caixa, text="⚠  " + res["alerta"], font=fonte(13, "bold"),
                                         text_color=ALERTA_TEXTO, justify="left", anchor="w"),
                            recuo=40).pack(anchor="w", padx=14, pady=10)

        ctk.CTkLabel(self.frm_res, text="POR QUE CHEGUEI A ESSA CONCLUSÃO", font=fonte(11, "bold"),
                     text_color=SUAVE).pack(anchor="w", pady=(0, 4))
        for r in regras:
            linha = ctk.CTkFrame(self.frm_res, fg_color="transparent")
            linha.pack(fill="x", pady=2)
            ctk.CTkLabel(linha, text=r.nome, width=44, height=22, corner_radius=6,
                         fg_color=AREA, text_color=TEXTO,
                         font=ctk.CTkFont(family="Consolas", size=11, weight="bold")
                         ).pack(side="left")
            self._quebravel(ctk.CTkLabel(linha, text=r.porque, font=fonte(12), text_color=TEXTO,
                                         justify="left", anchor="w"),
                            recuo=70).pack(side="left", padx=10)
        self.status.set("Diagnóstico concluído.")

    # =============================================================== métricas
    def _pagina_metricas(self):
        pg = ctk.CTkFrame(self, fg_color="transparent")
        pg.grid_columnconfigure(0, weight=1)
        pg.grid_rowconfigure(3, weight=1)
        self._cabecalho(pg, "Métricas do modelo",
                        "Treine a IA e avalie o desempenho em imagens que ela nunca viu."
                        ).grid(row=0, column=0, sticky="w", pady=(0, 18))

        # --- ações
        acoes = Cartao(pg)
        acoes.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        lin = ctk.CTkFrame(acoes, fg_color="transparent")
        lin.pack(fill="x", padx=18, pady=14)
        ctk.CTkButton(lin, text="🧠  Treinar IA", height=40, corner_radius=12, font=fonte(13, "bold"),
                      fg_color=VERDE, hover_color=VERDE_HOVER, command=self.treinar).pack(side="left")
        ctk.CTkLabel(lin, text="Épocas", font=fonte(12), text_color=SUAVE).pack(side="left", padx=(16, 6))
        self.opt_epocas = ctk.CTkSegmentedButton(lin, values=["3", "5", "10", "20"], font=fonte(12),
                                                 selected_color=VERDE, selected_hover_color=VERDE_HOVER)
        self.opt_epocas.set("5")
        self.opt_epocas.pack(side="left")
        ctk.CTkButton(lin, text="📊  Avaliar pasta de teste", height=40, corner_radius=12,
                      font=fonte(13, "bold"), fg_color="transparent", border_width=2,
                      border_color=VERDE, text_color=VERDE, hover_color=VERDE_FUNDO,
                      command=self.avaliar).pack(side="left", padx=(24, 0))
        self.barra = ctk.CTkProgressBar(lin, height=10, corner_radius=5, progress_color=VERDE,
                                        fg_color=CINZA_BARRA)
        self.barra.set(0)
        self.barra.pack(side="left", fill="x", expand=True, padx=(24, 0))

        # --- indicadores
        kpis = ctk.CTkFrame(pg, fg_color="transparent")
        kpis.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        self.kpi = {}
        for i, (chave, titulo, desc) in enumerate((
                ("acuracia", "Acurácia", "acertos ÷ total"),
                ("precisao", "Precisão média", "acerto quando a IA afirma"),
                ("recall", "Recall médio", "casos reais encontrados"),
                ("f1", "F1-score médio", "equilíbrio precisão × recall"))):
            kpis.grid_columnconfigure(i, weight=1, uniform="kpi")
            c = Cartao(kpis)
            c.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 7, 0 if i == 3 else 7))
            ctk.CTkLabel(c, text=titulo.upper(), font=fonte(11, "bold"),
                         text_color=SUAVE).pack(anchor="w", padx=18, pady=(14, 0))
            v = ctk.CTkLabel(c, text="—", font=fonte(32, "bold"), text_color=TEXTO)
            v.pack(anchor="w", padx=18)
            ctk.CTkLabel(c, text=desc, font=fonte(11), text_color=SUAVE).pack(anchor="w", padx=18,
                                                                               pady=(0, 14))
            self.kpi[chave] = v

        # --- tabela + matriz
        baixo = ctk.CTkFrame(pg, fg_color="transparent")
        baixo.grid(row=3, column=0, sticky="nsew")
        baixo.grid_columnconfigure(0, weight=1)
        baixo.grid_rowconfigure(0, weight=1)

        esq = Cartao(baixo, "Desempenho por classe")
        esq.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.frm_tabela = ctk.CTkFrame(esq, fg_color="transparent")
        self.frm_tabela.pack(fill="x", padx=20)
        ctk.CTkLabel(self.frm_tabela, text="Avalie uma pasta de teste para ver as métricas.",
                     font=fonte(12), text_color=SUAVE).pack(anchor="w")
        ctk.CTkLabel(esq, text="IMAGENS CLASSIFICADAS ERRADO", font=fonte(11, "bold"),
                     text_color=SUAVE).pack(anchor="w", padx=20, pady=(14, 4))
        self.txt_erros = ctk.CTkTextbox(esq, height=110, corner_radius=10, fg_color=AREA,
                                        font=ctk.CTkFont(family="Consolas", size=11))
        self.txt_erros.pack(fill="both", expand=True, padx=20, pady=(0, 18))

        self.c_matriz = Cartao(baixo, "Matriz de confusão")
        self.c_matriz.grid(row=0, column=1, sticky="nsew")
        self.frm_matriz = ctk.CTkFrame(self.c_matriz, fg_color="transparent", width=450, height=420)
        self.frm_matriz.pack(padx=12, pady=(0, 12))
        self.frm_matriz.pack_propagate(False)
        self.lbl_matriz_vazia = ctk.CTkLabel(self.frm_matriz, text="Ainda sem avaliação.",
                                             font=fonte(12), text_color=SUAVE)
        self.lbl_matriz_vazia.pack(expand=True)
        return pg

    def treinar(self):
        pasta = filedialog.askdirectory(title="Pasta com as imagens de treino (uma subpasta por classe)")
        if not pasta:
            return
        epocas = int(self.opt_epocas.get())
        self.barra.configure(mode="indeterminate")
        self.barra.start()

        def fim(acuracia):
            self.barra.stop()
            self.barra.configure(mode="determinate")
            self.barra.set(1)
            self._atualizar_modelo()
            self.status.set(f"Treino concluído. Acurácia na validação: {acuracia:.1%}")
            messagebox.showinfo("Treino concluído", f"Acurácia na validação: {acuracia:.1%}")

        self._em_segundo_plano(lambda: classificador.treinar(pasta, epocas), fim,
                               f"Treinando por {epocas} épocas… acompanhe no terminal.")

    def avaliar(self):
        pasta = filedialog.askdirectory(title="Pasta de teste (uma subpasta por classe)")
        if not pasta:
            return
        self.barra.configure(mode="determinate")
        self.barra.set(0)

        def progresso(feito, total):
            self.after(0, lambda: (self.barra.set(feito / total),
                                   self.status.set(f"Avaliando {feito}/{total} imagens…")))

        def tarefa():
            m = avaliacao.avaliar(pasta, progresso=progresso)
            m["arquivos"] = avaliacao.salvar_resultados(m)
            return m

        self._em_segundo_plano(tarefa, self._mostrar_metricas, "Carregando o modelo…")

    def _cor_valor(self, v):
        return VERDE if v >= 0.85 else ("#D97706", "#F59E0B") if v >= 0.6 else ("#DC2626", "#F87171")

    def _mostrar_metricas(self, m):
        self.metricas = m
        mc = m["macro"]
        for chave, v in (("acuracia", m["acuracia"]), ("precisao", mc["precisao"]),
                         ("recall", mc["recall"]), ("f1", mc["f1"])):
            self.kpi[chave].configure(text=f"{v:.1%}", text_color=self._cor_valor(v))

        t = self.frm_tabela
        self._limpar(t)
        cab = ("Classe", "Precisão", "Recall", "F1", "", "Imagens")
        for j, txt in enumerate(cab):
            t.grid_columnconfigure(j, weight=1 if j == 4 else 0)
            ctk.CTkLabel(t, text=txt, font=fonte(12, "bold"), text_color=SUAVE,
                         anchor="w" if j == 0 else "center").grid(row=0, column=j, sticky="ew",
                                                                 padx=4, pady=(0, 6))
        linhas = [(ROTULOS.get(c, c), m["por_classe"][c]) for c in m["classes"]]
        linhas.append(("Média (macro)", {**mc, "suporte": m["total"]}))
        for i, (nome, pc) in enumerate(linhas, start=1):
            negrito = "bold" if nome.startswith("Média") else "normal"
            ctk.CTkLabel(t, text=nome, font=fonte(13, negrito), text_color=TEXTO, anchor="w",
                         width=125).grid(row=i, column=0, sticky="w", padx=4, pady=4)
            for j, k in enumerate(("precisao", "recall", "f1"), start=1):
                ctk.CTkLabel(t, text=f"{pc[k]:.1%}", font=fonte(13, negrito), width=56,
                             text_color=self._cor_valor(pc[k]) if k == "f1" else TEXTO
                             ).grid(row=i, column=j, padx=4)
            b = ctk.CTkProgressBar(t, height=8, width=40, corner_radius=4, fg_color=CINZA_BARRA,
                                   progress_color=self._cor_valor(pc["f1"]))
            b.set(pc["f1"])
            b.grid(row=i, column=4, sticky="ew", padx=4)
            ctk.CTkLabel(t, text=str(pc["suporte"]), font=fonte(13, negrito), width=50,
                         text_color=TEXTO).grid(row=i, column=5, padx=4)

        self.txt_erros.delete("1.0", "end")
        erros = m.get("erros", [])
        if not erros:
            self.txt_erros.insert("1.0", "Nenhum erro. 🎉")
        for caminho, r, p in erros:
            self.txt_erros.insert("end", f"{caminho}\n    real: {r}  →  previsto: {p}\n")

        self._desenhar_matriz()
        txt, png = m["arquivos"]
        self.status.set("Avaliação concluída. Relatório salvo em relatorio_metricas.txt e matriz_confusao.png")

    def _desenhar_matriz(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        escuro = ctk.get_appearance_mode() == "Dark"
        fundo = CARTAO[1] if escuro else CARTAO[0]
        self.lbl_matriz_vazia.pack_forget()
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        self._canvas = FigureCanvasTkAgg(avaliacao.figura_matriz(self.metricas, escuro, fundo),
                                         master=self.frm_matriz)
        self._canvas.draw()
        w = self._canvas.get_tk_widget()
        w.configure(bg=fundo, highlightthickness=0)
        w.pack(fill="both", expand=True)


def iniciar():
    App().mainloop()


if __name__ == "__main__":
    iniciar()
