"""
Interface gráfica (Tkinter) do sistema de diagnóstico de folhas de tomate.

Aba "Diagnóstico": abre uma foto, mostra a imagem, as probabilidades da IA,
                   pergunta os sintomas e exibe o diagnóstico do sistema especialista.
Aba "Métricas":    treina a IA e avalia uma pasta de teste (acurácia, precisão,
                   recall, F1 e matriz de confusão).
"""
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import avaliacao
import classificador
from sistema_especialista import NOMES, PERGUNTAS, MotorInferencia

CORES = {"confirmado": "#1b7a2f", "provavel": "#b36b00", "inconclusivo": "#b00020"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Diagnóstico de Doenças em Folhas de Tomate — IA + Sistema Especialista")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.resultado_ia = None
        self.respostas = {}
        self._foto_tk = None
        self._canvas_matriz = None

        abas = ttk.Notebook(self)
        abas.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.aba_diag = ttk.Frame(abas, padding=10)
        self.aba_met = ttk.Frame(abas, padding=10)
        abas.add(self.aba_diag, text="  Diagnóstico  ")
        abas.add(self.aba_met, text="  Métricas  ")

        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status, anchor="w", padding=(8, 4)).pack(fill="x")

        self._montar_diagnostico()
        self._montar_metricas()
        self._atualizar_status_modelo()

    # ------------------------------------------------------------------ util
    def _atualizar_status_modelo(self):
        if classificador.modelo_existe():
            self.status.set("Modelo carregado do arquivo " + classificador.ARQ_MODELO)
        else:
            self.status.set("Nenhum modelo treinado ainda: vá na aba Métricas > Treinar IA.")

    def _em_segundo_plano(self, tarefa, ao_terminar, msg):
        """Roda uma tarefa demorada sem travar a janela."""
        self.status.set(msg)
        self.config(cursor="watch")

        def executar():
            try:
                resultado = tarefa()
                self.after(0, lambda: self._fim_tarefa(ao_terminar, resultado, None))
            except Exception as e:  # noqa: BLE001
                self.after(0, lambda e=e: self._fim_tarefa(ao_terminar, None, e))

        threading.Thread(target=executar, daemon=True).start()

    def _fim_tarefa(self, ao_terminar, resultado, erro):
        self.config(cursor="")
        if erro:
            self.status.set("Erro: " + str(erro))
            messagebox.showerror("Erro", str(erro))
        else:
            ao_terminar(resultado)

    # ----------------------------------------------------------- diagnóstico
    def _montar_diagnostico(self):
        esq = ttk.Frame(self.aba_diag)
        esq.pack(side="left", fill="y", padx=(0, 12))
        dir_ = ttk.Frame(self.aba_diag)
        dir_.pack(side="left", fill="both", expand=True)

        ttk.Button(esq, text="📂  Abrir imagem da folha...", command=self.abrir_imagem).pack(fill="x")
        self.lbl_imagem = tk.Label(esq, text="Nenhuma imagem\nselecionada", width=46, height=20,
                                   bg="#f2f2f2", relief="groove")
        self.lbl_imagem.pack(pady=10)
        self.lbl_arquivo = ttk.Label(esq, text="", wraplength=340, foreground="#555")
        self.lbl_arquivo.pack()

        self.frm_ia = ttk.LabelFrame(dir_, text="1. Resultado da IA (classificação da imagem)", padding=10)
        self.frm_ia.pack(fill="x")
        ttk.Label(self.frm_ia, text="Abra uma imagem para classificar.").pack(anchor="w")

        self.frm_sint = ttk.LabelFrame(dir_, text="2. Sintomas observados (sistema especialista)", padding=10)
        self.frm_sint.pack(fill="x", pady=8)
        ttk.Label(self.frm_sint, text="As perguntas aparecem depois da classificação.").pack(anchor="w")

        self.btn_diag = ttk.Button(dir_, text="🔎  Diagnosticar", command=self.diagnosticar,
                                   state="disabled")
        self.btn_diag.pack(anchor="w")

        frm_res = ttk.LabelFrame(dir_, text="3. Diagnóstico", padding=6)
        frm_res.pack(fill="both", expand=True, pady=(8, 0))
        self.txt_res = tk.Text(frm_res, wrap="word", height=12, font=("Segoe UI", 10),
                               relief="flat", padx=6, pady=6)
        self.txt_res.pack(fill="both", expand=True)
        self.txt_res.tag_configure("titulo", font=("Segoe UI", 13, "bold"))
        self.txt_res.tag_configure("negrito", font=("Segoe UI", 10, "bold"))
        for certeza, cor in CORES.items():
            self.txt_res.tag_configure(certeza, foreground=cor, font=("Segoe UI", 13, "bold"))
        self.txt_res.configure(state="disabled")

    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Escolha a foto da folha",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")])
        if not caminho:
            return
        img = Image.open(caminho)
        img.thumbnail((340, 340))
        self._foto_tk = ImageTk.PhotoImage(img)
        self.lbl_imagem.configure(image=self._foto_tk, text="", width=340, height=340)
        self.lbl_arquivo.configure(text=caminho)
        self.btn_diag.configure(state="disabled")
        self._escrever_resultado([])
        self._em_segundo_plano(lambda: classificador.classificar(caminho),
                               self._mostrar_resultado_ia, "Classificando a imagem...")

    def _mostrar_resultado_ia(self, resultado):
        self.resultado_ia = resultado
        classe, conf, probs = resultado

        for w in self.frm_ia.winfo_children():
            w.destroy()
        for c, p in sorted(probs.items(), key=lambda x: -x[1]):
            linha = ttk.Frame(self.frm_ia)
            linha.pack(fill="x", pady=2)
            fonte = ("Segoe UI", 10, "bold") if c == classe else ("Segoe UI", 10)
            ttk.Label(linha, text=c, width=20, font=fonte).pack(side="left")
            barra = ttk.Progressbar(linha, maximum=100, value=p * 100, length=300)
            barra.pack(side="left", padx=6)
            ttk.Label(linha, text=f"{p:.1%}", width=7, font=fonte).pack(side="left")

        for w in self.frm_sint.winfo_children():
            w.destroy()
        self.respostas = {}
        perguntas = PERGUNTAS.get(classe, [])
        if not perguntas:
            ttk.Label(self.frm_sint, text="Nenhuma pergunta necessária para esta classe.").pack(anchor="w")
        for fato, texto in perguntas:
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.frm_sint, text=texto, variable=var).pack(anchor="w", pady=1)
            self.respostas[fato] = var

        self.btn_diag.configure(state="normal")
        self.status.set(f"IA: {classe} ({conf:.1%}). Marque os sintomas e clique em Diagnosticar.")

    def diagnosticar(self):
        if not self.resultado_ia:
            return
        classe, conf, _ = self.resultado_ia
        fatos = {"classe_ia": classe, "confianca": conf}
        fatos.update({f: v.get() for f, v in self.respostas.items()})
        res, regras = MotorInferencia().inferir(fatos)

        partes = [
            (NOMES.get(res["doenca"], res["doenca"]) + "\n", "titulo"),
            (f"Certeza: {res.get('certeza', '-')}\n\n", res.get("certeza", "negrito")),
            ("Recomendação: ", "negrito"), (res.get("recomendacao", "-") + "\n\n", None),
        ]
        if "alerta" in res:
            partes += [("⚠ Alerta: ", "negrito"), (res["alerta"] + "\n\n", None)]
        partes.append(("Por que cheguei a essa conclusão:\n", "negrito"))
        for r in regras:
            partes.append((f"  [{r.nome}] {r.porque}\n", None))
        self._escrever_resultado(partes)
        self.status.set("Diagnóstico concluído.")

    def _escrever_resultado(self, partes):
        self.txt_res.configure(state="normal")
        self.txt_res.delete("1.0", "end")
        for texto, tag in partes:
            self.txt_res.insert("end", texto, tag) if tag else self.txt_res.insert("end", texto)
        self.txt_res.configure(state="disabled")

    # -------------------------------------------------------------- métricas
    def _montar_metricas(self):
        topo = ttk.Frame(self.aba_met)
        topo.pack(fill="x")
        ttk.Button(topo, text="🧠  Treinar IA...", command=self.treinar).pack(side="left")
        ttk.Label(topo, text="Épocas:").pack(side="left", padx=(12, 2))
        self.var_epocas = tk.IntVar(value=5)
        ttk.Spinbox(topo, from_=1, to=50, width=4, textvariable=self.var_epocas).pack(side="left")
        ttk.Separator(topo, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Button(topo, text="📊  Avaliar pasta de teste...", command=self.avaliar).pack(side="left")
        self.barra = ttk.Progressbar(topo, length=250, maximum=100)
        self.barra.pack(side="left", padx=12)

        self.lbl_resumo = ttk.Label(self.aba_met, text="", font=("Segoe UI", 12, "bold"))
        self.lbl_resumo.pack(anchor="w", pady=(10, 4))

        corpo = ttk.Frame(self.aba_met)
        corpo.pack(fill="both", expand=True)
        # a matriz é empacotada primeiro, à direita, para nunca ser espremida
        self.frm_matriz = ttk.LabelFrame(corpo, text="Matriz de confusão", padding=4)
        self.frm_matriz.pack(side="right", fill="y", padx=(10, 0))
        self.lbl_matriz_vazia = ttk.Label(self.frm_matriz, width=52, anchor="center",
                                          text="Avalie uma pasta de teste\npara ver a matriz.")
        self.lbl_matriz_vazia.pack(expand=True)
        esq = ttk.Frame(corpo)
        esq.pack(side="left", fill="both", expand=True)

        colunas = ("classe", "precisao", "recall", "f1", "suporte")
        self.tabela = ttk.Treeview(esq, columns=colunas, show="headings", height=7)
        for col, titulo, larg in zip(colunas, ("Classe", "Precisão", "Recall", "F1-score", "Suporte"),
                                     (140, 70, 70, 70, 60)):
            self.tabela.heading(col, text=titulo)
            self.tabela.column(col, width=larg, anchor="w" if col == "classe" else "center")
        self.tabela.pack(fill="x")

        ttk.Label(esq, text="Relatório completo:").pack(anchor="w", pady=(10, 2))
        self.txt_rel = tk.Text(esq, wrap="none", font=("Consolas", 9), height=14, width=50)
        self.txt_rel.pack(fill="both", expand=True)

        ttk.Label(esq, foreground="#555", wraplength=480, text=(
            "Precisão: das imagens que a IA disse ser da classe, quantas eram mesmo dela.  "
            "Recall: das imagens da classe, quantas a IA acertou.  "
            "F1: equilíbrio entre precisão e recall.  Suporte: nº de imagens da classe.")
        ).pack(anchor="w", pady=(6, 0))

    def treinar(self):
        pasta = filedialog.askdirectory(title="Pasta com as imagens de treino (uma subpasta por classe)")
        if not pasta:
            return
        epocas = self.var_epocas.get()
        self.barra.configure(mode="indeterminate")
        self.barra.start(12)

        def fim(acuracia):
            self.barra.stop()
            self.barra.configure(mode="determinate", value=100)
            self.status.set(f"Treino concluído. Acurácia na validação: {acuracia:.1%}")
            messagebox.showinfo("Treino concluído", f"Acurácia na validação: {acuracia:.1%}\n"
                                "Modelo salvo em " + classificador.ARQ_MODELO)

        self._em_segundo_plano(lambda: classificador.treinar(pasta, epocas), fim,
                               f"Treinando por {epocas} épocas... acompanhe o progresso no terminal.")

    def avaliar(self):
        pasta = filedialog.askdirectory(title="Pasta de teste (uma subpasta por classe)")
        if not pasta:
            return
        self.barra.configure(mode="determinate", value=0)

        def progresso(feito, total):
            self.after(0, lambda: (self.barra.configure(value=100 * feito / total),
                                   self.status.set(f"Avaliando: {feito}/{total} imagens...")))

        def tarefa():
            m = avaliacao.avaliar(pasta, progresso=progresso)
            m["arquivos"] = avaliacao.salvar_resultados(m)
            return m

        self._em_segundo_plano(tarefa, self._mostrar_metricas, "Carregando o modelo...")

    def _mostrar_metricas(self, m):
        mc = m["macro"]
        self.lbl_resumo.configure(text=(
            f"Acurácia: {m['acuracia']:.2%}   ({m['acertos']}/{m['total']} imagens)      "
            f"F1 médio: {mc['f1']:.2%}"))

        self.tabela.delete(*self.tabela.get_children())
        for c in m["classes"]:
            pc = m["por_classe"][c]
            self.tabela.insert("", "end", values=(c, f"{pc['precisao']:.1%}", f"{pc['recall']:.1%}",
                                                  f"{pc['f1']:.1%}", pc["suporte"]))
        self.tabela.insert("", "end", values=("Média (macro)", f"{mc['precisao']:.1%}",
                                              f"{mc['recall']:.1%}", f"{mc['f1']:.1%}", m["total"]))

        self.txt_rel.delete("1.0", "end")
        self.txt_rel.insert("1.0", avaliacao.relatorio_texto(m))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.lbl_matriz_vazia.pack_forget()
        if self._canvas_matriz:
            self._canvas_matriz.get_tk_widget().destroy()
        self._canvas_matriz = FigureCanvasTkAgg(avaliacao.figura_matriz(m), master=self.frm_matriz)
        self._canvas_matriz.draw()
        self._canvas_matriz.get_tk_widget().pack(fill="both", expand=True)

        txt, png = m["arquivos"]
        self.status.set(f"Avaliação concluída. Resultados salvos em {txt} e {png}")


def iniciar():
    App().mainloop()


if __name__ == "__main__":
    iniciar()
