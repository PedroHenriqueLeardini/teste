#!/usr/bin/env python3
"""Janela principal da aplicacao de remocao de fundo."""

import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Front.backend_bridge import ResultadoProcessamento, backend_processar
from Front.ui_components import (
    AnimatedButton,
    COLOR,
    CollapsiblePanel,
    DropZone,
    HeroFrame,
    IconCache,
    PreviewPanel,
    ProgressBar,
    RADIUS,
    SPACING,
    SegmentedToggle,
    font_spec,
)


# ─── APP PRINCIPAL ───
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Remoção de Fundo / Objeto")
        self.geometry("720x820")
        self.minsize(600, 700)
        self.configure(fg_color=COLOR["bg"])

        # Grid simples: hero fixo (row=0), main scrollable preenche o resto (row=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.fila: queue.Queue = queue.Queue()
        self.thread_proc: Optional[threading.Thread] = None
        self.resultado: Optional[ResultadoProcessamento] = None

        self._criar_ui()
        self._poll_queue()

    def _criar_ui(self):
        # ── HERO (fixo no topo) ──
        self.hero = HeroFrame(self)
        self.hero.grid(row=0, column=0, sticky="ew")

        # ── MAIN SCROLLABLE (preenche espaço restante) ──
        main = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        main.grid(row=1, column=0, sticky="nsew", padx=SPACING["2xl"], pady=SPACING["md"])
        main.grid_columnconfigure(0, weight=1)

        # 1. Drop Zone
        self.dropzone = DropZone(main, on_change=self._on_arquivo, on_remove=self._on_remover)
        self.dropzone.grid(row=0, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 2. Toggle Modo
        self.toggle = SegmentedToggle(main, on_change=self._on_modo)
        self.toggle.grid(row=1, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 3. Painel Avançado
        self.painel = CollapsiblePanel(main)
        self.painel.grid(row=2, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 4. Botão Processar
        self.btn_proc = AnimatedButton(
            main, text="  Processar Imagem", font=font_spec("button_lg"),
            fg_color=COLOR["primary"], hover_color=COLOR["primary_hover"],
            text_color=COLOR["text_on_primary"], corner_radius=RADIUS["md"],
            height=52, state="disabled", command=self._iniciar,
            image=IconCache.get("sparkle", (20, 20), COLOR["text_on_primary"]),
            compound="left", anchor="center",
        )
        self.btn_proc.grid(row=3, column=0, sticky="ew", pady=(0, SPACING["xl"]))

        # 5. Preview Panel
        self.preview = PreviewPanel(main, height=420)
        self.preview.grid(row=4, column=0, sticky="ew", pady=(0, SPACING["xl"]))
        self.preview.grid_propagate(False)

        # 6. Botão Salvar (aparece só quando concluir)
        self.btn_salvar = AnimatedButton(
            main, text="  Salvar Resultado", font=font_spec("button_lg"),
            fg_color=COLOR["success"], hover_color=COLOR["success_hover"],
            text_color=COLOR["text_on_primary"], corner_radius=RADIUS["md"],
            height=48, state="disabled", command=self._salvar,
            image=IconCache.get("download", (20, 20), COLOR["text_on_primary"]),
            compound="left", anchor="center",
        )

        # ── PROGRESS BAR COMO OVERLAY (não no grid, posicionada via place()) ──
        self.progress = ProgressBar(self)

    # ── CALLBACKS ──
    def _reset_state(self, habilitar_processar: bool = False):
        """Reseta estado comum: limpa preview, progresso, resultado, btn_salvar.
        habilitar_processar=True quando arquivo selecionado; False quando removido."""
        self.btn_proc.configure(state="normal" if habilitar_processar else "disabled")
        self.btn_salvar.configure(state="disabled")
        self.btn_salvar.grid_remove()
        self.progress.resetar()
        self.resultado = None
        self.preview.resetar()

    def _on_arquivo(self, caminho: Path):
        self._reset_state(habilitar_processar=True)

    def _on_remover(self):
        self._reset_state(habilitar_processar=False)

    def _on_modo(self, modo: str):
        pass  # o backend recebe o modo no processamento

    def _set_controles(self, ativo: bool):
        estado = "normal" if ativo else "disabled"
        self.painel.btn_toggle.configure(state=estado)
        self.painel.definir_estado(estado)
        self.toggle.btn_fundo.configure(state=estado)
        self.toggle.btn_objeto.configure(state=estado)

    def _iniciar(self):
        if not self.dropzone.caminho_atual:
            return

        self.btn_proc.configure(state="disabled", text="  Processando...")
        self._set_controles(False)
        self.progress.resetar()
        self.progress.mostrar()

        qualidade, sensibilidade = self.painel.obter_valores()
        objeto, metodo = self.painel.obter_opcoes()
        modo = self.toggle.valor

        self.thread_proc = threading.Thread(
            target=self._worker,
            args=(str(self.dropzone.caminho_atual), modo, qualidade, sensibilidade, objeto, metodo),
            daemon=True,
        )
        self.thread_proc.start()

    def _worker(self, entrada, modo, qualidade, sensibilidade, objeto, metodo):
        try:
            def cb(pct, msg):
                self.fila.put(("prog", pct, msg))
            res = backend_processar(entrada, modo, qualidade, sensibilidade, cb,
                                    objeto=objeto, metodo=metodo)
            self.fila.put(("ok", res))
        except Exception as e:
            self.fila.put(("err", str(e)))

    def _poll_queue(self):
        try:
            while True:
                tipo, *dados = self.fila.get_nowait()
                try:
                    if tipo == "prog":
                        self.progress.atualizar(*dados)
                    elif tipo == "ok":
                        self._finalizar_ok(dados[0])
                    elif tipo == "err":
                        self._finalizar_err(dados[0])
                except Exception as e:
                    # Se algo falhar ao finalizar, o polling continua rodando
                    self._finalizar_err(str(e))
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _finalizar_comum(self):
        """Estado comum pós-processamento (sucesso ou erro)."""
        self.btn_proc.configure(state="normal", text="  Processar Imagem")
        self._set_controles(True)

    def _finalizar_ok(self, res: ResultadoProcessamento):
        self.resultado = res
        self.progress.sucesso(res.tempo_segundos, res.mensagem)
        self.preview.mostrar_resultado(res.imagem)
        self._finalizar_comum()
        self.btn_salvar.grid(row=5, column=0, sticky="ew", pady=(SPACING["md"], 0))
        self.btn_salvar.configure(state="normal")

    def _finalizar_err(self, err: str):
        self.progress.erro(err)
        self._finalizar_comum()
        messagebox.showerror("Erro no processamento", err)

    def _salvar(self):
        if not self.resultado:
            return
        caminho = filedialog.asksaveasfilename(
            title="Salvar resultado", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WEBP", "*.webp"), ("Todos", "*.*")],
            initialfile=f"{self.dropzone.caminho_atual.stem}_processado.png",
        )
        if caminho:
            try:
                img = self.resultado.imagem
                # JPEG não suporta transparência
                if caminho.lower().endswith((".jpg", ".jpeg")) and img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(caminho)
                messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho}")
            except Exception as e:
                messagebox.showerror("Erro ao salvar", str(e))


if __name__ == "__main__":
    App().mainloop()