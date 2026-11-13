import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_processing import (
    carregar_imagem,
    converter_rgb,
    converter_hsv,
    converter_lab,
    converter_grayscale,
    criar_imagem_transparente
)

from dataset_utils import listar_imagens, carregar_par

from segmentation import (
    criar_mascara_rgb,
    criar_mascara_hsv,
    criar_mascara_lab,
    refinar_mascara,
    abertura,
    fechamento,
    dilatacao,
    erosao,
    gradiente
)

from evaluation import avaliar_mascara

from fft_analysis import analisar_fft, comparar_visualizacoes_fft, visualizar_fase_fft

def plotar_histogramas(
    imagem_espaco,
    mascaras,
    nome_espaco,
    canais,
    limites
):
    """
    Plota os histogramas da imagem inteira e dos pixels
    selecionados por cada mÃ¡scara.

    imagem_espaco: imagem convertida para RGB, HSV ou LAB.
    mascaras: dicionÃ¡rio com as mÃ¡scaras geradas.
    nome_espaco: nome do espaÃ§o de cor.
    canais: nomes dos trÃªs canais.
    limites: nÃºmero de bins e faixa de valores por canal.
    """

    titulos = ["Imagem inteira"] + list(mascaras.keys())

    fig, eixos = plt.subplots(2, 3, figsize=(16, 9))
    eixos = eixos.flatten()

    cores = {
        "RGB": ["red", "green", "blue"],
        "HSV": ["orange", "green", "blue"],
        "LAB": ["black", "red", "gold"]
    }

    nomes_canais = cores[nome_espaco]

    for indice, titulo in enumerate(titulos):
        eixo = eixos[indice]

        # None significa que todos os pixels serÃ£o analisados.
        if indice == 0:
            mascara_selecao = None
        else:
            mascara = mascaras[titulo]
            mascara_selecao = np.where(
                mascara > 0, 255, 0
            ).astype(np.uint8)

        for canal in range(3):
            bins, valor_maximo = limites[canal]

            histograma = cv2.calcHist(
                [imagem_espaco],
                [canal],
                mascara_selecao,
                [bins],
                [0, valor_maximo]
            )

            # Normaliza para porcentagem dos pixels analisados.
            total = histograma.sum()

            if total > 0:
                histograma = histograma * (100.0 / total)

            eixo.plot(
                np.arange(bins),
                histograma.flatten(),
                color=nomes_canais[canal],
                label=canais[canal]
            )

        eixo.set_title(titulo)
        eixo.set_xlabel("Intensidade do canal")
        eixo.set_ylabel("Pixels (%)")
        eixo.set_xlim(0, max(limite[1] for limite in limites))
        eixo.grid(alpha=0.2)
        eixo.legend()

    fig.suptitle(
        f"Histogramas â€” EspaÃ§o {nome_espaco}",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    # Seleciona uma imagem do dataset
    imagens = listar_imagens()
    caminho = imagens[0]
    # Carrega a imagem e o trimap correspondente.
    # Use o mesmo par de imagem e trimap em todos os experimentos.
    imagem, trimap = carregar_par(caminho)

    # Carrega a imagem original em BGR
    imagem = carregar_imagem(caminho)

    imagem_rgb = converter_rgb(imagem)
    imagem_hsv = converter_hsv(imagem)
    imagem_lab = converter_lab(imagem)

    # Limites de exemplo: precisam ser ajustados para a imagem.
    mascara_rgb = criar_mascara_rgb(
        imagem,
        limite_inferior=(80, 30, 20),
        limite_superior=(255, 180, 160),
    )

    mascara_hsv = criar_mascara_hsv(
        imagem,
        limite_inferior=(0, 30, 30),
        limite_superior=(25, 255, 255),
    )

    mascara_lab = criar_mascara_lab(
        imagem,
        limite_inferior=(0, 120, 120),
        limite_superior=(255, 180, 180),
    )

    rgb_transparente = criar_imagem_transparente(imagem, mascara_rgb)
    hsv_transparente = criar_imagem_transparente(imagem, mascara_hsv)
    lab_transparente = criar_imagem_transparente(imagem, mascara_lab)

    rgb_resultado = cv2.cvtColor(rgb_transparente, cv2.COLOR_BGRA2RGBA)
    hsv_resultado = cv2.cvtColor(hsv_transparente, cv2.COLOR_BGRA2RGBA)
    lab_resultado = cv2.cvtColor(lab_transparente, cv2.COLOR_BGRA2RGBA)

    #refinamento
    mascara_rgb_aberta = abertura(mascara_rgb)
    mascara_hsv_aberta = abertura(mascara_hsv)
    mascara_lab_aberta = abertura(mascara_lab)

    mascara_rgb_fechada = fechamento(mascara_rgb)
    mascara_hsv_fechada = fechamento(mascara_hsv)
    mascara_lab_fechada = fechamento(mascara_lab)

    mascara_rgb_dilatada = dilatacao(mascara_rgb)
    mascara_hsv_dilatada = dilatacao(mascara_hsv)
    mascara_lab_dilatada = dilatacao(mascara_lab)

    mascara_rgb_erosionada = erosao(mascara_rgb)
    mascara_hsv_erosionada = erosao(mascara_hsv)
    mascara_lab_erosionada = erosao(mascara_lab)

    rgb_aberto_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_rgb_aberta), cv2.COLOR_BGRA2RGBA)
    hsv_aberto_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_hsv_aberta), cv2.COLOR_BGRA2RGBA)
    lab_aberto_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_lab_aberta), cv2.COLOR_BGRA2RGBA)

    rgb_fechado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_rgb_fechada), cv2.COLOR_BGRA2RGBA)
    hsv_fechado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_hsv_fechada), cv2.COLOR_BGRA2RGBA)   
    lab_fechado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_lab_fechada), cv2.COLOR_BGRA2RGBA)

    rgb_dilatado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_rgb_dilatada), cv2.COLOR_BGRA2RGBA)
    hsv_dilatado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_hsv_dilatada), cv2.COLOR_BGRA2RGBA)
    lab_dilatado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_lab_dilatada), cv2.COLOR_BGRA2RGBA)

    rgb_erosionado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_rgb_erosionada), cv2.COLOR_BGRA2RGBA)
    hsv_erosionado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_hsv_erosionada), cv2.COLOR_BGRA2RGBA)
    lab_erosionado_resultado = cv2.cvtColor(criar_imagem_transparente(imagem, mascara_lab_erosionada), cv2.COLOR_BGRA2RGBA)

    #guarda as mÃ¡scaras refinadas em dicionarios separados
    mascaras_rgb = {
        "Original": mascara_rgb,
        "Abertura": mascara_rgb_aberta,
        "Fechamento": mascara_rgb_fechada,
        "DilataÃ§Ã£o": mascara_rgb_dilatada,
        "ErosÃ£o": mascara_rgb_erosionada,
    }

    mascaras_hsv = {
        "Original": mascara_hsv,
        "Abertura": mascara_hsv_aberta,
        "Fechamento": mascara_hsv_fechada,
        "DilataÃ§Ã£o": mascara_hsv_dilatada,
        "ErosÃ£o": mascara_hsv_erosionada,
    }

    mascaras_lab = {
        "Original": mascara_lab,
        "Abertura": mascara_lab_aberta,
        "Fechamento": mascara_lab_fechada,
        "DilataÃ§Ã£o": mascara_lab_dilatada,
        "ErosÃ£o": mascara_lab_erosionada,
    }


    # Histograma RGB
    plotar_histogramas(
        imagem_rgb,
        mascaras_rgb,
        "RGB",
        canais=["R", "G", "B"],
        limites=[(256, 256)] * 3
    )

    # Histograma HSV
    # No OpenCV, o canal H varia de 0 a 179.
    plotar_histogramas(
        imagem_hsv,
        mascaras_hsv,
        "HSV",
        canais=["H (Matiz)", "S (SaturaÃ§Ã£o)", "V (Valor)"],
        limites=[
            (180, 180),
            (256, 256),
            (256, 256)
        ]
    )

    # Histograma LAB
    # No OpenCV, os canais a e b sÃ£o codificados de 0 a 255.
    plotar_histogramas(
        imagem_lab,
        mascaras_lab,
        "LAB",
        canais=["L*", "a*", "b*"],
        limites=[(256, 256)] * 3
    )

    fig, eixos = plt.subplots(3, 6, figsize=(14, 12))

    eixos[0, 0].imshow(converter_rgb(imagem))
    eixos[0, 0].set_title("Imagem original")

    mascaras = [  
        # Imagem original, convertida para RGB apenas para exibiÃ§Ã£o
        (eixos[0, 0], imagem_rgb, "Imagem original (RGB)"),
        # EspaÃ§o HSV: exibido como uma imagem HSV convertida para RGb
        (eixos[1, 0], imagem_hsv, "EspaÃ§o HSV"),
        # EspaÃ§o LAB: exibido como uma imagem LAB convertida para RGb
        (eixos[2, 0], imagem_lab, "EspaÃ§o LAB"),
        # EspaÃ§o gray: exibido como uma imagem em tons de cinza
        (eixos[0, 1], mascara_rgb, "MÃ¡scara RGB"),
        (eixos[1, 1], mascara_hsv, "MÃ¡scara HSV"),
        (eixos[2, 1], mascara_lab, "MÃ¡scara LAB"),
        (eixos[0, 2], mascara_rgb_aberta, "MÃ¡scara RGB Aberta"),
        (eixos[1, 2], mascara_hsv_aberta, "MÃ¡scara HSV Aberta"),
        (eixos[2, 2], mascara_lab_aberta, "MÃ¡scara LAB Aberta"),
        (eixos[0, 3], mascara_rgb_fechada, "MÃ¡scara RGB Fechada"),
        (eixos[1, 3], mascara_hsv_fechada, "MÃ¡scara HSV Fechada"),
        (eixos[2, 3], mascara_lab_fechada, "MÃ¡scara LAB Fechada"),
        (eixos[0, 4], mascara_rgb_dilatada, "MÃ¡scara RGB Dilatada"),
        (eixos[1, 4], mascara_hsv_dilatada, "MÃ¡scara HSV Dilatada"),
        (eixos[2, 4], mascara_lab_dilatada, "MÃ¡scara LAB Dilatada"),
        (eixos[0, 5], mascara_rgb_erosionada, "MÃ¡scara RGB Erosionada"),
        (eixos[1, 5], mascara_hsv_erosionada, "MÃ¡scara HSV Erosionada"),
        (eixos[2, 5], mascara_lab_erosionada, "MÃ¡scara LAB Erosionada"),

    ]
    for eixo, mascara, titulo in mascaras:
        eixo.imshow(mascara, cmap="gray", vmin=0, vmax=255)
        eixo.set_title(titulo)

    for eixo in eixos.flat:
        eixo.axis("off")

    fig.suptitle(f"MÃ¡scaras por espaÃ§o de cores â€” {caminho.name}")


    fig, eixos = plt.subplots(3, 6, figsize=(14, 12))
    resultados = [  
        # Imagem original, convertida para RGB apenas para exibiÃ§Ã£o
        (eixos[0, 0], imagem_rgb, "Imagem original (RGB)"),
        # EspaÃ§o HSV: exibido como uma imagem HSV convertida para RGb
        (eixos[1, 0], imagem_hsv, "EspaÃ§o HSV"),
        # EspaÃ§o LAB: exibido como uma imagem LAB convertida para RGb
        (eixos[2, 0], imagem_lab, "EspaÃ§o LAB"),
        # EspaÃ§o gray: exibido como uma imagem em tons de cinza
        (eixos[0, 1], rgb_resultado, "MÃ¡scara RGB"),
        (eixos[1, 1], hsv_resultado, "MÃ¡scara HSV"),
        (eixos[2, 1], lab_resultado, "MÃ¡scara LAB"),
        (eixos[0, 2], rgb_aberto_resultado, "MÃ¡scara RGB Aberta"),
        (eixos[1, 2], hsv_aberto_resultado, "MÃ¡scara HSV Aberta"),
        (eixos[2, 2], lab_aberto_resultado, "MÃ¡scara LAB Aberta"),
        (eixos[0, 3], rgb_fechado_resultado, "MÃ¡scara RGB Fechada"),
        (eixos[1, 3], hsv_fechado_resultado, "MÃ¡scara HSV Fechada"),
        (eixos[2, 3], lab_fechado_resultado, "MÃ¡scara LAB Fechada"),
        (eixos[0, 4], rgb_dilatado_resultado, "MÃ¡scara RGB Dilatada"),
        (eixos[1, 4], hsv_dilatado_resultado, "MÃ¡scara HSV Dilatada"),
        (eixos[2, 4], lab_dilatado_resultado, "MÃ¡scara LAB Dilatada"),
        (eixos[0, 5], rgb_erosionado_resultado, "MÃ¡scara RGB Erosionada"),
        (eixos[1, 5], hsv_erosionado_resultado, "MÃ¡scara HSV Erosionada"),
        (eixos[2, 5], lab_erosionado_resultado, "MÃ¡scara LAB Erosionada"),

    ]

    for eixo, mascara, titulo in resultados:
        eixo.imshow(mascara, cmap="gray", vmin=0, vmax=255)
        eixo.set_title(titulo)
    eixos[1, 2].axis("off")


    for eixo in eixos.flat:
        eixo.axis("off")

    fig.suptitle(f"MÃ¡scaras por espaÃ§o de cores â€” {caminho.name}")

    # Gera as mÃ¡scaras de referÃªncia para a avaliaÃ§Ã£o.
    # Caso as mÃ¡scaras jÃ¡ tenham sido geradas com esta mesma imagem,
    # nÃ£o Ã© necessÃ¡rio recriÃ¡-las.


    # ==========================================
    # AVALIAÃ‡ÃƒO DE TODAS AS MÃSCARAS
    # ==========================================

    # Cada dicionÃ¡rio deve conter:
    # Original, Abertura, Fechamento, DilataÃ§Ã£o e ErosÃ£o.

    mascaras_por_espaco = {
        "RGB": mascaras_rgb,
        "HSV": mascaras_hsv,
        "LAB": mascaras_lab,
    }

    metricas_tradicionais = {
        "iou": "IoU",
        "precisao": "PrecisÃ£o",
        "recall": "Recall",
        "f1_score": "F1-score",
        "acuracia": "AcurÃ¡cia",
    }

    metricas_incerteza = {
        "pixels_regiao_incerta": "Pixels na regiÃ£o incerta",
        "pixels_previstos_na_incerteza":
            "Pixels previstos como animal na incerteza",
        "proporcao_mascara_na_incerteza":
            "ProporÃ§Ã£o da mÃ¡scara na incerteza",
        "proporcao_incerteza_como_animal":
            "RegiÃ£o incerta classificada como animal",
        "proporcao_incerteza_como_fundo":
            "RegiÃ£o incerta classificada como fundo",
    }

    # Avalia cada mÃ¡scara.
    resultados_avaliacao = {}

    for espaco, mascaras in mascaras_por_espaco.items():
        for nome_mascara, mascara in mascaras.items():

            identificador = f"{espaco} - {nome_mascara}"

            resultados_avaliacao[identificador] = avaliar_mascara(
                mascara,
                trimap
            )

    # Imprime todas as mÃ©tricas no terminal.
    for identificador, resultado in resultados_avaliacao.items():
        print(f"\n{'=' * 60}")
        print(identificador)
        print(f"{'=' * 60}")

        print("\nMÃ©tricas tradicionais:")
        for chave, nome in metricas_tradicionais.items():
            print(f"  {nome}: {resultado[chave]:.4f}")

        print("\nMÃ©tricas da regiÃ£o de incerteza:")
        for chave, nome in metricas_incerteza.items():
            valor = resultado[chave]

            if chave.startswith("proporcao_"):
                print(f"  {nome}: {valor:.4f} ({valor * 100:.2f}%)")
            else:
                print(f"  {nome}: {valor}")

    # RÃ³tulos das 15 mÃ¡scaras avaliadas.
    rotulos = list(resultados_avaliacao.keys())
    posicoes = np.arange(len(rotulos))


    # ==========================================
    # GRÃFICO 1: MÃ‰TRICAS TRADICIONAIS
    # ==========================================

    fig, eixos = plt.subplots(
        3, 2,
        figsize=(18, 14)
    )

    eixos = eixos.flatten()

    for indice, (chave, nome) in enumerate(
        metricas_tradicionais.items()
    ):
        eixo = eixos[indice]

        valores = [
            resultados_avaliacao[rotulo][chave]
            for rotulo in rotulos
        ]

        barras = eixo.bar(posicoes, valores)

        eixo.set_title(nome)
        eixo.set_ylabel("Valor")
        eixo.set_ylim(0, 1.05)
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels(
            rotulos,
            rotation=60,
            ha="right",
            fontsize=8
        )
        eixo.grid(axis="y", linestyle="--", alpha=0.3)

    # Remove o sexto painel, que nÃ£o serÃ¡ utilizado.
    fig.delaxes(eixos[5])

    fig.suptitle(
        "ComparaÃ§Ã£o das mÃ©tricas tradicionais por mÃ¡scara",
        fontsize=16
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])


    # ==========================================
    # GRÃFICO 2: MÃ‰TRICAS DA REGIÃƒO DE INCERTEZA
    # ==========================================

    fig, eixos = plt.subplots(
        3, 2,
        figsize=(18, 15)
    )

    eixos = eixos.flatten()

    for indice, (chave, nome) in enumerate(
        metricas_incerteza.items()
    ):
        eixo = eixos[indice]

        valores = [
            resultados_avaliacao[rotulo][chave]
            for rotulo in rotulos
        ]

        barras = eixo.bar(posicoes, valores)

        eixo.set_title(nome)
        eixo.set_xticks(posicoes)
        eixo.set_xticklabels(
            rotulos,
            rotation=60,
            ha="right",
            fontsize=8
        )
        eixo.grid(axis="y", linestyle="--", alpha=0.3)

        if chave.startswith("proporcao_"):
            eixo.set_ylabel("ProporÃ§Ã£o")
            eixo.set_ylim(0, 1.05)
        else:
            eixo.set_ylabel("Quantidade de pixels")

        # Mostra os valores nas barras quando a quantidade
        # de rÃ³tulos permite uma leitura razoÃ¡vel.
        if chave in (
            "pixels_regiao_incerta",
            "pixels_previstos_na_incerteza"
        ):
            eixo.bar_label(
                barras,
                fmt="%.0f",
                padding=2,
                fontsize=7
            )
        else:
            eixo.bar_label(
                barras,
                fmt="%.2f",
                padding=2,
                fontsize=7
            )

    # Remove o sexto painel nÃ£o utilizado.
    fig.delaxes(eixos[5])

    fig.suptitle(
        "ComparaÃ§Ã£o das mÃ©tricas da regiÃ£o de incerteza",
        fontsize=16
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.tight_layout()
    plt.show()

    analisar_fft(imagem)

    comparar_visualizacoes_fft(imagem)

    # imagem Ã© uma imagem colorida no formato BGR
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Calcula a FFT bidimensional
    fft = np.fft.fft2(cinza)

    # Centraliza as frequÃªncias
    fft_centralizada = np.fft.fftshift(fft)

    # ObtÃ©m a fase em radianos
    fase = np.angle(fft_centralizada)

    # Exibe o espectro de fase
    plt.figure(figsize=(7, 6))
    plt.imshow(
        fase,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi
    )
    plt.colorbar(label="Fase (radianos)")
    plt.title("Espectro de fase da imagem")
    plt.axis("off")
    plt.show()

    resultado = visualizar_fase_fft(imagem)
    print(resultado)


if __name__ == '__main__':
    main()

