import cv2
import numpy as np

# Funções de filtragem de frequência no domínio da frequência (somente análise para este módulo)
# As funções de filtragem estão implementadas nesta seção.
# Funções de filtragem no domínio da frequência (somente análise)
# A implementação utiliza o OpenCV para FFT e NumPy para manipulacao da grade de frequencia
# A filtragem de imagens é feita exclusivamente para imagens em escala de cinza para simplificar a representacao dos canais.
# Imagens coloridas sao convertidas para escala de cinza e retornadas no mesmo formato de tipo de entrada.
# Os cortes de frequencia sao normalizados em ciclos por pixel (0, 0.5], onde 0.5 e a frequencia de Nyquist.


def _validar_imagem(imagem):
	# Garante que as funções de conversão recebam dados de imagem utilizáveis.
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("A imagem deve ser um array NumPy não vazio.")
	# As funções deste módulo esperam uma imagem colorida BGR com 3 canais.
	if imagem.ndim != 3 or imagem.shape[2] != 3:
		raise ValueError("A imagem deve estar no formato BGR com três canais.")


def _normalizar_mascara(mascara):
	# Confirma que existe uma máscara válida antes de processá-la.
	if not isinstance(mascara, np.ndarray) or mascara.size == 0:
		raise ValueError("A máscara deve ser um array NumPy não vazio.")

	# Converte máscaras coloridas ou com canal único para uma matriz 2D.
	if mascara.ndim == 3:
		if mascara.shape[2] == 3:
			mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
		elif mascara.shape[2] == 4:
			mascara = cv2.cvtColor(mascara, cv2.COLOR_BGRA2GRAY)
		elif mascara.shape[2] == 1:
			mascara = mascara[:, :, 0]
		else:
			raise ValueError("A máscara deve ter um, três ou quatro canais.")
	elif mascara.ndim != 2:
		raise ValueError("A máscara deve ser bidimensional ou ter canais de cor.")

	# Transforma qualquer valor positivo em branco (selecionado) e zero
	# nos demais pixels (não selecionados); o tipo uint8 é exigido pelo OpenCV.
	return np.where(mascara > 0, 255, 0).astype(np.uint8)


def carregar_imagem(caminho):
	# Abre o arquivo como imagem colorida no formato BGR padrão do OpenCV.
	imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
	if imagem is None:
		raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")
	return imagem


def converter_rgb(imagem):
	_validar_imagem(imagem)
	# Troca a ordem dos canais BGR para RGB, usada por várias bibliotecas de visualização.
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def converter_hsv(imagem):
	_validar_imagem(imagem)
	# Converte para matiz, saturação e brilho, componentes úteis para analisar cores.
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)


def converter_lab(imagem):
	_validar_imagem(imagem)
	# Converte para o espaço perceptual Lab, que separa luminosidade de informação de cor.
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)


def converter_grayscale(imagem):
	_validar_imagem(imagem)
	# Reduz a imagem colorida a um canal de intensidade (tons de cinza).
	return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def aplicar_mascara(imagem, mascara):
	_validar_imagem(imagem)
	# Padroniza a máscara para valores binários aceitos pelo OpenCV.
	mascara = _normalizar_mascara(mascara)
	# Exige alinhamento pixel a pixel para não aplicar a máscara incorretamente.
	if mascara.shape != imagem.shape[:2]:
		raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")
	# Mantém os pixels brancos na máscara e zera os pixels fora da região selecionada.
	return cv2.bitwise_and(imagem, imagem, mask=mascara)


def criar_alpha(mascara):
	# Usa a máscara binária como canal alfa: 255 opaco e 0 transparente.
	return _normalizar_mascara(mascara)


def criar_imagem_transparente(imagem, mascara):
	_validar_imagem(imagem)
	# Normaliza a máscara e verifica que cada pixel corresponde à imagem.
	alpha = criar_alpha(mascara)
	if alpha.shape != imagem.shape[:2]:
		raise ValueError("A máscara deve ter as mesmas dimensões da imagem.")

	# Adiciona o quarto canal e o preenche com os níveis de transparência da máscara.
	imagem_bgra = cv2.cvtColor(imagem, cv2.COLOR_BGR2BGRA)
	imagem_bgra[:, :, 3] = alpha
	return imagem_bgra


def salvar_imagem(caminho, imagem):
	# Evita tentar gravar dados ausentes ou que não estejam em formato de array.
	if not isinstance(imagem, np.ndarray) or imagem.size == 0:
		raise ValueError("A imagem a salvar deve ser um array NumPy não vazio.")
	# cv2.imwrite retorna False quando não consegue gravar no caminho informado.
	if not cv2.imwrite(str(caminho), imagem):
		raise OSError(f"Não foi possível salvar a imagem: {caminho}")

# ---------------------------------------------------------------------
# Funções de filtragem de frequência no domínio da frequência (LPF/HPF)
# ---------------------------------------------------------------------

def _get_freq_grid(shape):
    """
    Retorna a grade de frequência (ciclos/pixel) para uma imagem quadrada de shape (h, w).
    Retorna duas matrizes (fx, fy) alinhadas com a FFT do OpenCV, onde cada valor
    é a frequência normalizada com Nyquist (máximo = 0.5).
    """
    h, w = shape
    # Frequências em ciclos por pixel, na ordem do OpenCV (não deslocada).
    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    # meshgrid com indexing='xy' para corresponder às coordenadas da imagem (colunas, linhas)
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing="xy")
    return fx_grid, fy_grid

def aplicar_lpf(imagem, frequencia_corte):
    """
    Aplica um filtro de passa-baixa circular (LPF) à imagem no domínio da frequência.

    Parâmetros:
        imagem: array NumPy (BGR ou escala de cinza).
        frequencia_corte: frequência de corte em ciclos/pixel (0, 0.5].

    Retorna:
        imagem filtrada (mesmo dtype de entrada, mesma forma).
    """
    if not (0 < frequencia_corte <= 0.5):
        raise ValueError("frequencia_corte deve estar no intervalo (0, 0.5] ciclos/pixel.")

    _validar_imagem(imagem)
    # Se for colorida (3 canais), converter para escala de cinza para filtragem
    se_escala_de_cinza = imagem.ndim == 2
    se_bgr = imagem.ndim == 3 and imagem.shape[2] == 3
    if not (se_escala_de_cinza or se_bgr):
        raise ValueError("A imagem deve ser BGR (3 canais) ou escala de cinza.")

    if se_bgr:
        cinza = converter_grayscale(imagem)
        formato_original = imagem
    else:
        cinza = imagem.copy()
        formato_original = None

    # Converter para float64 para FFT
    cinza_f = cinza.astype(np.float64)
    # FFT do OpenCV espera float32, mas NumPy funciona igualmente; usar cv2.dft para centralização.
    dft = cv2.dft(cinza_f)
    # Centralizar para facilitar a criação do filtro
    dft_shift = np.fft.fftshift(dft)
    # Grade de frequência (centralizada para corresponder à DFT deslocada)
    fx, fy = _get_freq_grid(cinza_f.shape)
    fx, fy = np.fft.fftshift(fx), np.fft.fftshift(fy)
    # Calcular a magnitude da frequência radial em ciclos/pixel
    freq_mag = np.hypot(fx, fy)
    # Máscara do filtro: 1 onde freq_mag <= corte, 0 caso contrário
    filtro = (freq_mag <= frequencia_corte).astype(np.float64)
    # Aplicar filtro
    dft_shift_filtrada = dft_shift * filtro
    # Reverter centralização
    dft_filtrada = np.fft.ifftshift(dft_shift_filtrada)
    # Converter para imagem real
    imagem_filtrada_f = cv2.idft(dft_filtrada)
    # Pegar a magnitude real (parte real)
    imagem_filtrada_f = imagem_filtrada_f.real
    # Normalizar para o mesmo intervalo de dtype original
    if imagem.dtype == np.uint8:
        imagem_filtrada_f = np.clip(imagem_filtrada_f, 0, 255)
        imagem_filtrada = np.rint(imagem_filtrada_f).astype(np.uint8)
    else:
        # Preservar o tipo original (espera-se que seja float32 ou float64)
        imagem_filtrada = imagem_filtrada_f.astype(imagem.dtype)

    if formato_original is not None:
        # Converter de volta para BGR usando as médias dos canais? Usar fusão de escala de cinza para BGR para consistência
        # (Poderia recriar a imagem colorida original, mas mantendo os valores da escala de cinza)
        # Simples: gerar uma imagem BGR convertida de volta da escala de cinza usando cv2.cvtColor(cinza_f, CV_8UC3)
        # Mas precisamos corresponder ao tipo de entrada original (uint8?)
        imagem_filtrada = cv2.cvtColor(imagem_filtrada, cv2.COLOR_GRAY2BGR)

    return imagem_filtrada

def aplicar_hpf(imagem, frequencia_corte):
    """
    Aplica um filtro de passa-alta circular (HPF) à imagem no domínio da frequência.

    Parâmetros:
        imagem: array NumPy (BGR ou escala de cinza).
        frequencia_corte: frequência de corte em ciclos/pixel (0, 0.5].

    Retorna:
        imagem filtrada (mesmo dtype de entrada, mesma forma).
    """
    if not (0 < frequencia_corte <= 0.5):
        raise ValueError("frequencia_corte deve estar no intervalo (0, 0.5] ciclos/pixel.")

    _validar_imagem(imagem)
    se_escala_de_cinza = imagem.ndim == 2
    se_bgr = imagem.ndim == 3 and imagem.shape[2] == 3
    if not (se_escala_de_cinza or se_bgr):
        raise ValueError("A imagem deve ser BGR (3 canais) ou escala de cinza.")

    if se_bgr:
        cinza = converter_grayscale(imagem)
        formato_original = imagem
    else:
        cinza = imagem.copy()
        formato_original = None

    cinza_f = cinza.astype(np.float64)
    dft = cv2.dft(cinza_f)
    dft_shift = np.fft.fftshift(dft)
    fx, fy = _get_freq_grid(cinza_f.shape)
    fx, fy = np.fft.fftshift(fx), np.fft.fftshift(fy)
    freq_mag = np.hypot(fx, fy)
    # Filtro passa-alta: 1 onde freq_mag > corte
    filtro = (freq_mag > frequencia_corte).astype(np.float64)
    dft_shift_filtrada = dft_shift * filtro
    dft_filtrada = np.fft.ifftshift(dft_shift_filtrada)
    imagem_filtrada_f = cv2.idft(dft_filtrada).real

    if imagem.dtype == np.uint8:
        imagem_filtrada_f = np.clip(imagem_filtrada_f, 0, 255)
        imagem_filtrada = np.rint(imagem_filtrada_f).astype(np.uint8)
    else:
        imagem_filtrada = imagem_filtrada_f.astype(imagem.dtype)

    if formato_original is not None:
        imagem_filtrada = cv2.cvtColor(imagem_filtrada, cv2.COLOR_GRAY2BGR)

    return imagem_filtrada

def visualizar_perfil_frequencia(imagem, mostrar=True):
    """
    Calcula e opcionalmente exibe o perfil radial de energia da imagem.
    Retorna frequências em ciclos/pixel e energia média por faixa.
    Este é um wrapper em torno da função existente `perfil_radial` de fft_mascara.py.
    """
    try:
        from fft_mascara import perfil_radial
    except ImportError:
        raise ImportError(
            "fft_mascara.py não está disponível; certifique-se de que este módulo está instalado."
        )

    if imagem is None:
        raise ValueError("A imagem não foi carregada.")

    # Converter para escala de cinza se necessário (o mesmo método que em fft_mascara)
    if imagem.ndim == 3:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        cinza = imagem.copy()

    freq, energia = perfil_radial(cinza)

    if mostrar:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(freq, energia)
        plt.xlabel("Frequência (ciclos/pixel)")
        plt.ylabel("Energia média")
        plt.title("Perfil radial de energia da imagem")
        plt.grid(alpha=0.3)
        plt.show()

    return freq, energia