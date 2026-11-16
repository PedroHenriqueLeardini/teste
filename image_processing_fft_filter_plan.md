# Plano de escrita: adição de filtros de imagem LPF e HPF ao `image_processing.py`

## Propósito
Este plano descreve as novas funções a serem adicionadas ao módulo existente `image_processing.py`:

- `aplicar_lpf(imagem, frequencia_corte)` – passa-baixa no domínio da frequência (remove altas frequências, suaviza)
- `aplicar_hpf(imagem, frequencia_corte)` – passa-alta no domínio da frequência (preserva altas frequências, destaca bordas)
- `visualizar_perfil_frequencia(imagem)` – wrapper para exibir dados do perfil de frequência (como `perfil_radial` de `fft_mascara.py`, mas para qualquer imagem)

Todas as funções mantêm a consistência com a API já presente:
- Validação de entrada (formato NumPy, valores divisíveis por 3, etc.)
- Retorna o mesmo tipo (float32/uint8 quando apropriado)
- Tratamento de tipos de erro (`ValueError`, `TypeError`)
- Compatível com imagem BGR ou escala de cinza (será convertido internamente para escala de cinza para filtragem)

## Diretrizes de projeto

### Cobertura de segurança e validação
- Validar a imagem para garantir que seja um array NumPy não vazio (reutilizar `_validar_imagem`)
- Aceitar tanto 2D (escala de cinza) quanto 3D (BGR). Converter para escala de cinza se for BGR para filtragem.
- Garantir `frequencia_corte` dentro do intervalo admissível `(0, 0.5]` ciclos/pixel (a frequência máxima é Nyquist, 0.5 ciclos/pixel)
- Manter precisão numérica: usar `float64` para cálculos da FFT, reverter para o tipo original ao final
- Verificar níveis de ruído alto (componente DC constante) e casos extremos (tamanho pequeno, corte alto)

### Convenções de implementação
- Usar as rotinas de FFT do OpenCV: `cv2.dft`/`cv2.idft` (mais rápidas) ou da biblioteca NumPy (`np.fft.fft2`/`ifft2`) com centralização (`fftshift`) para processamento intuitivo.
- Usar máscara circular de frequência em coordenadas normalizadas (frequências de 0 a Nyquist)
- O tamanho do kernel do filtro pode ser inferido da forma da imagem ou ser um parâmetro opcional
- Preservar o canal alfa (se presente) para filtros não alfarráficos
- As funções devem ser idempotentes para corte=0 (sem filtragem) e corte=0.5 (zero em todas as frequências não DC)

### Regra de escrita
- Adicionar docstrings completas com `Parameters` / `Returns` / `Raises` / `Examples`
- Usar a função de ajuda existente `_validar_imagem` para validação reutilizável
- Adicionar mensagens de erro claras e descritivas
- Manter o arquivo organizado: novas funções após as existentes, manter importações no topo
- Seguir o estilo `snake_case` para nomes de funções
- Adicionar uma breve descrição da filtragem no topo do arquivo (opcional, pode ser uma tag de comentário)

## Teste e depuração

### Abordagem de testes unitários
- Criar arquivo de testes unitários dedicado (o projeto usa `tests/unit/`). Usar a mesma estrutura que `test_legacy_compatibility.py` para manter a compatibilidade.
- Usar fixtures fornecidas por `pytest`: dados de imagem reais (por exemplo, usando `Path` para `removedor_fundo/infrastructure/datasets/oxford_pet.py`)
- Testar três caminhos de dados:
  1. Imagem BGR (3 canais) → filtragem de escala de cinza interna
  2. Imagem já em escala de cinza (2D)
  3. Imagem None ou inválida → deve levantar `ValueError`
- Testar diferentes cortes: 0 (sem filtro), 0.1, 0.25, 0.5 (máximo). Esperar suavidade progressiva para LPF, nitidez para HPF.
- Validar tipos de saída (mesmo dtype de entrada ou recomendado)
- Usar métricas FFT: comparar energia antes e depois, verificar espectro após filtragem.

### Verificação de segurança
- Validar `frequencia_corte`: negativo, >0.5, não numérico → `ValueError`
- Garantir que o tamanho da imagem >= 8 (para evitar divisão por zero na FFT de alta frequência)
- Verificar a presença de NaN, Inf nos resultados após filtragem (limpar se detectado)

### Documentação e comentários
- Adicionar exemplos no docstring: aplicar LPF com corte 0.1 → observar redução de ruído; aplicar HPF com corte 0.2 → bordas realçadas.
- Escrever um breve README de resumo para a nova API (talvez adicionar um comentário no topo de `image_processing.py`)
- Incluir o perfil de frequência no registro de logs do `removedor_fundo` para depuração (usar `logging_config.py`)

## Sequência de implementação

1. **Definir helper `get_freq_grid`** (computar matriz de frequência normalizada)
2. **Implementar `aplicar_lpf`**
   - Criar máscara de filtro (circular, transmissão baixa)
   - Aplicar FFT → multiplicar pelo filtro → IFFT → reverter centralização → retornar (mesmo dtype)
   - Testar comportamento idempotente e limites do corte
3. **Implementar `aplicar_hpf`**
   - Igual, mas máscara de transmissão alta (complementar ao LPF)
   - Ou `hpf = 1 - lpf_mask`
4. **Implementar `visualizar_perfil_frequencia`**
   - Wrapper para `fft_mascara.perfil_radial` (ou copiar lógica)
   - Retornar `(frequencias, energias)` para traçar ou integrar com Matplotlib
   - Opcionalmente salvar figura em arquivo
5. **Executar testes unitários** (usar `execute` para scripts de teste TDD)
6. **Executar depuração sistemática** para validação de segurança e casos extremos
7. **Integrar com a interface do usuário** (interface Tkinter, pipeline)
   - Expor parâmetros de corte na interface gráfica do usuário
   - Chamar o filtro na etapa pós-segmentação (opcional)
8. **Documentar** o uso em `README.md` e no manual `ROADMAP.md` para a nova funcionalidade de filtragem.

## Membros de suporte

- Usar `image_processing.py` já existente para consistência.
- Aproveitar `fft_mascara.py` existente para lógica de perfil de frequência, para evitar duplicação.
- Usar `removedor_fundo.infrastructure.logging_config` para logs de validação de filtro.
- Usar `removedor_fundo.domain.errors` para lançar `UnsafeSegmentation` ou `InvalidCommand` se o filtro for chamado a partir do pipeline.

## Dependências

- NumPy (já presente)
- OpenCV (`cv2`) (já presente)
- Matplotlib (apenas para `visualizar_perfil_frequencia`, opcional)

## Métricas de sucesso

- Todas as funções passam por testes unitários (≥95% de cobertura)
- Sem AssertionError relacionado a tipos de saída
- Os logs registram chamadas de filtro e cortes
- A interface do usuário pode exibir resultados LPF/HPF lado a lado
- A linha do pipeline de experimentos permite “Filtrar com LPF de 0.1 ciclos/pixel” e avaliar métricas de IoU.

---
*Próximo passo*: Escolher habilidade `writing-plans` para refinar o plano, depois escolher `systematic-debugging` para validação de segurança e executar testes com `execute*`.*