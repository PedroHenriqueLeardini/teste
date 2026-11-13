"""Fachada de compatibilidade para o repositorio Oxford-IIIT Pet."""

import os
from pathlib import Path

from removedor_fundo.infrastructure.datasets.oxford_pet import (
    OxfordPetDatasetRepository,
)


RAIZ_DATASET = Path(
    os.environ.get("OXFORD_PET_DATASET")
    or Path(__file__).resolve().parent / "datasets" / "oxford_pets"
)
PASTA_IMAGENS = RAIZ_DATASET / "images" / "images"
PASTA_TRIMAPS = RAIZ_DATASET / "annotations" / "annotations" / "trimaps"


def _repository():
    return OxfordPetDatasetRepository(RAIZ_DATASET)


def listar_imagens():
    """Lista imagens JPG/JPEG do dataset em ordem alfabética."""
    return list(_repository().image_paths())


def obter_trimap(caminho_imagem):
    """Retorna o caminho para o trimap correspondente."""
    return _repository().trimap_path(Path(caminho_imagem))


def carregar_par(caminho_imagem):
    """Carrega e valida a imagem BGR e seu trimap pareado."""
    return _repository().load_pair(Path(caminho_imagem))
