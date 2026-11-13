import os
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from removedor_fundo.domain.models import DatasetSample


class OxfordPetDatasetRepository:
    """Leitor explicito de pares de imagem/trimap do Oxford-IIIT Pet."""

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            root = Path(
                os.environ.get("OXFORD_PET_DATASET")
                or Path(__file__).resolve().parents[3] / "datasets" / "oxford_pets"
            )
        self.root = Path(root)
        self._images = self.root / "images" / "images"
        self._trimaps = self.root / "annotations" / "annotations" / "trimaps"

    def image_paths(self) -> Sequence[Path]:
        if not self._images.is_dir():
            raise FileNotFoundError(f"Pasta de imagens nao encontrada: {self._images}")
        return sorted(
            path
            for path in self._images.iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in {".jpg", ".jpeg"}
        )

    def trimap_path(self, image_path: Path) -> Path:
        path = self._trimaps / f"{Path(image_path).stem}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Trimap nao encontrada: {path}")
        return path

    def load_pair(self, image_path: Path):
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")
        trimap_path = self.trimap_path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        trimap = cv2.imread(str(trimap_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Nao foi possivel carregar a imagem: {image_path}")
        if trimap is None or trimap.ndim != 2:
            raise ValueError(f"Trimap invalida: {trimap_path}")
        if image.shape[:2] != trimap.shape:
            raise ValueError(f"Imagem e trimap com dimensoes diferentes: {image_path.name}")
        if not set(np.unique(trimap).tolist()).issubset({1, 2, 3}):
            raise ValueError(f"Trimap com rotulos inesperados: {trimap_path}")
        return image, trimap

    def samples(self, limit: Optional[int] = None) -> Sequence[DatasetSample]:
        if not self._trimaps.is_dir():
            raise FileNotFoundError(f"Pasta de trimaps nao encontrada: {self._trimaps}")

        image_paths = self.image_paths()
        if limit is not None:
            if limit < 1:
                raise ValueError("O limite de amostras deve ser positivo.")
            image_paths = image_paths[:limit]

        samples: List[DatasetSample] = []
        for image_path in image_paths:
            image, trimap = self.load_pair(image_path)
            samples.append(
                DatasetSample(sample_id=image_path.stem, image=image, trimap=trimap)
            )
        return samples
