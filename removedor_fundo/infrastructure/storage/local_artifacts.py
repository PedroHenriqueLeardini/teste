import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from removedor_fundo.domain.models import DatasetEvaluation, ProcessingResult
from removedor_fundo.infrastructure.imaging.opencv_adapters import save_image


class LocalArtifactRepository:
    def __init__(self, root: Path = Path("artifacts")) -> None:
        self._root = Path(root)

    def save_processing_result(
        self, result: ProcessingResult, name: str
    ) -> Mapping[str, str]:
        directory = self._root / "results"
        directory.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(name)
        paths = {
            "raw_mask": directory / f"{stem}_mask_raw.png",
            "processed_mask": directory / f"{stem}_mask_processed.png",
            "image": directory / f"{stem}_result.png",
            "metadata": directory / f"{stem}.json",
        }
        save_image(paths["raw_mask"], result.raw_mask)
        save_image(paths["processed_mask"], result.processed_mask)
        save_image(paths["image"], result.transparent_image)
        paths["metadata"].write_text(
            json.dumps(
                {
                    "method": result.method,
                    "refined": result.refined,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {key: str(path) for key, path in paths.items()}

    def save_dataset_evaluations(
        self, evaluations: Sequence[DatasetEvaluation], name: str
    ) -> str:
        directory = self._root / "experiments"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_name(name)}.json"
        rows = [
            {
                "sample_id": item.sample_id,
                "method": item.method,
                "raw_metrics": dict(item.raw_metrics),
                "processed_metrics": dict(item.processed_metrics),
            }
            for item in evaluations
        ]
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not cleaned:
        raise ValueError("O nome do artefato precisa conter caracteres validos.")
    return cleaned
