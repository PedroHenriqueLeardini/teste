import unicodedata
from typing import Optional, Sequence


CATEGORIAS_VOC = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
)

OBJETOS_PT = {
    "cachorro": ["dog"], "cao": ["dog"], "cachorra": ["dog"],
    "gato": ["cat"], "gata": ["cat"],
    "animal": ["cat", "dog"], "pet": ["cat", "dog"],
    "pessoa": ["person"], "gente": ["person"], "homem": ["person"],
    "mulher": ["person"], "crianca": ["person"],
    "carro": ["car"], "onibus": ["bus"], "trem": ["train"],
    "bicicleta": ["bicycle"], "moto": ["motorbike"],
    "cavalo": ["horse"], "vaca": ["cow"], "ovelha": ["sheep"],
    "passaro": ["bird"], "ave": ["bird"], "barco": ["boat"],
    "aviao": ["aeroplane"], "garrafa": ["bottle"],
    "cadeira": ["chair"], "sofa": ["sofa"], "planta": ["pottedplant"],
    "mesa": ["diningtable"], "tv": ["tvmonitor"],
}


def normalize_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip()


def classes_for_object(object_name: Optional[str]) -> Sequence[int]:
    if object_name is None:
        return list(range(1, len(CATEGORIAS_VOC)))
    name = normalize_text(object_name)
    labels = OBJETOS_PT.get(name)
    if labels is None and name in CATEGORIAS_VOC and name != "background":
        labels = [name]
    if labels is None:
        raise ValueError(f"Objeto nao suportado pelo modelo: {object_name}")
    return [CATEGORIAS_VOC.index(label) for label in labels]

