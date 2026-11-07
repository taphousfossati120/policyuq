from __future__ import annotations

from attrs import frozen

VOCAB_SIZE = 87


@frozen
class Descriptor:
    group: str
    text: str


def _zones() -> list[Descriptor]:
    items: list[Descriptor] = []
    for side in ("right", "left"):
        for level in ("apex", "mid-gland", "base"):
            for zone in ("peripheral zone", "transition zone"):
                items.append(Descriptor("location", f"{side} {level} {zone}"))
    return items


def _t2() -> list[Descriptor]:
    grades = [Descriptor("signal_t2", f"T2 hypointensity grade {i}") for i in range(1, 6)]
    grades.append(Descriptor("signal_t2", "T2 homogeneous low signal"))
    grades.append(Descriptor("signal_t2", "T2 heterogeneous signal"))
    return grades


def _dwi() -> list[Descriptor]:
    grades = [Descriptor("signal_dwi", f"DWI hyperintensity grade {i}") for i in range(1, 6)]
    grades.append(Descriptor("signal_dwi", "DWI focal high signal"))
    grades.append(Descriptor("signal_dwi", "DWI diffuse high signal"))
    return grades


def _adc() -> list[Descriptor]:
    items: list[Descriptor] = []
    for level in ("markedly", "moderately", "mildly"):
        for site in ("focal", "regional"):
            items.append(Descriptor("morphology_adc", f"ADC {level} restricted {site}"))
    for tail in ("equivocal", "borderline", "minimal", "no"):
        items.append(Descriptor("morphology_adc", f"ADC {tail} restriction"))
    return items


def _shape() -> list[Descriptor]:
    items: list[Descriptor] = []
    for shape in ("round", "oval", "lenticular", "wedge-shaped", "crescentic", "irregular"):
        for extent in ("well-defined", "ill-defined"):
            items.append(Descriptor("morphology_shape", f"{extent} {shape} lesion"))
    return items


def _margin() -> list[Descriptor]:
    names = (
        "circumscribed",
        "indistinct",
        "spiculated",
        "lobulated",
        "infiltrative",
        "smooth",
        "angular",
        "microlobulated",
        "ill-defined",
        "sharp",
        "irregular",
    )
    return [Descriptor("morphology_margin", f"{name} margin") for name in names]


def _size() -> list[Descriptor]:
    bands = (
        "<0.5 cm",
        "0.5-0.8 cm",
        "0.8-1.0 cm",
        "1.0-1.3 cm",
        "1.3-1.5 cm",
        "1.5-1.8 cm",
        "1.8-2.0 cm",
        "2.0-2.5 cm",
        "2.5-3.0 cm",
        ">3.0 cm",
    )
    return [Descriptor("morphology_size", f"diameter {band}") for band in bands]


def _epe() -> list[Descriptor]:
    states = (
        "no extraprostatic extension",
        "capsular abutment",
        "capsular contact under 1 cm",
        "capsular contact over 1 cm",
        "capsular irregularity",
        "capsular bulge",
        "measurable extracapsular extension",
        "neurovascular bundle involvement",
        "gross extracapsular extension",
        "bladder neck involvement",
    )
    return [Descriptor("associated_epe", state) for state in states]


def _svi() -> list[Descriptor]:
    states = (
        "no seminal vesicle involvement",
        "seminal vesicle abutment",
        "seminal vesicle contact",
        "suspected seminal vesicle invasion",
        "early seminal vesicle invasion",
        "established seminal vesicle invasion",
        "bilateral seminal vesicle invasion",
        "seminal vesicle obliteration",
    )
    return [Descriptor("associated_svi", state) for state in states]


def build_vocabulary() -> list[Descriptor]:
    descriptors = (
        _zones() + _t2() + _dwi() + _adc() + _shape() + _margin() + _size() + _epe() + _svi()
    )
    if len(descriptors) != VOCAB_SIZE:
        raise ValueError(f"vocabulary must hold {VOCAB_SIZE} descriptors, built {len(descriptors)}")
    return descriptors


DESCRIPTORS: list[Descriptor] = build_vocabulary()
