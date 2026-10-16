DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
)

PPE_CLASSES: tuple[str, ...] = (
    "Hardhat",
    "Mask",
    "Safety Vest",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
)

VIOLATION_CLASSES: tuple[str, ...] = ("NO-Hardhat", "NO-Mask", "NO-Safety Vest")

PPE_PAIRS: dict[str, str] = {
    "Hardhat": "NO-Hardhat",
    "Mask": "NO-Mask",
    "Safety Vest": "NO-Safety Vest",
}


def default_class_groups(class_names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "all": tuple(class_names),
        "ppe": PPE_CLASSES,
        "violation": VIOLATION_CLASSES,
    }
