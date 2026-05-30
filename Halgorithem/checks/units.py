UNIT_ALIASES = {
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
    "cm": "centimeter",
    "centimeter": "centimeter",
    "centimeters": "centimeter",
    "km": "kilometer",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "mile": "mile",
    "miles": "mile",
}

NORMALIZATION = {
    "gram": ("mass", 0.001),
    "kilogram": ("mass", 1.0),
    "pound": ("mass", 0.45359237),
    "meter": ("length", 1.0),
    "centimeter": ("length", 0.01),
    "kilometer": ("length", 1000.0),
    "mile": ("length", 1609.344),
}


def normalize_unit(unit):
    canonical = UNIT_ALIASES.get((unit or "").lower())
    return NORMALIZATION.get(canonical) if canonical else None
