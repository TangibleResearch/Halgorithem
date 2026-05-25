import re


UNIT_ALIASES = {
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
    "km": "kilometer",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "mile": "mile",
    "miles": "mile",
    "c": "celsius",
    "celsius": "celsius",
    "f": "fahrenheit",
    "fahrenheit": "fahrenheit",
}

NORMALIZATION = {
    "gram": ("kilogram", 0.001, 0.0),
    "kilogram": ("kilogram", 1.0, 0.0),
    "meter": ("meter", 1.0, 0.0),
    "kilometer": ("meter", 1000.0, 0.0),
    "mile": ("meter", 1609.34, 0.0),
}

QUANTITY_RE = re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)\b")


def format_number(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def normalized_quantity(value, unit):
    canonical = UNIT_ALIASES.get(unit.lower())
    if not canonical:
        return None
    if canonical == "celsius":
        return float(value), "celsius"
    if canonical == "fahrenheit":
        return (float(value) - 32.0) * 5.0 / 9.0, "celsius"
    target = NORMALIZATION.get(canonical)
    if not target:
        return None
    target_unit, factor, offset = target
    return float(value) * factor + offset, target_unit


def normalize_units(sentence):
    changes = []

    def replace(match):
        raw_value = match.group("value")
        raw_unit = match.group("unit")
        normalized = normalized_quantity(raw_value, raw_unit)
        if not normalized:
            return match.group(0)
        normalized_value, normalized_unit = normalized
        normalized_text = f"{format_number(normalized_value)} {normalized_unit}"
        original_text = match.group(0)
        if original_text.lower() != normalized_text.lower():
            changes.append(
                {
                    "original": original_text,
                    "normalized": normalized_text,
                    "value": normalized_value,
                    "unit": normalized_unit,
                }
            )
        return normalized_text

    return QUANTITY_RE.sub(replace, sentence or ""), changes


def unit_representation_mismatch(left, right, tolerance=0.03):
    left_quantities = [
        (match.group(0), *normalized_quantity(match.group("value"), match.group("unit")))
        for match in QUANTITY_RE.finditer(left or "")
        if normalized_quantity(match.group("value"), match.group("unit"))
    ]
    right_quantities = [
        (match.group(0), *normalized_quantity(match.group("value"), match.group("unit")))
        for match in QUANTITY_RE.finditer(right or "")
        if normalized_quantity(match.group("value"), match.group("unit"))
    ]
    for left_original, left_value, left_unit in left_quantities:
        for right_original, right_value, right_unit in right_quantities:
            if left_unit != right_unit or left_original.lower() == right_original.lower():
                continue
            if right_value == 0:
                continue
            if abs(left_value - right_value) / abs(right_value) <= tolerance:
                return {
                    "source": left_original,
                    "response": right_original,
                    "normalized": f"{format_number(left_value)} {left_unit}",
                }
    return None
