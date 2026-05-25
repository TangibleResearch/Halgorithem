import re


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def token_set(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def overlap_ratio(left, right):
    left_tokens = {t for t in token_set(left) if len(t) > 2}
    right_tokens = {t for t in token_set(right) if len(t) > 2}
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)
