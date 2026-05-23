from urllib.parse import urlparse


HIGH_TRUST_DOMAINS = {
    "wikipedia.org": 0.78,
    "britannica.com": 0.82,
    "nasa.gov": 0.92,
    "nih.gov": 0.90,
    "noaa.gov": 0.90,
    "sec.gov": 0.92,
    "whitehouse.gov": 0.86,
}


def score_source(source_name, text):
    source_name = source_name or ""
    text = text or ""
    parsed = urlparse(source_name)
    host = parsed.netloc.lower().removeprefix("www.")

    score = 0.55
    for domain, domain_score in HIGH_TRUST_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            score = max(score, domain_score)

    if source_name.startswith("inline_text") or not parsed.scheme:
        score = max(score, 0.65)
    if len(text.split()) < 80:
        score -= 0.15
    if text.count("\n") > len(text.split()) / 4:
        score -= 0.05

    return round(max(0.0, min(score, 1.0)), 2)
