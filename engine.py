import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from Halgorithem import Halgorithm
from Halgorithem.web import WebScraper

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


class Engine:
    def __init__(self, model=DEFAULT_MODEL, sentences_per_chunk=2, sentence_overlap=1, timeout=10):
        self.model = model
        self.timeout = timeout
        self.client = None
        self.algo = Halgorithm(sentences_per_chunk=sentences_per_chunk, sentence_overlap=sentence_overlap)

    def openai_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        return self.client

    def scrape_urls(self, urls: Iterable[str]):
        if not (urls := list(urls or [])):
            return []
        docs = []
        with TemporaryDirectory(prefix="halgorithem-scrape-") as tmp:
            prev = Path.cwd()
            os.chdir(tmp)
            try:
                WebScraper(urls).scrape()
            finally:
                os.chdir(prev)
            for i, url in enumerate(urls, 1):
                f = Path(tmp) / f"file{i - 1}.txt"
                if not f.exists():
                    print(f"Warning: scrape failed for {url}")
                    continue
                text = f.read_text(encoding="utf-8")
                docs.append({"file_id": i, "file_path": url, "text": text})
        return docs

    def load_truth_files(self, file_paths: Iterable[str]):
        # delegates to Halgorithm.load_files() — same logic, no duplication
        return self.algo.load_files(list(file_paths or []))

    def generate(self, prompt, source_docs=None):
        source_docs = source_docs or []
        per_doc = 18000 // max(len(source_docs), 1)
        context = "\n\n---\n\n".join(
            f"Source: {d['file_path']}\n{d['text'][:per_doc]}"
            for d in source_docs
        )
        user_input = (
            f"Question:\n{prompt}\n\nUse only these scraped/source documents as factual grounding:\n{context}"
            if context else prompt
        )
        return self.openai_client().responses.create(
            model=self.model,
            instructions="Answer clearly and factually. When source documents are supplied, do not add facts that are not supported by those documents.",
            input=user_input,
        ).output_text

    def verify(self, ai_output, source_docs, threshold=0.30):
        # compare_to_docs accepts dicts natively — no conversion needed
        claims = self.algo.compare_to_docs(
            truth_docs=source_docs,
            ai_output=ai_output,
            threshold=threshold,
        )
        return {"claims": claims, "summary": self.summarize(claims)}

    def _load_sources(self, urls=None, truth_file_paths=None):
        return self.scrape_urls(urls or []) + self.load_truth_files(truth_file_paths or [])

    def run(self, prompt, urls=None, truth_file_paths=None, threshold=0.30):
        if not (source_docs := self._load_sources(urls, truth_file_paths)):
            raise ValueError("No source documents loaded. Provide urls or truth_file_paths.")
        ai_output = self.generate(prompt, source_docs)
        verification = self.verify(ai_output, source_docs, threshold)
        return {**verification, "ai_output": ai_output, "sources": [d["file_path"] for d in source_docs]}

    def summarize(self, claims):
        if not (total := len(claims)):
            return "No verifiable claims found."
        counts = {s: sum(1 for c in claims if c.get("status") == s)
                  for s in ("SUPPORTED", "WEAK_SUPPORT", "CONTRADICTION", "HALLUCINATION")}
        return (f"{counts['SUPPORTED']}/{total} supported, {counts['WEAK_SUPPORT']}/{total} weak, "
                f"{counts['CONTRADICTION']}/{total} contradictions, {counts['HALLUCINATION']}/{total} hallucinations")


_engine = Engine()

def run(prompt, urls=None, truth_file_paths=None, threshold=0.30):
    return _engine.run(prompt=prompt, urls=urls, truth_file_paths=truth_file_paths, threshold=threshold)

def generate(prompt, urls=None, truth_file_paths=None):
    return _engine.generate(prompt, _engine._load_sources(urls, truth_file_paths))

def verify(ai_output, urls=None, truth_file_paths=None, threshold=0.30):
    if not (source_docs := _engine._load_sources(urls, truth_file_paths)):
        raise ValueError("No source documents loaded. Provide urls or truth_file_paths.")
    return _engine.verify(ai_output, source_docs, threshold)