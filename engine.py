import os
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from Halgorithem import Halgorithm
from Halgorithem.web import WebScraper


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


@dataclass
class SourceDocument:
    file_id: int
    file_path: str
    text: str

    def as_halgorithem_doc(self):
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "text": self.text,
        }


class Engine:
    def __init__(
        self,
        model=DEFAULT_MODEL,
        sentences_per_chunk=2,
        sentence_overlap=1,
        timeout=10,
    ):
        self.model = model
        self.timeout = timeout
        self.client = None
        self.algo = Halgorithm(
            sentences_per_chunk=sentences_per_chunk,
            sentence_overlap=sentence_overlap,
        )

    def openai_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        return self.client

    def scrape_urls(self, urls: Iterable[str]):
        urls = list(urls or [])
        docs = []

        if not urls:
            return docs

        with TemporaryDirectory(prefix="halgorithem-scrape-") as temp_dir:
            with self.pushd(temp_dir):
                scraper = WebScraper(urls)
                scraper.scrape()

                for index, url in enumerate(urls, start=1):
                    scraped_file = Path(temp_dir) / f"file{index - 1}.txt"
                    if not scraped_file.exists():
                        print(f"Warning: scrape failed for {url}")
                        continue

                    text = scraped_file.read_text(encoding="utf-8")
                    print(f"Scraped {url} → {len(text)} chars")  # debug line
                    docs.append(
                        SourceDocument(
                            file_id=index,
                            file_path=url,
                            text=text,
                        )
                    )

        return docs

    @contextmanager
    def pushd(self, directory):
        previous = Path.cwd()
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous)

    def load_truth_files(self, file_paths: Iterable[str]):
        docs = []
        for index, file_path in enumerate(file_paths or [], start=1):
            path = Path(file_path)
            docs.append(
                SourceDocument(
                    file_id=index,
                    file_path=str(path),
                    text=path.read_text(encoding="utf-8"),
                )
            )
        return docs

    def build_context(self, docs, max_chars=18000):
        sections = []
        remaining = max_chars

        for doc in docs:
            text = doc.text[:remaining]
            if not text:
                break
            sections.append(f"Source: {doc.file_path}\n{text}")
            remaining -= len(text)

        return "\n\n---\n\n".join(sections)

    def generate(self, prompt, source_docs=None):
        source_docs = source_docs or []
        context = self.build_context(source_docs)

        if context:
            user_input = (
                f"Question:\n{prompt}\n\n"
                f"Use only these scraped/source documents as factual grounding:\n{context}"
            )
        else:
            user_input = prompt

        response = self.openai_client().responses.create(
            model=self.model,
            instructions=(
                "Answer clearly and factually. When source documents are supplied, "
                "do not add facts that are not supported by those documents."
            ),
            input=user_input,
        )
        return response.output_text

    def verify(self, ai_output, source_docs, threshold=0.30):
        # source_docs is now required - no silent fallback
        print(f"Verifying against {len(source_docs)} source docs")  # debug
        docs = [doc.as_halgorithem_doc() for doc in source_docs]
        claims = self.algo.compare_to_docs(
            truth_docs=docs,
            ai_output=ai_output,
            threshold=threshold,
        )
        return {
            "claims": claims,
            "summary": self.summarize(claims),
        }

    def run(self, prompt, urls=None, truth_file_paths=None, threshold=0.30):
        source_docs = []
        source_docs.extend(self.scrape_urls(urls or []))
        source_docs.extend(self.load_truth_files(truth_file_paths or []))

        # explicit error if nothing loaded
        if not source_docs:
            raise ValueError(
                "No source documents loaded. "
                "Provide urls or truth_file_paths."
            )

        print(f"Total source docs: {len(source_docs)}")  # debug

        ai_output = self.generate(prompt, source_docs=source_docs)
        verification = self.verify(
            ai_output=ai_output,
            source_docs=source_docs,  # explicit pass, no fallback
            threshold=threshold,
        )

        return {
            "ai_output": ai_output,
            "claims": verification["claims"],
            "summary": verification["summary"],
            "sources": [doc.file_path for doc in source_docs],
        }

    def summarize(self, claims):
        total = len(claims)
        if total == 0:
            return "No verifiable claims found."

        supported = sum(1 for c in claims if c.get("status") == "SUPPORTED")
        weak = sum(1 for c in claims if c.get("status") == "WEAK_SUPPORT")
        contradictions = sum(1 for c in claims if c.get("status") == "CONTRADICTION")
        hallucinations = sum(1 for c in claims if c.get("status") == "HALLUCINATION")

        return (
            f"{supported}/{total} supported, "
            f"{weak}/{total} weak, "
            f"{contradictions}/{total} contradictions, "
            f"{hallucinations}/{total} hallucinations"
        )


_engine = Engine()


def run(prompt, urls=None, truth_file_paths=None, threshold=0.30):
    return _engine.run(
        prompt=prompt,
        urls=urls,
        truth_file_paths=truth_file_paths,
        threshold=threshold,
    )


def generate(prompt, urls=None, truth_file_paths=None):
    source_docs = []
    source_docs.extend(_engine.scrape_urls(urls or []))
    source_docs.extend(_engine.load_truth_files(truth_file_paths or []))
    return _engine.generate(prompt, source_docs=source_docs)


def verify(ai_output, urls=None, truth_file_paths=None, threshold=0.30):
    source_docs = []
    source_docs.extend(_engine.scrape_urls(urls or []))
    if truth_file_paths:
        source_docs.extend(_engine.load_truth_files(truth_file_paths))

    if not source_docs:
        raise ValueError(
            "No source documents loaded. "
            "Provide urls or truth_file_paths."
        )

    return _engine.verify(
        ai_output=ai_output,
        source_docs=source_docs,
        threshold=threshold,
    )
