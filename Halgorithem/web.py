from bs4 import BeautifulSoup
import requests
import html2text
from pathlib import Path

class WebScraper:
    def __init__(self, list_of_urls, output_dir="."):
        self.urls = list_of_urls
        self.output_dir = Path(output_dir)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True
        self.converter.ignore_images = True
        self.converter.ignore_tables = False
        self.counter = 0

    def scrape(self):
        results = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; HalgorithemBot/1.0)"
        }
        for url in self.urls:
            try:
                # use clean Wikipedia API instead of scraping
                if "wikipedia.org/wiki/" in url:
                    title = url.split("/wiki/")[-1]
                    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                    response = requests.get(api_url, 
                                        timeout=5, 
                                        headers=headers)
                    response.raise_for_status()
                    plain_text = response.json().get("extract", "")
                else:
                    page = requests.get(url, timeout=5, headers=headers)
                    page.raise_for_status()
                    soup = BeautifulSoup(page.content, "html.parser")
                    for tag in soup(["nav", "footer", "script",
                                    "style", "header", "aside"]):
                        tag.decompose()
                    plain_text = self.converter.handle(str(soup))
                    plain_text = plain_text[:8000]  # cap non-wiki sources

                file_path = self.output_dir / f"file{self.counter}.txt"
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(plain_text)
                print(f"Scraped: {url} → {file_path.name}")
                results.append({"url": url, "file_path": str(file_path), "text": plain_text, "ok": True, "error": None})
                self.counter += 1

            except requests.exceptions.Timeout:
                print(f"Timeout: {url}")
                results.append({"url": url, "file_path": None, "text": "", "ok": False, "error": "timeout"})
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error {e}: {url}")
                results.append({"url": url, "file_path": None, "text": "", "ok": False, "error": str(e)})
            except Exception as e:
                print(f"Failed {url}: {e}")
                results.append({"url": url, "file_path": None, "text": "", "ok": False, "error": str(e)})
        return results


def scrape_url_texts(urls, output_dir="."):
    """Scrapes URL sources into text records that can feed the PRISM verifier directly."""
    return [result for result in WebScraper(urls, output_dir=output_dir).scrape() if result.get("ok") and result.get("text")]
