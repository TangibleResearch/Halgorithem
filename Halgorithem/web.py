from bs4 import BeautifulSoup
import requests
import html2text

class WebScraper:
    def __init__(self, list_of_urls):
        self.urls = list_of_urls
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True
        self.converter.ignore_images = True
        self.converter.ignore_tables = False
        self.counter = 0

    def scrape(self):
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

                with open(f"file{self.counter}.txt", "w",
                        encoding="utf-8") as f:
                    f.write(plain_text)
                print(f"Scraped: {url} → file{self.counter}.txt")
                self.counter += 1

            except requests.exceptions.Timeout:
                print(f"Timeout: {url}")
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error {e}: {url}")
            except Exception as e:
                print(f"Failed {url}: {e}")