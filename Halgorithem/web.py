import asyncio
import concurrent.futures
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup
import html2text
import httpx


class WebScraper:
    def __init__(self, list_of_urls, output_dir="."):
        self.urls = list_of_urls
        self.output_dir = Path(output_dir)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True
        self.converter.ignore_images = True
        self.converter.ignore_tables = False

    async def _fetch_wikipedia(self, client, url):
        title = unquote(url.split("/wiki/")[-1])
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{title}"
        response = await client.get(api_url)
        response.raise_for_status()
        data = response.json()
        sections = data.get("lead", {}).get("sections", [])
        plain_text = "\n".join(section.get("text", "") for section in sections if section.get("text"))
        if plain_text:
            return plain_text
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = await client.get(summary_url)
        response.raise_for_status()
        return response.json().get("extract", "")

    async def _fetch_page(self, client, url):
        response = await client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
            tag.decompose()
        return self.converter.handle(str(soup))[:8000]

    async def _scrape_one(self, client, index, url):
        try:
            if "wikipedia.org/wiki/" in url:
                plain_text = await self._fetch_wikipedia(client, url)
            else:
                plain_text = await self._fetch_page(client, url)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"file{index}.txt"
            path.write_text(plain_text, encoding="utf-8")
            print(f"Scraped: {url} -> {path}")
            return str(path)
        except httpx.TimeoutException:
            print(f"Timeout: {url}")
        except httpx.HTTPStatusError as e:
            print(f"HTTP error {e}: {url}")
        except Exception as e:
            print(f"Failed {url}: {e}")
        return None

    async def scrape_async(self):
        headers = {"User-Agent": "Mozilla/5.0 (compatible; HalgorithemBot/1.0)"}
        timeout = httpx.Timeout(5.0)
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
            results = await asyncio.gather(
                *(self._scrape_one(client, index, url) for index, url in enumerate(self.urls))
            )
        return [path for path in results if path]

    def scrape(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.scrape_async())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.scrape_async()).result()
