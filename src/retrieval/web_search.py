import requests
from ddgs import DDGS
from bs4 import BeautifulSoup
from langchain.schema import Document
from src.utils.config_manager import ConfigManager


class WebSearcher:
    """Web-search retriever using DuckDuckGo (no API key needed)."""

    def __init__(self, config: ConfigManager = None):
        # Load configuration or fallback
        self.config = config or ConfigManager()
        # Number of search results to retrieve
        self.num_results = self.config.get("web_search.num_results", 5)

    def search(self, query: str) -> list[Document]:
        """
        Perform a DuckDuckGo search via DDGS, fetch pages, return list of Documents.
        """
        docs = []
        # Use DDGS context manager for rate limiting
        with DDGS() as ddgs:
            # ddgs.text() yields dicts with keys: 'title', 'body', 'url'
            for hit in ddgs.text(query, max_results=self.num_results):
                title   = hit.get("title", "").strip()
                body    = hit.get("body", "") or ""
                snippet = body[:200].strip()  # first 200 characters as snippet
                url     = (hit.get("href") or hit.get("url") or "").strip()

                # Build a LangChain Document
                content = f"{title}\n\n{snippet}" if title else snippet
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "title": title,
                        }
                    )
                )
        return docs

    def _fetch_text(self, url: str) -> str:
        """
        Fetch a URL and extract its <p> text. Returns empty string on failure.
        """
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paras = [p.get_text().strip() for p in soup.find_all("p")]
            return "\n\n".join([p for p in paras if p])
        except Exception:
            return ""
