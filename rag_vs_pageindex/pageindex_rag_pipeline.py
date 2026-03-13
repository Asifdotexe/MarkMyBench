"""
PageIndex Pipeline Module.

This module provides the foundation for the PageIndex (Vectorless RAG)
approach. It processes documents into hierarchical semantic trees and
utilizes LLM-driven navigation for context extraction.
"""
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# NOTE: Load .env explicitly from the benchmark directory
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

_SUPPORTED_EXTENSIONS = {".pdf"}
_HTML_URL_PREFIXES = ("http://", "https://")

# Default paths for saving the PageIndex tree
_TREE_FILE = "pageindex_tree.pkl"


class PageIndexPipeline:
    """
    Pipeline for executing a Vectorless PageIndex workflow.
    """

    def __init__(self, db_path: str = "./pageindex_db", model: str = "gemini-2.5-flash") -> None:
        """
        Initializes the pipeline with necessary configurations for tree construction.

        :param db_path: Path to the local directory where the semantic tree is persisted.
        :param model: The Gemini model to use for tree generation and navigation.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # NOTE: Temperature 0 for deterministic tree generation and retrieval.
        self._llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        self._tree: dict[str, Any] = {}

        # Warm-start: reload tree if it already exists to save massive API costs.
        tree_file = self.db_path / _TREE_FILE
        if tree_file.exists():
            with open(tree_file, "rb") as fh:
                self._tree = pickle.load(fh)

    def ingest_document(self, source: str) -> str:
        """
        Ingests a document from a local PDF or remote HTML URL, extracting raw text
        while embedding explicit structural boundaries (`<page_N>` or `<section_N>`).

        These explicit tags are the backbone of the PageIndex approach — they demarcate
        the natural semantic boundaries of the document which the LLM will later summarize.

        :param source: A local file path (PDF) or a full HTTP/HTTPS URL (HTML).
        """
        if source.startswith(_HTML_URL_PREFIXES):
            return self._parse_html_url(source)
        return self._parse_pdf(source)

    def _parse_pdf(self, file_path: str) -> str:
        """
        Extracts text from a local PDF, wrapping each page's content in `<page_N>` tags.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found at: '{file_path}'")
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{path.suffix}'.")

        extracted_nodes: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # NOTE: We inject explicit structural boundaries here.
                    node_content = (
                        f"<page_{page_num}>\n"
                        f"{page_text.strip()}\n"
                        f"</page_{page_num}>"
                    )
                    extracted_nodes.append(node_content)

        return "\n\n".join(extracted_nodes)

    def _parse_html_url(self, url: str) -> str:
        """
        Fetches an HTML page and extracts text, grouping lines into logical sections
        wrapped in `<section_N>` tags to simulate page boundaries.
        """
        headers = {
            "User-Agent": "MarkMyBench/0.1 (Academic Use; asifdotexe@gmail.com)"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        raw_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # NOTE: Since HTML lacks natural "pages", we group text into coarse sections
        # of ~50 lines (roughly equivalent to a printed page) and wrap them.
        # This prevents the LLM from having to summarize an entire 10-K report in one go.
        extracted_nodes: list[str] = []
        section_num = 1
        lines_per_section = 50

        for i in range(0, len(lines), lines_per_section):
            section_text = "\n".join(lines[i : i + lines_per_section])
            node_content = (
                f"<section_{section_num}>\n"
                f"{section_text}\n"
                f"</section_{section_num}>"
            )
            extracted_nodes.append(node_content)
            section_num += 1

        return "\n\n".join(extracted_nodes)

    def generate_semantic_tree(self, text: str) -> dict[str, Any]:
        """
        Processes the document text to construct a hierarchical semantic tree.

        Parses the explicit structural tags (`<page_N>` or `<section_N>`) injected
        during ingestion. For each section, it calls the LLM to generate a
        one-sentence summary. The resulting tree acts as a semantic "Table of Contents",
        allowing queries to be routed logically rather than via vector distance.

        :param text: The raw text extracted from the document containing structural tags.
        """
        if self._tree:
            # NOTE: If we loaded a cached tree during __init__, return it immediately.
            # Building this tree costs API tokens and time, so we only do it once.
            return self._tree

        # Regex to find all structurally tagged sections.
        # re.DOTALL ensures '.' matches newlines as well.
        pattern = re.compile(r"<(page|section)_(\d+)>\n(.*?)\n</\1_\2>", re.DOTALL)
        matches = list(pattern.finditer(text))

        if not matches:
            raise ValueError("No structural tags found in the text. Did you run ingest_document?")

        def summarize_section(match: re.Match) -> dict[str, Any]:
            tag_type = match.group(1)
            node_id = int(match.group(2))
            content = match.group(3).strip()

            messages = [
                (
                    "system",
                    (
                        "You are an expert financial analyst and document structurer. "
                        "Summarize the following document section in exactly ONE concise sentence. "
                        "Focus heavily on the main topics, entities, and numerical metrics discussed "
                        "so that this summary can be used as an index to find the section later."
                    )
                ),
                ("human", content)
            ]

            summary = self._llm.invoke(messages).content
            return {
                "id": node_id,
                "type": tag_type,
                "summary": summary,
                "content": content
            }

        # NOTE: We use a ThreadPoolExecutor to summarize sections concurrently.
        # 10-K documents can have 100+ pages; doing this sequentially would take
        # several minutes. max_workers=5 is a safe middle ground that speeds up
        # processing without immediately triggering OpenAI's Tier 1 Rate Limits (429s).
        nodes = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            nodes = list(executor.map(summarize_section, matches))

        # Ensure nodes remain in their original document order.
        nodes.sort(key=lambda n: n["id"])

        self._tree = {"nodes": nodes}

        # Persist the tree to disk to save API costs on future runs.
        with open(self.db_path / _TREE_FILE, "wb") as fh:
            pickle.dump(self._tree, fh)

        return self._tree

    def retrieve_context_via_tree(self, query: str, tree: dict[str, Any]) -> list[str]:
        """
        Navigates the semantic tree using an LLM to extract relevant context.

        :param query: The user query to evaluate against the tree.
        :param tree: The hierarchical semantic tree representation of the document.
        """
        # FIXME: LLM-driven navigation might become a bottleneck under heavy loads.
        # Consider implementing caching strategies for repetitive or similar queries.
        # TODO: Implement reasoning-based tree traversal logic to find relevant paths.
        pass

    def generate_answer(self, query: str, context: list[str]) -> str:
        """
        Generates a concise answer based on the context extracted from the tree.

        :param query: The user query.
        :param context: The context strings extracted via tree navigation.
        """
        # TODO: Implement final generation step based on tree-derived context.
        pass
