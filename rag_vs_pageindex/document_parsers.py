"""
Document parsing utilities shared across benchmarking pipelines.

This module centralises extraction logic to avoid duplicating parsing code 
between Traditional RAG and Vectorless PageIndex approaches.
"""

from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup

_SUPPORTED_EXTENSIONS = {".pdf"}
_HTML_URL_PREFIXES = ("http://", "https://")


def ingest_document(source: str, keep_structure: bool = False) -> str:
    """
    Ingests a document from either a local PDF path or a remote HTML URL.

    :param source: A local file path (PDF) or a full HTTP/HTTPS URL (HTML).
    :param keep_structure: If True, wraps sequential blocks of text in explicit
                           `<page_N>` or `<section_N>` XML-style tags. The PageIndex
                           pipeline relies on these tags to build its semantic True.
    :raises FileNotFoundError: If a local path is given but does not exist.
    :raises ValueError: If the file extension is unsupported for local paths.
    :raises requests.HTTPError: If the HTTP request for a URL source fails.
    """
    if source.startswith(_HTML_URL_PREFIXES):
        return parse_html_url(source, keep_structure)

    return parse_pdf(source, keep_structure)


def parse_pdf(file_path: str, keep_structure: bool = False) -> str:
    """
    Extracts plain text from a local PDF file using ``pdfplumber``.

    :param file_path: Absolute or relative path to the ``.pdf`` file.
    :param keep_structure: If True, wraps text on each page in `<page_N>` tags.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found at path: '{file_path}'")

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Only the following extensions are supported: {_SUPPORTED_EXTENSIONS}"
        )

    extracted_pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue

            text = page_text.strip()
            
            if keep_structure:
                node_content = (
                    f"<page_{page_num}>\n"
                    f"{text}\n"
                    f"</page_{page_num}>"
                )
                extracted_pages.append(node_content)
            else:
                extracted_pages.append(text)

    return "\n\n".join(extracted_pages)


def parse_html_url(url: str, keep_structure: bool = False) -> str:
    """
    Fetches an HTML page from a URL and extracts its visible body text.
    Designed to gracefully handle SEC EDGAR 10-K filings.

    :param url: A valid HTTP/HTTPS URL.
    :param keep_structure: If True, chunks text into arbitrary 50-line sections
                           wrapped in `<section_N>` tags to emulate pages.
    """
    headers = {
        "User-Agent": (
            "MarkMyBench/0.1 Benchmarking Research Tool "
            "(Academic Use; contact: asifdotexe@gmail.com)"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")

    if keep_structure:
        # PageIndex logic: Aggressive blank removal, wrapped in 50-line <section_N> nodes
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
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

    else:
        # Traditional RAG logic: Preserve structural whitespace for the LangChain chunker
        lines = raw_text.splitlines()
        cleaned_lines: list[str] = []
        blank_streak = 0
        for line in lines:
            if line.strip():
                blank_streak = 0
                cleaned_lines.append(line.strip())
            else:
                blank_streak += 1
                if blank_streak <= 1:
                    cleaned_lines.append("")

        return "\n".join(cleaned_lines)
