"""
Traditional RAG Pipeline Module.

This module provides the Traditional Retrieval-Augmented Generation (RAG)
approach, including parsing, chunking, embedding, and similarity search logic.
"""

from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# NOTE: We intentionally keep parsing logic for both PDF and HTML in this
# single module. SEC EDGAR 10-K filings are served as HTML pages, while
# the NIST document is a standard PDF. Rather than forcing the caller to
# pre-process files into a single format, we detect format at runtime —
# this keeps the benchmark self-contained and reduces setup friction.
_SUPPORTED_EXTENSIONS = {".pdf"}
_HTML_URL_PREFIXES = ("http://", "https://")


class TraditionalRAGPipeline:
    """
    Pipeline for executing a Traditional RAG workflow.

    Supports both local PDF files and remote HTML URLs (e.g., SEC EDGAR)
    as document sources. Internally handles format detection, text extraction,
    chunking, embedding, and FAISS-based retrieval.
    """

    def __init__(self, db_path: str = "./faiss_db") -> None:
        """
        Initializes the pipeline with necessary configurations.

        :param db_path: Path to the local FAISS vector database index file.
        """
        self.db_path = db_path
        # TODO: Initialize embedding model and FAISS index client here.

    def ingest_document(self, source: str) -> str:
        """
        Ingests a document from either a local PDF path or a remote HTML URL
        and returns the extracted plain text.

        Routes to the appropriate parser based on input type:
        - Local ``*.pdf`` files  → ``pdfplumber``
        - HTTP/HTTPS URLs        → ``requests`` + ``BeautifulSoup``

        :param source: A local file path (PDF) or a full HTTP/HTTPS URL pointing
                       to an HTML document (e.g., an SEC EDGAR filing page).
        :raises FileNotFoundError: If a local path is given but does not exist.
        :raises ValueError: If the file extension is unsupported for local paths.
        :raises requests.HTTPError: If the HTTP request for a URL source fails.
        """
        if source.startswith(_HTML_URL_PREFIXES):
            return self._parse_html_url(source)

        # Treat anything that is not a URL as a local filesystem path.
        return self._parse_pdf(source)

    def _parse_pdf(self, file_path: str) -> str:
        """
        Extracts plain text from a local PDF file using ``pdfplumber``.

        ``pdfplumber`` is chosen over alternatives (e.g., ``pypdf``) because
        it handles complex PDF layouts — including multi-column text and embedded
        tables found in financial 10-K filings — significantly more reliably.

        :param file_path: Absolute or relative path to the ``.pdf`` file.
        :raises FileNotFoundError: If the file does not exist at the given path.
        :raises ValueError: If the file does not have a ``.pdf`` extension.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found at path: '{file_path}'")

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Only the following extensions are supported: {_SUPPORTED_EXTENSIONS}"
            )

        # NOTE: We join page text with double newlines to preserve section
        # boundaries between pages. This matters during chunking — a naive
        # single-space join would bleed unrelated paragraphs into the same chunk.
        extracted_pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())

        return "\n\n".join(extracted_pages)

    def _parse_html_url(self, url: str) -> str:
        """
        Fetches an HTML page from a URL and extracts its visible body text.

        Designed for SEC EDGAR HTML filings, which expose the full 10-K document
        as a single enriched HTML page. BeautifulSoup with the ``lxml`` backend
        is used for speed and robust handling of malformed HTML tags that are
        common in the older EDGAR filing format.

        :param url: A valid HTTP or HTTPS URL pointing to an HTML document.
        :raises requests.HTTPError: If the server returns a non-2xx status code.
        """
        # NOTE: We set a realistic browser User-Agent here to avoid SEC EDGAR's
        # bot-detection returning a 403 Forbidden response. This is a standard
        # courtesy header, not a deceptive practice.
        headers = {
            "User-Agent": (
                "MarkMyBench/0.1 Benchmarking Research Tool "
                "(Academic Use; contact: asifdotexe@gmail.com)"
            )
        }

        response = requests.get(url, headers=headers, timeout=30)

        # Raise immediately on any HTTP error (4xx, 5xx) so the caller
        # gets an explicit error rather than silently parsing an error page.
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")

        # Remove non-content tags before extracting text to avoid polluting
        # the corpus with navigation menus, scripts, and inline styles.
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # get_text() with a separator preserves line breaks from block elements.
        raw_text = soup.get_text(separator="\n")

        # Collapse excessive blank lines (4+ consecutive) into a single blank line.
        # Financial HTML filings often have large whitespace gaps around tables.
        lines = raw_text.splitlines()
        cleaned_lines: list[str] = []
        blank_streak = 0
        for line in lines:
            if line.strip():
                blank_streak = 0
                cleaned_lines.append(line.strip())
            else:
                blank_streak += 1
                # Allow a maximum of one blank separator line between sections.
                if blank_streak <= 1:
                    cleaned_lines.append("")

        return "\n".join(cleaned_lines)

    def chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """
        Splits raw document text into overlapping chunks suitable for embedding.

        Uses LangChain's ``RecursiveCharacterTextSplitter`` with a natural-language
        separator hierarchy: paragraph → line → word → character. This preserves
        semantic coherence as long as possible before falling back to hard cuts,
        which is critical for financial documents where sentences within a paragraph
        often share critical context (e.g., risk factors, footnotes).

        :param text: The raw text extracted from the document.
        :param chunk_size: Maximum number of characters per chunk.
        :param overlap: Number of overlapping characters between consecutive chunks.
                        Overlap prevents losing context that straddles a chunk boundary
                        (e.g., a key term defined at the end of one chunk being referenced
                        at the start of the next).
        :raises ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty or whitespace-only text.")

        # NOTE: The separator list is ordered from coarsest to finest granularity.
        # The splitter works left-to-right: it tries each separator in turn and
        # only moves to a finer split if the current chunk still exceeds chunk_size.
        # Double-newline (paragraph) → newline (line) → space (word) → "" (character).
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            # "\n\n"  →  "\n"  →  " "  →  ""
            # paragraph     line    word   char
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_text(text)

        # NOTE: Filter out any chunks that are effectively empty after splitting.
        # This can happen at the tail end of documents with trailing whitespace.
        return [chunk for chunk in chunks if chunk.strip()]

    def embed_and_store(self, chunks: list[str]) -> bool:
        """
        Generates embeddings for the provided chunks and stores them in the vector database.

        :param chunks: A list of text chunks to be embedded.
        """
        # NOTE: Storing vectors locally (e.g., FAISS) reduces latency and API costs
        # during frequent benchmarking runs, improving reproducibility.
        # TODO: Implement embedding generation and FAISS index storage logic.
        pass

    def retrieve_context(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieves the most relevant document chunks based on cosine similarity to the query.

        :param query: The user query to search for.
        :param top_k: The number of topmost relevant chunks to return.
        """
        # FIXME: Context retrieval might need reranking if standard cosine similarity
        # yields poor results for highly nuanced queries.
        # TODO: Implement cosine similarity search via FAISS index.
        pass

    def generate_answer(self, query: str, context: list[str]) -> str:
        """
        Generates an answer using an LLM based on the provided context.

        :param query: The user query.
        :param context: The context chunks retrieved from the vector database.
        """
        # TODO: Implement LLM generation logic.
        pass
