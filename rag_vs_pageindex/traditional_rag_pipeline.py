"""
Traditional RAG Pipeline Module.

This module provides the Traditional Retrieval-Augmented Generation (RAG)
approach, including parsing, chunking, embedding, and similarity search logic.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# NOTE: The .env file lives inside rag_vs_pageindex/ (not the repo root) because
# this benchmark is a self-contained module. We resolve the path relative to this
# file so the pipeline can be invoked from any working directory.
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# NOTE: We intentionally keep parsing logic for both PDF and HTML in this
# single module. SEC EDGAR 10-K filings are served as HTML pages, while
# the NIST document is a standard PDF. Rather than forcing the caller to
# pre-process files into a single format, we detect format at runtime —
# this keeps the benchmark self-contained and reduces setup friction.
_SUPPORTED_EXTENSIONS = {".pdf"}
_HTML_URL_PREFIXES = ("http://", "https://")

# File names used under db_path to persist the index and chunk text.
_INDEX_FILE = "faiss.index"
_CHUNKS_FILE = "chunks.pkl"


class TraditionalRAGPipeline:
    """
    Pipeline for executing a Traditional RAG workflow.

    Supports both local PDF files and remote HTML URLs (e.g., SEC EDGAR)
    as document sources. Internally handles format detection, text extraction,
    chunking, embedding, and FAISS-based retrieval.
    """

    def __init__(self, db_path: str = "./faiss_db", model: str = "text-embedding-3-small") -> None:
        """
        Initialises the pipeline and, if a persisted index already exists at
        ``db_path``, loads it into memory to avoid redundant re-embedding.

        :param db_path: Directory where the FAISS index and chunk list are saved.
        :param model: OpenAI embedding model name. Defaults to ``text-embedding-3-small``
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # NOTE: We store both the FAISS index and the raw chunk strings on disk.
        # FAISS only stores float vectors; it has no concept of the original text.
        # The chunk list is the lookup table that maps a FAISS result index back
        # to the actual passage we want to return to the LLM.
        self._embedder = OpenAIEmbeddings(model=model)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[str] = []

        # NOTE: Temperature is set to 0 for full determinism. Benchmark answers
        # must be reproducible across runs — any stochastic variation would make
        # Context Precision / Recall metrics incomparable between pipeline types.
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Warm-start: if a previous run already built the index, reload it
        # so the caller does not have to re-embed the entire corpus.
        index_file = self.db_path / _INDEX_FILE
        chunks_file = self.db_path / _CHUNKS_FILE
        if index_file.exists() and chunks_file.exists():
            self._index = faiss.read_index(str(index_file))
            with open(chunks_file, "rb") as fh:
                self._chunks = pickle.load(fh)

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

    def embed_and_store(self, chunks: list[str]) -> None:
        """
        Generates OpenAI embeddings for the provided text chunks and adds them
        to the in-memory FAISS index, then persists both index and chunk list to disk.

        This method is additive, calling it multiple times with different
        document chunks accumulates all chunks into the same index. This lets
        you index several documents sequentially without rebuilding from scratch.

        Uses ``IndexFlatIP`` (Inner Product) over ``IndexFlatL2`` (Euclidean)
        because OpenAI embeddings are L2-normalised, making inner product
        equivalent to cosine similarity, the metric recommended for text retrieval.

        :param chunks: Non-empty list of text chunks to embed and index.
        :raises ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("Cannot embed an empty chunk list.")

        # NOTE: embed_documents returns a list[list[float]], one vector per chunk.
        # We convert to a float32 numpy array because FAISS requires contiguous
        # float32 arrays (not Python lists or float64).
        raw_vectors = self._embedder.embed_documents(chunks)
        vectors = np.array(raw_vectors, dtype=np.float32)

        # Lazily initialise the FAISS index on the first call, using the actual
        # embedding dimension rather than hardcoding it. This lets us swap models
        # without breaking the index creation logic.
        if self._index is None:
            dimension = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dimension)

        self._index.add(vectors)
        self._chunks.extend(chunks)

        # Persist to disk after every batch so that a partial run is recoverable.
        faiss.write_index(self._index, str(self.db_path / _INDEX_FILE))
        with open(self.db_path / _CHUNKS_FILE, "wb") as fh:
            pickle.dump(self._chunks, fh)

    def retrieve_context(self, query: str, top_k: int = 5) -> list[str]:
        """
        Embeds the query and retrieves the ``top_k`` most relevant chunks from
        the FAISS index using inner-product (cosine) similarity.

        :param query: The natural-language question to search for.
        :param top_k: Number of top-matching chunks to return.
        :raises RuntimeError: If the index is empty (no documents have been indexed yet).
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError(
                "FAISS index is empty. Run embed_and_store() before retrieve_context()."
            )

        query_vector = np.array(
            [self._embedder.embed_query(query)], dtype=np.float32
        )

        # FIXME: Pure cosine similarity may rank verbose chunks higher than
        # precise ones. Consider adding a reciprocal-rank-fusion reranking step
        # once baseline metrics are established.
        _, indices = self._index.search(query_vector, top_k)

        # FAISS returns -1 for unfilled slots when the index has fewer than top_k
        # entries — filter those out to avoid an IndexError on self._chunks.
        return [self._chunks[i] for i in indices[0] if i != -1]

    def generate_answer(self, query: str, context: list[str]) -> str:
        """
        Generates a grounded answer using GPT-4o, strictly based on the provided
        context passages retrieved from the FAISS index.

        The system prompt instructs the model to answer only from the given context
        and to explicitly state when the answer cannot be determined. This reduces
        hallucinations and makes the output more suitable for RAGAS evaluation,
        where faithfulness to retrieved context is a key metric.

        :param query: The natural-language question to answer.
        :param context: Retrieved text passages from the FAISS index.
        :raises ValueError: If ``context`` is empty.
        """
        if not context:
            raise ValueError("Cannot generate an answer without context passages.")

        # NOTE: We join context chunks with a numbered list rather than a single
        # blob of text. This helps the model distinguish between separate passages
        # and reduces the risk of it blending unrelated facts into one sentence.
        formatted_context = "\n\n".join(
            f"[{i + 1}] {chunk}" for i, chunk in enumerate(context)
        )

        messages = [
            (
                "system",
                (
                    "You are a precise research assistant. "
                    "Answer the user's question using ONLY the context passages provided below. "
                    "If the answer cannot be found in the context, respond with: "
                    "'I could not find a relevant answer in the provided context.' "
                    "Do not speculate or add information beyond what is in the context."
                ),
            ),
            (
                "human",
                f"Context passages:\n{formatted_context}\n\nQuestion: {query}",
            ),
        ]

        response = self._llm.invoke(messages)
        return response.content
