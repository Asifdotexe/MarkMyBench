"""
PageIndex Pipeline Module.

This module provides the foundation for the PageIndex (Vectorless RAG)
approach. It processes documents into hierarchical semantic trees and
utilizes LLM-driven navigation for context extraction.
"""
import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# NOTE: Load .env explicitly from the benchmark directory
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

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
        
        Delegates to the shared `document_parsers` module with `keep_structure=True`.

        :param source: A local file path (PDF) or a full HTTP/HTTPS URL (HTML).
        """
        from rag_vs_pageindex.document_parsers import ingest_document as parser_ingest
        return parser_ingest(source, keep_structure=True)

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

        Instead of vector similarity, the LLM reads a compact "Table of Contents"
        built from the node summaries and decides which sections are most likely
        to contain the answer. This is the core differentiator of the PageIndex
        approach: retrieval is driven by reasoning, not by embedding distance.

        :param query: The user query to evaluate against the tree.
        :param tree: The hierarchical semantic tree representation of the document.
        :raises RuntimeError: If the tree is empty (no documents have been indexed yet).
        """
        nodes = tree.get("nodes", [])
        if not nodes:
            raise RuntimeError(
                "Semantic tree is empty. Run generate_semantic_tree() first."
            )

        # NOTE: We deliberately exclude the heavy raw `content` from this prompt.
        # Sending only the summaries keeps the token count low and forces the LLM
        # to make a routing decision based on high-level topic relevance, which is
        # the fundamental idea behind PageIndex.
        toc_lines = [
            f"[{node['id']}] {node['summary']}" for node in nodes
        ]
        toc_text = "\n".join(toc_lines)

        messages = [
            (
                "system",
                (
                    "You are a document retrieval router. You will be given a Table of "
                    "Contents where each entry has an integer ID and a one-sentence summary. "
                    "Your task is to select the IDs of the sections most likely to contain "
                    "information relevant to the user's question.\n\n"
                    "Rules:\n"
                    "- Return ONLY a JSON array of integer IDs, e.g. [1, 4, 7].\n"
                    "- Select between 1 and 5 sections (prefer fewer if the question is narrow).\n"
                    "- Do NOT include any explanation, markdown, or extra text."
                ),
            ),
            (
                "human",
                f"Table of Contents:\n{toc_text}\n\nQuestion: {query}",
            ),
        ]

        response = self._llm.invoke(messages).content.strip()

        # NOTE: The LLM sometimes wraps the JSON in a markdown code-fence.
        # We strip that defensively before parsing.

        cleaned = response
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        selected_ids: list[int] = json.loads(cleaned)

        # Build a fast lookup from node ID -> full content text.
        id_to_content = {node["id"]: node["content"] for node in nodes}

        # FIXME: LLM-driven navigation might become a bottleneck under heavy loads.
        # Consider implementing caching strategies for repetitive or similar queries.
        return [
            id_to_content[nid]
            for nid in selected_ids
            if nid in id_to_content
        ]

    def generate_answer(self, query: str, context: list[str]) -> str:
        """
        Generates a grounded answer using Gemini 2.5 Flash, strictly based on
        the context sections extracted via tree navigation.

        The system prompt mirrors the Traditional RAG pipeline's generation step
        to ensure a fair, apples-to-apples benchmarking comparison between the
        two retrieval strategies.

        :param query: The user query.
        :param context: The context strings extracted via tree navigation.
        :raises ValueError: If ``context`` is empty.
        """
        if not context:
            raise ValueError("Cannot generate an answer without context passages.")

        # NOTE: We number the context passages so the model can distinguish between
        # separate sections and avoid blending unrelated facts into one sentence.
        formatted_context = "\n\n".join(
            f"[{i + 1}] {passage}" for i, passage in enumerate(context)
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
