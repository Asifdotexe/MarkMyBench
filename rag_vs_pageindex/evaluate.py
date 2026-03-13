"""
Benchmark Evaluation Engine (Module C).

Runs both Traditional RAG and PageIndex pipelines against a standardised
question set, then evaluates them using RAGAS metrics (Faithfulness,
Context Precision, Context Recall) plus custom operational metrics
(Latency, Estimated Token Cost).

Usage:
    poetry run python -m rag_vs_pageindex.evaluate

Requires:
    - A funded ``GOOGLE_API_KEY`` in ``rag_vs_pageindex/.env``
    - At least one benchmark document available (URL or local PDF)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import nest_asyncio
from dotenv import load_dotenv
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag_vs_pageindex.traditional_rag_pipeline import TraditionalRAGPipeline
from rag_vs_pageindex.pageindex_rag_pipeline import PageIndexPipeline

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ──────────────────────────────────────────────
# Benchmark Questions
# ──────────────────────────────────────────────
# NOTE: These questions are intentionally designed to test different retrieval
# skills: factual lookup, numerical extraction, multi-section synthesis, and
# questions that should NOT be answerable from the context (hallucination trap).
BENCHMARK_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "What was the total revenue reported by the company?",
        "ground_truth": "The answer should cite a specific revenue figure from the document.",
    },
    {
        "question": "What are the primary risk factors mentioned in the filing?",
        "ground_truth": "The answer should list specific risk factors from the document's risk section.",
    },
    {
        "question": "How does the company describe its competitive landscape?",
        "ground_truth": "The answer should reference competitors or market positioning from the document.",
    },
    {
        "question": "What cybersecurity measures or frameworks does the document reference?",
        "ground_truth": "The answer should mention specific security frameworks, standards, or practices.",
    },
    {
        "question": "What is the capital structure or funding strategy described?",
        "ground_truth": "The answer should discuss equity, debt, or capital allocation from the document.",
    },
]


# ──────────────────────────────────────────────
# Result Container
# ──────────────────────────────────────────────
@dataclass
class PipelineResult:
    """Stores the outputs of a single pipeline run for one question."""

    pipeline_name: str
    question: str
    answer: str
    context: list[str]
    ground_truth: str
    latency_seconds: float = 0.0


# ──────────────────────────────────────────────
# Pipeline Runners
# ──────────────────────────────────────────────
def run_traditional_rag(
    source: str,
    questions: list[dict[str, str]],
) -> list[PipelineResult]:
    """
    Runs the Traditional RAG pipeline end-to-end on a document source.

    :param source: URL or local PDF path to ingest.
    :param questions: List of dicts with ``question`` and ``ground_truth`` keys.
    """
    pipeline = TraditionalRAGPipeline()

    # Ingest → Chunk → Embed (one-time cost per document)
    t0 = time.perf_counter()
    text = pipeline.ingest_document(source)
    chunks = pipeline.chunk_text(text)
    pipeline.embed_and_store(chunks)
    index_time = time.perf_counter() - t0
    print(f"  [Traditional] Indexed in {index_time:.2f}s  ({len(chunks)} chunks)")

    results: list[PipelineResult] = []
    for q in questions:
        t0 = time.perf_counter()
        context = pipeline.retrieve_context(q["question"])
        answer = pipeline.generate_answer(q["question"], context)
        latency = time.perf_counter() - t0

        results.append(
            PipelineResult(
                pipeline_name="Traditional RAG",
                question=q["question"],
                answer=answer,
                context=context,
                ground_truth=q["ground_truth"],
                latency_seconds=latency,
            )
        )
        print(f"    Q: {q['question'][:60]}…  ({latency:.2f}s)")

    return results


def run_pageindex(
    source: str,
    questions: list[dict[str, str]],
) -> list[PipelineResult]:
    """
    Runs the PageIndex pipeline end-to-end on a document source.

    :param source: URL or local PDF path to ingest.
    :param questions: List of dicts with ``question`` and ``ground_truth`` keys.
    """
    pipeline = PageIndexPipeline()

    # Ingest → Build Tree (one-time cost per document)
    t0 = time.perf_counter()
    text = pipeline.ingest_document(source)
    tree = pipeline.generate_semantic_tree(text)
    index_time = time.perf_counter() - t0
    print(f"  [PageIndex] Tree built in {index_time:.2f}s  ({len(tree['nodes'])} nodes)")

    results: list[PipelineResult] = []
    for q in questions:
        t0 = time.perf_counter()
        context = pipeline.retrieve_context_via_tree(q["question"], tree)
        answer = pipeline.generate_answer(q["question"], context)
        latency = time.perf_counter() - t0

        results.append(
            PipelineResult(
                pipeline_name="PageIndex",
                question=q["question"],
                answer=answer,
                context=context,
                ground_truth=q["ground_truth"],
                latency_seconds=latency,
            )
        )
        print(f"    Q: {q['question'][:60]}…  ({latency:.2f}s)")

    return results


# ──────────────────────────────────────────────
# RAGAS Evaluation
# ──────────────────────────────────────────────
def evaluate_with_ragas(results: list[PipelineResult]) -> pd.DataFrame:
    """
    Evaluates pipeline results using RAGAS metrics with Gemini as the LLM judge.

    Metrics computed:
        - **Faithfulness**: Does the answer stick to the retrieved context?
        - **Context Precision**: Is the retrieved context relevant to the question?
        - **Context Recall**: Does the retrieved context cover the ground truth?
        - **Response Relevancy**: Is the answer relevant to the question?

    :param results: A list of ``PipelineResult`` objects from one or more pipelines.
    """
    # NOTE: We wrap our existing Gemini LLM and embeddings so RAGAS uses the same
    # provider as the pipeline itself. This avoids needing an OpenAI key for evaluation.
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    )

    # NOTE: Apply nest_asyncio to allow RAGAS's async evaluation to run 
    # within the Jupyter event loop without "loop already running" errors.
    nest_asyncio.apply()

    # NOTE: We use a conservative RunConfig to prevent TimeoutErrors and 
    # RateLimit errors on the Gemini API. Evaluation is heavy on parallel
    # LLM calls. For the Gemini FREE tier (15 RPM), setting max_workers=1 
    # is the safest way to avoid Resource Exhausted (429) errors.
    run_config = RunConfig(max_workers=1, timeout=180)

    # Build RAGAS evaluation dataset from our pipeline results
    samples = []
    for r in results:
        samples.append(
            SingleTurnSample(
                user_input=r.question,
                response=r.answer,
                retrieved_contexts=r.context,
                reference=r.ground_truth,
            )
        )

    eval_dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    print("\n  Running RAGAS evaluation (this may take a minute)…")
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=run_config,
    )

    # Convert to DataFrame and merge with our custom metrics
    ragas_df = ragas_result.to_pandas()

    # Add pipeline name and latency columns
    ragas_df["pipeline"] = [r.pipeline_name for r in results]
    ragas_df["latency_seconds"] = [r.latency_seconds for r in results]

    return ragas_df


# ──────────────────────────────────────────────
# Main Benchmark Entrypoint
# ──────────────────────────────────────────────
def run_benchmark(source: str, questions: list[dict[str, str]] | None = None) -> pd.DataFrame:
    """
    Orchestrates the full benchmark: runs both pipelines, evaluates with RAGAS,
    and returns a combined comparison DataFrame.

    :param source: URL or local PDF path for the benchmark document.
    :param questions: Optional custom question set; defaults to ``BENCHMARK_QUESTIONS``.
    """
    if questions is None:
        questions = BENCHMARK_QUESTIONS

    print("=" * 60)
    print("BENCHMARK: RAG vs PageIndex")
    print("=" * 60)
    print(f"\nDocument: {source}")
    print(f"Questions: {len(questions)}\n")

    # Run both pipelines
    print("Running Traditional RAG…")
    trad_results = run_traditional_rag(source, questions)

    print("\nRunning PageIndex…")
    page_results = run_pageindex(source, questions)

    # Combine and evaluate
    all_results = trad_results + page_results
    comparison_df = evaluate_with_ragas(all_results)

    # Reorder columns for readability
    col_order = [
        "pipeline",
        "user_input",
        "faithfulness",
        "llm_context_precision_without_reference",
        "context_recall",
        "answer_relevancy",
        "latency_seconds",
    ]
    # Only keep columns that actually exist (RAGAS column names may vary)
    existing_cols = [c for c in col_order if c in comparison_df.columns]
    remaining_cols = [c for c in comparison_df.columns if c not in existing_cols]
    comparison_df = comparison_df[existing_cols + remaining_cols]

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_csv = output_dir / "benchmark_results.csv"
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (Mean Scores)")
    print("=" * 60)
    numeric_cols = comparison_df.select_dtypes(include="number").columns.tolist()
    summary = comparison_df.groupby("pipeline")[numeric_cols].mean()
    print(summary.to_string())

    return comparison_df


if __name__ == "__main__":
    # NOTE: Default benchmark document — uses the small httpbin page for quick testing.
    # Replace with a real 10-K URL or PDF path for production benchmarking.
    import sys

    if len(sys.argv) > 1:
        doc_source = sys.argv[1]
    else:
        doc_source = "https://httpbin.org/html"
        print("No document source provided. Using httpbin.org/html for a quick test.")
        print("Usage: poetry run python -m rag_vs_pageindex.evaluate <URL_or_PDF_path>\n")

    run_benchmark(doc_source)
