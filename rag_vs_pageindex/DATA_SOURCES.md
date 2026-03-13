# Benchmark Data Sources

This document records the exact source files used in the RAG vs. PageIndex benchmark.
Anyone reproducing the results must use these same documents to ensure a fair comparison.

> **Parsing Note:** The 10-K filings link to SEC EDGAR's official HTML format.
> The `unstructured` library can parse these directly, or you may save them as PDF
> via your browser's "Save as PDF" to standardize ingestion across all three files.
> Do not substitute with third-party PDF re-exports, as formatting differences may
> affect chunking and retrieval quality.

---

## Documents

| # | Document | Type | Source |
|---|----------|------|--------|
| 1 | Rocket Lab USA 2024 Annual Report (10-K) | SEC Filing (HTML) | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=RKLB&type=10-K&dateb=&owner=include&count=10) |
| 2 | NIST Cybersecurity Framework 2.0 | Official PDF | [NIST](https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf) |
| 3 | Oracle Corporation FY2024 Annual Report (10-K) | SEC Filing (HTML) | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=ORCL&type=10-K&dateb=&owner=include&count=10) |

---

## Local Storage Convention

Once downloaded, place raw files into:

```
rag_vs_pageindex/
└── data/
    ├── rocket_lab_10k_2024.pdf
    ├── nist_csf_2_0.pdf
    └── oracle_10k_fy2024.pdf
```

> **Note:** The `data/` directory is `.gitignore` to avoid committing large binary files.
> Always re-download from the source links above if you are cloning this repo fresh.
