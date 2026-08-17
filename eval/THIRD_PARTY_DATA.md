# Third-Party Evaluation Data

PunditRAG includes evaluation adapters and download scripts for the datasets
listed below. Their raw data is not distributed with this repository and is
not covered by PunditRAG's MIT License.

| Dataset | Upstream repository | Local download script |
|---|---|---|
| RGB | <https://github.com/chen700564/RGB> | `python eval/download_rgb.py` |
| CRUD-RAG | <https://github.com/IAAR-Shanghai/CRUD_RAG> | `python eval/download_crud.py` |
| MTRAG | <https://github.com/IBM/mt-rag-benchmark> | `python eval/download_mtrag.py` |

The scripts place downloaded files under `eval/raw/`, which is intentionally
excluded from Git. Before downloading, using, or redistributing a dataset,
review its upstream license, terms, and any restrictions that apply to the
underlying source material. Upstream terms remain authoritative.

The small fixtures under `eval/datasets/documents/` were written specifically
for PunditRAG and are distributed under this repository's MIT License. They are
synthetic test documents, not official product manuals.
