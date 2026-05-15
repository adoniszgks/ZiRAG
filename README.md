# ZIRAG

ZIRAG aims to extend classical Retrieval-Augmented Generation by supporting both 
multimodal document corpora and multimodal user queries.

---

## Requirements

- Python 3.10+
- `colpali-engine`
- `pymupdf`
- `torch`
- `chromadb`

> **Note:** A CUDA-capable GPU is **strongly recommended**. Inference with ColQwen2 is 
very slow on CPU.

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
# For CUDA (recommended):
pip install colpali-engine pymupdf chromadb
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install colpali-engine pymupdf chromadb torch
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Multimodal Retrieval | ColQwen2 (`vidore/colqwen2-v1.0`) |
| PDF Processing | PyMuPDF (`fitz`) |
| Image Processing | Pillow (PIL) |
| ML Framework | PyTorch |
| Vector Database | ChromaDB |
| Lexical Retrieval | BM25S *(planned)* |
| LLM Answer Generation | Claude API / OpenAI *(planned)* |
| Chat Interface | Gradio *(planned)* |

