# RAG-For-Financial-Services

- A Retrieval-Augmented Generation (RAG) system designed for analyzing financial customer complaints.
The system retrieves relevant complaint excerpts from a vector database (Chroma) and generates AI-driven responses using Google Gemini models.

## Project Structure

        RAG-For-Financial-Services/
        │
        ├── data/                      # CSV datasets
        │   ├── complaints.csv
        │   ├── filtered_complaints.csv
        │   └── chunked_complaints.csv
        │
        ├── vector_store/              # Persisted Chroma vector DB
        │
        ├── src/                       # Core pipelines
        │   ├── embedding_pipeline.py  # Preprocessing & embedding generation
        │   ├── rag_pipeline.py        # RAG question-answering pipeline
        │   └── __init__.py
        │
        ├── scripts/                   # Evaluation and utility scripts
        │   └── evaluate_rag.py
        │
        ├── reports/                   # Evaluation outputs
        │   └── evaluation_output.csv
        │
        ├── screenshots/               # Screenshots for documentation
        │
        ├── app.py                     # Gradio web application for interactive chat
        ├── requirements.txt           # Project dependencies
        ├── .env                       # Environment variables (Gemini API key)
        ├── .gitignore                 # Ignored files/folders
        ├── LICENSE
        └── README.md                  # Project documentation


### Features
- RAG Pipeline
    - Combines vector similarity search (Chroma) and Google Gemini LLM for contextual answers.

- Embedding Pipeline
    - Preprocesses complaints into chunked embeddings using sentence-transformers/all-MiniLM-L6-v2.

- Interactive Chat Interface
    - Built with Gradio, allowing non-technical users to ask questions and view sources.

- Evaluation Framework
    - Script for batch-testing and generating manual scoring reports.

- Source Transparency
    - Displays complaint excerpts used to generate each answer.

## Setup Instructions
```
    git clone git@github.com:YeabsiraNigusse/RAG-For-Financial-Services.git
    cd RAG-For-Financial-Services
```
## Create Virtual Environment
```    python3 -m venv .venv
    source .venv/bin/activate   # Linux / Mac
    .venv\Scripts\activate      # Windows
```
##  Install Dependencies
    ```pip install -r requirements.txt```

## Set Up Environment Variables

- GEMINI_API_KEY=your_google_gemini_api_key

# Usage
### 1. Generate Embeddings and Build Vector Store
- Run the embedding pipeline to preprocess complaints:
```
python src/embedding_pipeline.py
```
- Loads filtered_complaints.csv
- Chunks narratives
- Saves chunked_complaints.csv
- Persists embeddings in vector_store/

### 2. Test the RAG Pipeline
- Run an example query:

```
python src/rag_pipeline.py
```
Expected output:
```
Question:
Why are customers frustrated with credit reporting issues?

Generated Answer:
[LLM-generated answer]

Context (top 2 chunks):
- Product: Credit reporting | ID: 123456
- Product: Credit reporting | ID: 789012
```
### 3. Launch Interactive Chat App
- Run the Gradio interface:

```python app.py```
- Open the link in your browser (default: http://127.0.0.1:7860).

### 4. Evaluate the System
- Run the evaluation script to generate a CSV for manual scoring:
    ```
        python scripts/evaluate_rag.py
        Output: reports/evaluation_output.csv
    ```
Example columns:
``` 
Question	Generated Answer	Retrieved Sources	Quality Score (1-5)	Comments 
```

### Key Dependencies
- LangChain (langchain-community, langchain-chroma, langchain-huggingface)

- Gradio (UI)

- Pandas (Data processing)

- Sentence-Transformers (Embeddings)

- Google Generative AI Python SDK (google-generativeai)

