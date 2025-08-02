# src/rag_pipeline.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, os.path.abspath('..'))
# --- Load .env file ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# --- Configure Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Load Chroma Vector Store ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_dir = "vector_store/chroma_index"

vector_db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model
)

# --- Retrieve Top-k Chunks ---
def retrieve_chunks(query: str, k: int = 5):
    docs = vector_db.similarity_search(query, k=k)
    return docs

# --- Generate Answer using Gemini ---
def generate_answer_with_gemini(context: str, question: str) -> str:
    prompt_template = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.

If the context doesn't contain the answer, say:
"I don't have enough information to answer that."

--- Context ---
{context}

--- Question ---
{question}

--- Answer ---
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt_template)
    return response.text.strip()

# --- RAG Pipeline Flow ---
def answer_question(question: str, k: int = 5):
    docs = retrieve_chunks(question, k)
    context = "\n---\n".join([doc.page_content for doc in docs])
    answer = generate_answer_with_gemini(context, question)
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved_sources": [doc.metadata for doc in docs]
    }
test_questions = [
    "What is the most common issue with credit reporting?",
    "Do customers complain about loan forgiveness issues?",
    "What happens when a credit card account is closed?",
    "Are there delays in processing student loans?",
    "Is there a common problem with mortgage billing errors?",
    "What issues arise from identity theft in complaints?",
    "Do people report problems with dispute resolution?",
    "How do consumers describe poor customer service?",
]

# --- Example Usage ---
if __name__ == "__main__":
    sample_question = "Why are customers frustrated with credit reporting issues?"
    result = answer_question(sample_question)

    print("\nQuestion:")
    print(result["question"])
    print("\nGenerated Answer:")
    print(result["answer"])
    print("\nContext (top 2 chunks):")
    for chunk in result["retrieved_sources"][:2]:
        print(f"- Product: {chunk['product']} | ID: {chunk['complaint_id']}")

