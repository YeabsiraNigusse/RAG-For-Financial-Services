

import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import sys

sys.path.insert(0, os.path.abspath('..'))

# -------------------------------
# Step 1: Load Filtered Data
# -------------------------------
df = pd.read_csv("data/filtered_complaints.csv")

# # ✅ Optional: Limit number of rows for testing
# df = df[:500]  # Remove or increase for full processing

# -------------------------------
# Step 2: Chunk Narratives
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs = []
chunk_records = []  # <-- used for saving to CSV

for _, row in df.iterrows():
    chunks = splitter.split_text(row["Cleaned Narrative"])
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "complaint_id": row["Complaint ID"],
                "product": row["Product"],
                "chunk_index": i
            }
        ))
        chunk_records.append({
            "complaint_id": row["Complaint ID"],
            "product": row["Product"],
            "chunk_index": i,
            "chunk_text": chunk
        })

print(f"✅ Total Chunks Created: {len(docs)}")

# -------------------------------
# Step 3: Save Chunks to CSV
# -------------------------------
chunk_df = pd.DataFrame(chunk_records)
os.makedirs("data", exist_ok=True)
chunk_df.to_csv("data/chunked_complaints.csv", index=False)
print("✅ Chunked data saved to data/chunked_complaints.csv")

# -------------------------------
# Step 4: Load Embedding Model
# -------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# Step 5: Create Vector Store
# -------------------------------
persist_dir = "vector_store/chroma_index"
if os.path.exists(persist_dir):
    import shutil
    shutil.rmtree(persist_dir)

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

vector_db.persist()
print(f"✅ Vector store created and saved to: {persist_dir}")
