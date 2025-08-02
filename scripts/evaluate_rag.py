import os
import pandas as pd
import sys

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath('.'))

from src.rag_pipeline import answer_question  # ✅ use your existing pipeline

# -------------------------------
# Step 1: Define Test Questions
# -------------------------------
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

# -------------------------------
# Step 2: Evaluate RAG Pipeline
# -------------------------------
results = []

for question in test_questions:
    output = answer_question(question, k=5)

    # Get top 2 chunks for the report
    top_sources = [
        f"Product: {src['product']} | Complaint ID: {src['complaint_id']}"
        for src in output["retrieved_sources"][:2]
    ]

    results.append({
        "Question": question,
        "Generated Answer": output["answer"],
        "Retrieved Sources": "\n---\n".join(top_sources),
        "Quality Score (1-5)": "",  # Leave blank for manual scoring
        "Comments": ""              # Leave blank for manual notes
    })

# -------------------------------
# Step 3: Export Evaluation CSV
# -------------------------------
os.makedirs("reports", exist_ok=True)
output_file = "reports/evaluation_output.csv"
df_eval = pd.DataFrame(results)
df_eval.to_csv(output_file, index=False)

print(f"✅ Evaluation complete. Results saved to: {output_file}")
