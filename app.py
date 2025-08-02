import gradio as gr
from src.rag_pipeline import answer_question

# --- Chat Function ---
def chat_with_rag(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    result = answer_question(question, k=5)
    
    # Prepare sources as formatted string
    sources = "\n".join(
        [f"- Product: {src['product']} | Complaint ID: {src['complaint_id']}" 
         for src in result["retrieved_sources"]]
    )
    
    return result["answer"], sources

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 📊 CrediTrust RAG Chatbot")
    gr.Markdown("Ask a question about customer complaints and see the AI answer with sources.")

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Enter your question",
                placeholder="e.g., Why are customers frustrated with credit reporting issues?",
                lines=2
            )
            submit_btn = gr.Button("Ask")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=5):
            answer_output = gr.Textbox(label="AI Answer", lines=6)
            sources_output = gr.Textbox(label="Sources Used", lines=6)

    # Button actions
    submit_btn.click(
        fn=chat_with_rag,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[answer_output, sources_output]
    )
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=question_input
    )

# --- Launch App ---
if __name__ == "__main__":
    demo.launch()
