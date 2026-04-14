import gradio as gr
from validator import PromptValidator
import pandas as pd

# Initialize the backend
print("Loading validator backend...")
validator = PromptValidator()

# Custom CSS for premium look
custom_css = """
.container { 
    max-width: 900px; 
    margin: auto; 
    padding: 20px;
    background: #0f172a;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    color: #f8fafc;
}
.header { text-align: center; margin-bottom: 30px; }
.header h1 { font-size: 2.5rem; color: #38bdf8; font-weight: 800; }
.header p { color: #94a3b8; }
.status-card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
}
.status-red { background: #fee2e2; color: #991b1b; border-left: 5px solid #ef4444; }
.status-yellow { background: #fefce8; color: #854d0e; border-left: 5px solid #eab308; }
.status-green { background: #f0fdf4; color: #166534; border-left: 5px solid #22c55e; }
.result-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.result-table th { text-align: left; padding: 10px; color: #94a3b8; border-bottom: 1px solid #1e293b; }
.result-table td { padding: 10px; border-bottom: 1px solid #1e293b; }
"""

def process_validation(prompt):
    if not prompt or len(prompt.strip()) < 5:
        return {
            status_output: "Please enter a valid prompt (min 5 chars).",
            results_area: gr.update(visible=False), 
            btn_proceed: gr.update(interactive=False), 
            btn_try_again: gr.update(interactive=True),
            results_table: []
        }
    
    # Check Exact Match
    if validator.check_exact_match(prompt):
        html_status = f'<div class="status-card status-red">Exact match found — change the prompt</div>'
        return {
            status_output: html_status,
            results_area: gr.update(visible=False), 
            btn_proceed: gr.update(interactive=False), 
            btn_try_again: gr.update(interactive=True),
            results_table: []
        }
    
    # Semantic Search
    print("Executing semantic search...")
    results = validator.check_semantic_similarity(prompt)
    print(f"Search returned {len(results)} results.")
    
    max_similarity = 0
    table_data = []
    for p, distance in results:
        similarity = 1 - distance
        max_similarity = max(max_similarity, similarity)
        table_data.append([f"{similarity:.2%}", p[:150] + "..."])
    
    # Global Warning level
    if max_similarity > 0.85:
        status_class = "status-red"
        status_msg = f"High Similarity Detected ({max_similarity:.2%})"
    elif max_similarity > 0.70:
        status_class = "status-yellow"
        status_msg = f"Medium Similarity Detected ({max_similarity:.2%})"
    else:
        status_class = "status-green"
        status_msg = f"Low Similarity Detected ({max_similarity:.2%})"
    
    html_status = f'<div class="status-card {status_class}">{status_msg}</div>'
    
    try:
        print(f"Returning to UI: status={status_msg}, visible=True, table_rows={len(table_data)}")
        return {
            status_output: html_status,
            results_area: gr.update(visible=True),
            btn_proceed: gr.update(interactive=True),
            btn_try_again: gr.update(interactive=True),
            results_table: table_data
        }
    except Exception as e:
        print(f"ERROR in return: {str(e)}")
        return {status_output: f"UI Update Error: {str(e)}"}

def handle_proceed(prompt):
    validator.add_prompt(prompt)
    return {
        status_output: '<div class="status-card status-green">✅ Prompt successfully saved to database!</div>',
        results_area: gr.update(visible=False),
        btn_proceed: gr.update(interactive=False),
        input_prompt: gr.update(interactive=True, value="")
    }

def handle_reset():
    return {
        status_output: "", 
        results_area: gr.update(visible=False), 
        btn_proceed: gr.update(interactive=False), 
        input_prompt: gr.update(value="", interactive=True),
        results_table: []
    }

with gr.Blocks() as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# 🚀 Prompt Uniqueness Validator")
            gr.Markdown("Ensuring semantic diversity in prompt engineering")
        
        input_prompt = gr.Textbox(
            label="Enter New Prompt", 
            placeholder="Type your prompt here...", 
            lines=4
        )
        
        with gr.Row():
            btn_validate = gr.Button("🔍 Validate Uniqueness", variant="primary")
            btn_try_again = gr.Button("🔄 Try Again", interactive=True)
        
        status_output = gr.HTML("")
        
        with gr.Group(visible=False) as results_area:
            gr.Markdown("### 📊 Top 5 Similar Matches")
            results_table = gr.Dataframe(
                headers=["Similarity", "Existing Prompt Snippet"],
                datatype=["str", "str"],
                interactive=False,
                elem_classes="result-table"
            )
            btn_proceed = gr.Button("✅ Proceed & Save to DB", variant="secondary")

    # Event Handlers
    btn_validate.click(
        fn=process_validation,
        inputs=[input_prompt],
        outputs=[status_output, results_area, btn_proceed, btn_try_again, results_table]
    )
    
    btn_proceed.click(
        fn=handle_proceed,
        inputs=[input_prompt],
        outputs=[status_output, results_area, btn_proceed, input_prompt]
    )
    
    btn_try_again.click(
        fn=handle_reset,
        outputs=[status_output, results_area, btn_proceed, input_prompt, results_table]
    )

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=gr.themes.Soft())
