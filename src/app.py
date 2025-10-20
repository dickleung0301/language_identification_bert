import torch
import gradio as gr
from model import LanguageIdentifier
    
# initialize mbert
lang_id_mbert = LanguageIdentifier()

examples = [
    ["Hello, how are you today? This is a sample English text."],
    ["Hola, ¿cómo estás hoy? Este es un texto de ejemplo en español."],
    ["Bonjour, comment allez-vous aujourd'hui? Ceci est un exemple de texte français."],
    ["Hallo, wie geht es dir heute? Dies ist ein deutscher Beispieltext."],
    ["Ciao, come stai oggi? Questo è un testo di esempio in italiano."],
    ["こんにちは、今日はお元気ですか？これは日本語のサンプルテキストです。"],
    ["你好，你今天好吗？这是一个中文示例文本。"],
]

# create a gradio interface with blocks
with gr.Blocks(
    title="Language Identification",
    theme=gr.themes.Soft(),
    css="""
    .language-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
) as demo:
    
    # header
    with gr.Row():
        gr.Markdown(
            """
            <div class="language-header">
                <h1>🌍 Language Identification System</h1>
                <p>Identify the language of text in supported language using BERT-based model</p>
            </div>
            """,
            elem_classes="language-header"
        )

    # main content
    with gr.Row():
        # left column - input
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input Text")

            text_input = gr.Textbox(
                label="Enter text to identify",
                placeholder="Type or paste any text here...",
                lines=8,
                max_lines=15
            )

            with gr.Row():
                submit_btn = gr.Button("🔍 Identify Language", variant="primary", scale=2)
                clear_btn = gr.ClearButton(components=[text_input], value="🗑️ Clear", scale=1)

            with gr.Accordion("⚙️ Advanced Options", open=False):
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of top predictions to show"
                )

                show_probs = gr.Checkbox(
                    label="Show probability chart",
                    value=True
                )

        # right column - output
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")

            result_text = gr.Markdown(
                label="Detected Result",
                value="Results will appear here..."
            )

            confidence_plot = gr.Label(
                label="Confidence Scores"
            )

            with gr.Accordion("ℹ️ About", open=False):
                gr.Markdown(
                    """
                    **Model Information:**
                    - Fine-tuned multilingual-BERT with LoRA adpaters
                    - Trained on multilingual language identification data
                    - Support 20 languages

                    **How to use:**
                    1. Enter text in the input box
                    2. Click 'Identify Language' button
                    3. View the detected language and confidence score
                    """
                )

    # examples
    gr.Markdown("### 💡 Try These Examples")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=[confidence_plot, result_text],
        fn=lang_id_mbert.predict,
        cache_examples=False,
        label="Examples in different languages"
    )

    # statistics 
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📈 Model Statistics")
            stats = gr.DataFrame(
                value=[
                    ["# of Supported Languages", "20"],
                    ["Base Model", "Multilingual-BERT"],
                    ["Max Input Length", "512 Tokens"],
                    ["Device", "GPU" if torch.cuda.is_available() else "CPU"]
                ],
                headers=["Metric", "Value"],
                label="System Info"
            )

    # event handler
    def predict_with_options(text, top_k_val, use_advanced_mode):
    
        results, text_result = lang_id_mbert.predict(text, top_k=int(top_k_val))

        if not use_advanced_mode:
            results = {}

        return results, text_result

    submit_btn.click(
        fn=predict_with_options,
        inputs=[text_input, top_k, show_probs],
        outputs=[confidence_plot, result_text]
    )

    # footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666">
            <p>• Powered by Hugging Face Transformers</p>
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True
    )