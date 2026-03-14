import uuid
import gradio as gr

from app.ui.chat_interface import sync_run_pipeline, PIPELINE_STAGES, AGENT_LABELS

EXAMPLE_PROMPTS = [
    "Write a blog post about getting started with async Python for backend developers",
    "Create a comprehensive guide on building a personal finance budget in 2026",
    "Write an article about the top 5 productivity habits for remote software engineers",
    "Create content about how to start a vegetable garden for beginners",
    "Write a post about the benefits of mindfulness meditation for busy professionals",
]

PIPELINE_MD = "\n".join(
    f"{i + 1}. {AGENT_LABELS[stage]}" for i, stage in enumerate(PIPELINE_STAGES)
)

# Gradio 6: theme/css go to launch(), not Blocks()
THEME = gr.themes.Soft()
CSS = """
    footer { display: none !important; }
"""


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Content Creation Studio") as demo:
        gr.Markdown(
            "# AI Content Creation Studio\n"
            "Describe your content goal below. The 6-agent pipeline will handle ideation, "
            "outlining, writing, SEO, image prompts, and final packaging automatically."
        )

        session_id = gr.State(lambda: str(uuid.uuid4()))

        with gr.Row():
            # ── Main chat column ────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=620,
                    show_label=False,
                    render_markdown=True,
                    placeholder=(
                        "Your content pipeline results will appear here.\n\n"
                        "Each agent's output is shown as a collapsible step."
                    ),
                )

                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="E.g. Write a blog post about async Python for beginners...",
                        show_label=False,
                        scale=9,
                        lines=1,
                        max_lines=3,
                    )
                    submit_btn = gr.Button("Run Pipeline", variant="primary", scale=1, min_width=120)

                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=txt,
                    label="Example prompts",
                )

            # ── Sidebar ─────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### Pipeline Stages")
                gr.Markdown(PIPELINE_MD)

                gr.Markdown("---")
                gr.Markdown("### How It Works")
                gr.Markdown(
                    "1. Enter a content brief\n"
                    "2. Each agent processes sequentially\n"
                    "3. Click an agent step to expand it\n"
                    "4. Final output is a JSON package ready for publishing"
                )

                gr.Markdown("---")
                session_display = gr.Textbox(
                    label="Session ID",
                    interactive=False,
                    max_lines=1,
                )

        # ── Event wiring ────────────────────────────────────────────────
        def _on_submit(msg, history, sid):
            yield from sync_run_pipeline(msg, history, sid)

        submit_btn.click(
            fn=_on_submit,
            inputs=[txt, chatbot, session_id],
            outputs=[chatbot],
        )
        txt.submit(
            fn=_on_submit,
            inputs=[txt, chatbot, session_id],
            outputs=[chatbot],
        )

        # Show session ID on load
        demo.load(
            fn=lambda sid: sid,
            inputs=[session_id],
            outputs=[session_display],
        )

    return demo
