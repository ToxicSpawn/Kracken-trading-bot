"""
Gradio web UI for LLM inference.
Provides a user-friendly interface for testing and interacting with models.
"""
import os
import logging
from typing import Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from ..llm.fine_tuned_model import FineTunedLLM
from ..rag.retrieval_system import RAGSystem
from ..security.guardrails import SecurityGuard, PromptInjectionGuard

logger = logging.getLogger(__name__)

if not GRADIO_AVAILABLE:
    logger.warning("Gradio not available. Install with: pip install gradio")


class GradioUI:
    """Gradio-based web UI for LLM inference."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        enable_rag: bool = True,
    ):
        """Initialize Gradio UI."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio not installed. Install with: pip install gradio")

        self.model_name = model_name or os.getenv(
            "LLM_MODEL_NAME",
            "mistralai/Mistral-7B-v0.1"
        )
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH")
        self.enable_rag = enable_rag

        # Initialize components
        self.llm: Optional[FineTunedLLM] = None
        self.rag: Optional[RAGSystem] = None
        self.security_guard = SecurityGuard()
        self.injection_guard = PromptInjectionGuard()

    def _load_model(self):
        """Lazy load model."""
        if self.llm is None:
            try:
                self.llm = FineTunedLLM(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    use_quantization=True,
                )
                logger.info("Model loaded for Gradio UI")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def _load_rag(self):
        """Lazy load RAG."""
        if self.enable_rag and self.rag is None:
            try:
                rag_dir = os.getenv("RAG_PERSIST_DIR", "rag_store")
                self.rag = RAGSystem(persist_directory=rag_dir, vectorstore_type="faiss")
                logger.info("RAG system loaded for Gradio UI")
            except Exception as e:
                logger.warning(f"RAG not available: {e}")

    def generate_text(self, prompt: str, use_rag: bool, max_length: int, temperature: float) -> str:
        """Generate text from prompt."""
        # Security checks
        validation = self.security_guard.validate(prompt)
        if not validation["valid"]:
            return f"❌ Security violation: {', '.join(validation['violations'])}"

        if self.injection_guard.detect(prompt):
            return "❌ Potential prompt injection detected. Please rephrase your query."

        prompt = validation["sanitized"]

        try:
            self._load_model()

            # Use RAG if enabled
            if use_rag and self.enable_rag:
                self._load_rag()
                if self.rag:
                    # Add RAG context
                    relevant_docs = self.rag.similarity_search(prompt, k=3)
                    if relevant_docs:
                        context = "\n\nRelevant context:\n"
                        context += "\n".join([doc.page_content[:300] for doc in relevant_docs])
                        prompt = f"{prompt}\n{context}"

            # Generate
            response = self.llm.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
            )

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"❌ Error: {str(e)}"

    def rag_query(self, query: str, k: int) -> str:
        """Query RAG system."""
        if not self.enable_rag:
            return "RAG is not enabled"

        try:
            self._load_rag()
            if not self.rag:
                return "RAG system not available"

            docs = self.rag.similarity_search(query, k=k)
            if not docs:
                return "No relevant documents found"

            result = "📚 Found documents:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"{i}. Source: {doc.metadata.get('source', 'unknown')}\n"
                result += f"   {doc.page_content[:200]}...\n\n"

            return result

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"❌ Error: {str(e)}"

    def launch(self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
        """Launch Gradio interface."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio not installed")

        # Create interface
        with gr.Blocks(title="LLM Inference UI") as demo:
            gr.Markdown("# 🤖 Fine-Tuned LLM Inference Interface")

            with gr.Tabs():
                # Text generation tab
                with gr.Tab("Text Generation"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter your prompt here...",
                                lines=5,
                            )
                            with gr.Row():
                                use_rag_checkbox = gr.Checkbox(
                                    label="Use RAG",
                                    value=self.enable_rag,
                                )
                                max_length_slider = gr.Slider(
                                    minimum=50,
                                    maximum=2048,
                                    value=512,
                                    step=50,
                                    label="Max Length",
                                )
                                temperature_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature",
                                )
                            generate_btn = gr.Button("Generate", variant="primary")
                        with gr.Column(scale=2):
                            output_text = gr.Textbox(
                                label="Generated Text",
                                lines=10,
                            )

                    generate_btn.click(
                        fn=self.generate_text,
                        inputs=[prompt_input, use_rag_checkbox, max_length_slider, temperature_slider],
                        outputs=output_text,
                    )

                # RAG query tab
                if self.enable_rag:
                    with gr.Tab("RAG Query"):
                        with gr.Row():
                            with gr.Column():
                                rag_query_input = gr.Textbox(
                                    label="Query",
                                    placeholder="Enter your query...",
                                    lines=3,
                                )
                                k_slider = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=5,
                                    step=1,
                                    label="Number of documents (k)",
                                )
                                rag_query_btn = gr.Button("Search", variant="primary")
                            with gr.Column():
                                rag_output = gr.Textbox(
                                    label="Results",
                                    lines=10,
                                )

                        rag_query_btn.click(
                            fn=self.rag_query,
                            inputs=[rag_query_input, k_slider],
                            outputs=rag_output,
                        )

        # Launch
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
        )


def launch_ui(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    enable_rag: bool = True,
    port: int = 7860,
):
    """Convenience function to launch UI."""
    ui = GradioUI(model_name=model_name, model_path=model_path, enable_rag=enable_rag)
    ui.launch(server_port=port)


if __name__ == "__main__":
    launch_ui()

