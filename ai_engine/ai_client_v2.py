"""
Enhanced AI client using fine-tuned LLMs and RAG.
Replaces the OpenAI-based ai_client.py with local fine-tuned models.
"""
import os
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from .llm.fine_tuned_model import FineTunedLLM
from .rag.retrieval_system import RAGSystem

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert quantitative developer and Python performance engineer.
You are helping improve an algorithmic crypto trading bot in production.

STRICT OUTPUT FORMAT:
- Reply with ONLY valid JSON.
- JSON schema:
{
  "analysis": "string",
  "risks": ["string", ...],
  "recommended_changes": ["string", ...],
  "updated_files": [
    {
      "path": "relative/path/from_bot_root.py",
      "reason": "what this change does",
      "content": "FULL NEW CONTENT of the file"
    }
  ]
}

Rules:
- Focus on robustness, risk controls, and performance.
- Never increase leverage without tightening risk controls.
- Prefer small, incremental, safe improvements.
- Only include up to 3 updated_files per run.
- If unsure about a file, do NOT include it.
"""


class EnhancedAIClient:
    """
    Enhanced AI client using fine-tuned LLMs with RAG support.
    Falls back to OpenAI if local model is not available.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        use_rag: bool = True,
        rag_persist_dir: Optional[str] = None,
        use_openai_fallback: bool = True,
    ):
        """
        Initialize the enhanced AI client.

        Args:
            model_name: Base model name (defaults to env var or Mistral-7B)
            model_path: Path to fine-tuned LoRA adapter
            use_rag: Enable RAG for document retrieval
            rag_persist_dir: Directory to persist RAG vectorstore
            use_openai_fallback: Fall back to OpenAI if local model fails
        """
        self.model_name = model_name or os.getenv(
            "LLM_MODEL_NAME",
            "mistralai/Mistral-7B-v0.1"
        )
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH")
        self.use_rag = use_rag
        self.use_openai_fallback = use_openai_fallback

        # Initialize RAG if enabled
        self.rag_system = None
        if self.use_rag:
            rag_dir = rag_persist_dir or os.getenv(
                "RAG_PERSIST_DIR",
                str(Path(__file__).parent.parent / "rag_store")
            )
            try:
                self.rag_system = RAGSystem(
                    persist_directory=rag_dir,
                    vectorstore_type="faiss",
                )
                logger.info("RAG system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG system: {e}")

        # Initialize LLM
        self.llm = None
        try:
            self.llm = FineTunedLLM(
                model_name=self.model_name,
                model_path=self.model_path,
                use_quantization=True,
            )
            logger.info(f"Loaded fine-tuned LLM: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            if self.use_openai_fallback:
                logger.info("Will use OpenAI API as fallback")
                try:
                    import openai
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    self.use_openai = True
                except ImportError:
                    logger.error("OpenAI package not available for fallback")
                    self.use_openai = False
            else:
                self.use_openai = False
                raise

    def request_improvements(self, context: str) -> Dict[str, Any]:
        """
        Request improvements using fine-tuned LLM with optional RAG.

        Args:
            context: Code and log context

        Returns:
            Dictionary with analysis, risks, recommended_changes, and updated_files
        """
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}"

        # Try RAG-enhanced query first
        if self.rag_system and self.llm:
            try:
                # Search for relevant documents
                relevant_docs = self.rag_system.similarity_search(
                    "trading bot improvement suggestions risk controls",
                    k=3
                )
                if relevant_docs:
                    doc_context = "\n\nRelevant documentation:\n"
                    doc_context += "\n".join([
                        doc.page_content[:500] for doc in relevant_docs
                    ])
                    prompt = f"{prompt}\n{doc_context}"
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Generate response
        if self.llm:
            try:
                response = self.llm.generate(
                    prompt,
                    max_length=2048,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                )
                return self._parse_response(response)
            except Exception as e:
                logger.error(f"Local LLM generation failed: {e}")
                if self.use_openai_fallback and self.use_openai:
                    return self._openai_request(prompt)
                raise

        elif self.use_openai_fallback and self.use_openai:
            return self._openai_request(prompt)

        raise RuntimeError("No LLM available and OpenAI fallback disabled")

    def _openai_request(self, prompt: str) -> Dict[str, Any]:
        """Fallback to OpenAI API."""
        import openai
        model = os.getenv("AI_ENGINE_MODEL", "gpt-4-turbo-preview")

        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        raw = resp.choices[0].message.content or ""
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse LLM response as JSON."""
        # Try to extract JSON from response
        try:
            # Look for JSON block in markdown
            if "```json" in raw:
                start = raw.find("```json") + 7
                end = raw.find("```", start)
                raw = raw[start:end].strip()
            elif "```" in raw:
                start = raw.find("```") + 3
                end = raw.find("```", start)
                raw = raw[start:end].strip()

            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("AI response not valid JSON, saving raw content.")
            parsed = {
                "analysis": "AI response was not valid JSON.",
                "risks": [],
                "recommended_changes": [],
                "updated_files": [],
                "raw": raw,
            }

        return parsed

    def add_knowledge_base(self, sources: List[str], source_type: str = "url"):
        """
        Add documents to the knowledge base.

        Args:
            sources: List of URLs, file paths, or directory paths
            source_type: "url", "file", or "directory"
        """
        if not self.rag_system:
            logger.warning("RAG system not initialized. Cannot add knowledge base.")
            return

        try:
            if source_type == "url":
                self.rag_system.load_from_urls(sources)
            elif source_type == "file":
                for file_path in sources:
                    self.rag_system.load_from_file(file_path)
            elif source_type == "directory":
                for directory in sources:
                    self.rag_system.load_from_directory(directory)
            else:
                raise ValueError(f"Unknown source_type: {source_type}")

            logger.info(f"Added {len(sources)} {source_type} sources to knowledge base")
        except Exception as e:
            logger.error(f"Failed to add knowledge base: {e}")


# Backward compatibility
def request_improvements(context: str) -> Dict[str, Any]:
    """Backward-compatible function for existing code."""
    client = EnhancedAIClient()
    return client.request_improvements(context)

