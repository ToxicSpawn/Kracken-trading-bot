"""
Voice transcription and multi-language support using Whisper.
"""
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_PIPELINE_AVAILABLE = True
except ImportError:
    TRANSFORMERS_PIPELINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class VoiceTranscriber:
    """
    Voice transcription using Whisper.
    Supports multiple languages and translation.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
    ):
        """
        Initialize voice transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu, auto)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )

        self.model_size = model_size
        self.device = device or ("cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        logger.info(f"Loading Whisper model: {self.model_size}")
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            task: "transcribe" or "translate"
            **kwargs: Additional Whisper parameters

        Returns:
            Dictionary with text, language, and segments
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio: {audio_path}")

        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            **kwargs
        )

        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
        }

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio from bytes.

        Args:
            audio_bytes: Audio file bytes
            language: Language code
            task: "transcribe" or "translate"
            **kwargs: Additional Whisper parameters

        Returns:
            Dictionary with text, language, and segments
        """
        import tempfile

        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.transcribe(tmp_path, language=language, task=task, **kwargs)
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink()

        return result


class MultiLanguageTranslator:
    """Multi-language translation using NLLB or SeamlessM4T."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
    ):
        """
        Initialize translator.

        Args:
            model_name: HuggingFace model name for translation
        """
        if not TRANSFORMERS_PIPELINE_AVAILABLE:
            raise ImportError(
                "Transformers pipeline not available. Install with: pip install transformers"
            )

        self.model_name = model_name
        self.translator = None
        self._load_model()

    def _load_model(self):
        """Load translation model."""
        logger.info(f"Loading translation model: {self.model_name}")
        try:
            self.translator = pipeline(
                "translation",
                model=self.model_name,
                src_lang="eng_Latn",
                tgt_lang="fra_Latn",  # Default to French, can be changed
            )
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate(
        self,
        text: str,
        source_lang: str = "eng_Latn",
        target_lang: str = "fra_Latn",
    ) -> str:
        """
        Translate text.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        if self.translator is None:
            raise RuntimeError("Translation model not loaded")

        # Reload pipeline with new language pair if needed
        if hasattr(self.translator, 'tokenizer'):
            # Simple approach: recreate pipeline
            self.translator = pipeline(
                "translation",
                model=self.model_name,
                src_lang=source_lang,
                tgt_lang=target_lang,
            )

        result = self.translator(text)
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("translation_text", text)
        return text

