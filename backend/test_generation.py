import unittest
from unittest.mock import Mock, patch

from app.generation import generate_answer
from app.models import RetrievedChunk


class GenerationTimeoutTests(unittest.TestCase):
    def test_generate_answer_returns_context_fallback_on_timeout(self):
        settings = Mock(
            final_context_k=3,
            gemini_base_url="https://example.test/v1",
            gemini_api_key="test-key",
            llm_timeout_seconds=1.0,
            llm_max_retries=0,
            gemini_model="test-model",
            gemini_max_tokens=200,
            gemini_temperature=0.0,
            confidence_scale_factor=1.6,
        )
        client = Mock()
        client.chat.completions.create.side_effect = TimeoutError("Request timed out.")
        chunks = [
            RetrievedChunk(
                document="sample.pdf",
                page=1,
                chunk_index=2,
                chunk="Michael Brown leads Gamma Corp according to the uploaded document.",
                relevance_score=0.82,
            )
        ]

        with patch("app.generation.get_settings", return_value=settings), patch(
            "app.generation.OpenAI", return_value=client
        ):
            response = generate_answer("What company is led by Michael Brown?", chunks)

        self.assertIn("LLM provider timed out", response.answer)
        self.assertIn("Michael Brown leads Gamma Corp", response.answer)
        self.assertEqual(response.sources[0].document, "sample.pdf")
        self.assertEqual(response.confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
