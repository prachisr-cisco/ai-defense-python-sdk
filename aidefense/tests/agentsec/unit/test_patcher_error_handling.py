"""Tests for patcher error handling (Task Group 2).

Tests that LLM patchers handle inspector errors gracefully and respect fail_open settings.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aidefense.runtime.agentsec.decision import Decision
from aidefense.runtime.agentsec.exceptions import SecurityPolicyError


class TestOpenAIPatcherErrorHandling:
    """Test error handling in OpenAI patcher."""

    def test_inspector_error_with_fail_open_true_allows_call(self):
        """Test that inspector errors with fail_open=True allow the LLM call."""
        from aidefense.runtime.agentsec.patchers.openai import (
            _wrap_chat_completions_create,
            _get_inspector,
        )
        
        # Mock the inspector to raise an error
        mock_inspector = MagicMock()
        mock_inspector.inspect_conversation.side_effect = Exception("API error")
        mock_inspector.fail_open = True
        
        # Mock the wrapped function
        mock_wrapped = MagicMock(return_value=MagicMock())
        
        # Mock context and state
        with patch("aidefense.runtime.agentsec.patchers.openai._get_inspector", return_value=mock_inspector):
            with patch("aidefense.runtime.agentsec.patchers.openai._should_inspect", return_value=True):
                with patch("aidefense.runtime.agentsec.patchers.openai.get_inspection_context") as mock_ctx:
                    mock_ctx.return_value.metadata = {}
                    with patch("aidefense.runtime.agentsec.patchers.openai._state") as mock_state:
                        mock_state.get_llm_mode.return_value = "monitor"
                        mock_state.get_api_llm_fail_open.return_value = True
                        
                        # Should not raise, should allow the call
                        result = _wrap_chat_completions_create(
                            mock_wrapped,
                            None,
                            [],
                            {"messages": [{"role": "user", "content": "test"}]},
                        )
        
        # The wrapped function should have been called
        mock_wrapped.assert_called_once()

    def test_streaming_wrapper_raises_on_inspection_error_fail_open_false(self):
        """Test StreamingInspectionWrapper raises SecurityPolicyError when fail_open=False."""
        from aidefense.runtime.agentsec.patchers.openai import StreamingInspectionWrapper

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"

        def mock_stream():
            yield mock_chunk
            yield mock_chunk

        mock_inspector = MagicMock()
        mock_inspector.inspect_conversation.side_effect = Exception("API error")
        mock_inspector.fail_open = False

        with patch("aidefense.runtime.agentsec.patchers.openai._get_inspector", return_value=mock_inspector):
            with patch("aidefense.runtime.agentsec.patchers.openai._should_inspect", return_value=True):
                with patch("aidefense.runtime.agentsec.patchers.openai._state") as mock_state:
                    mock_state.get_api_llm_fail_open.return_value = False
                    wrapper = StreamingInspectionWrapper(
                        mock_stream(),
                        [{"role": "user", "content": "test"}],
                        {},
                    )

                    with pytest.raises(SecurityPolicyError):
                        list(wrapper)

    def test_streaming_wrapper_allows_on_inspection_error_fail_open_true(self):
        """Test StreamingInspectionWrapper allows through when fail_open=True."""
        from aidefense.runtime.agentsec.patchers.openai import StreamingInspectionWrapper

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"

        def mock_stream():
            yield mock_chunk
            yield mock_chunk

        mock_inspector = MagicMock()
        mock_inspector.inspect_conversation.side_effect = Exception("API error")
        mock_inspector.fail_open = True

        with patch("aidefense.runtime.agentsec.patchers.openai._get_inspector", return_value=mock_inspector):
            with patch("aidefense.runtime.agentsec.patchers.openai._should_inspect", return_value=True):
                with patch("aidefense.runtime.agentsec.patchers.openai._state") as mock_state:
                    mock_state.get_api_llm_fail_open.return_value = True
                    wrapper = StreamingInspectionWrapper(
                        mock_stream(),
                        [{"role": "user", "content": "test"}],
                        {},
                    )

                    chunks = list(wrapper)
                    assert len(chunks) == 2






