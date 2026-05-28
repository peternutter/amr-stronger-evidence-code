import os
from dataclasses import dataclass
from typing import Literal

import anthropic
import openai

from src.utils import RankedLogger
from src.utils.types import Conversation

log = RankedLogger(__name__, rank_zero_only=True)

EffortLevel = Literal["low", "medium", "high"]


@dataclass
class GenerationResult:
    """Result from model generation, including optional thinking/reasoning content."""

    text: str
    thinking: str | None = None
    used_thinking: bool = False
    # Token usage tracking
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None  # Anthropic thinking or OpenAI reasoning tokens

    def __str__(self) -> str:
        """Return just the text for backwards compatibility."""
        return self.text

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "thinking": self.thinking,
            "used_thinking": self.used_thinking,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
        }


class RemoteModel:
    def __init__(
        self,
        provider: Literal["openai", "anthropic"],
        model: str,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        # OpenAI reasoning effort (for o-series, gpt-5)
        effort: EffortLevel | None = None,
        # Anthropic thinking budget (token count)
        thinking_budget: int | None = None,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.effort = effort  # OpenAI
        self.thinking_budget = thinking_budget  # Anthropic

        assert (
            self.max_tokens is not None or self.max_completion_tokens is not None
        ), "RemoteModel requires either max_tokens or max_completion_tokens."

        if provider == "openai":
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key, "OpenAI API key not found. Set OPENAI_API_KEY environment variable."

            self.client = openai.OpenAI(api_key=api_key)
            effort_info = f" (effort={effort})" if effort else ""
            log.info(f"OpenAI client initialized for response evaluation{effort_info}")
        elif provider == "anthropic":
            # Initialize Anthropic client
            api_key = os.getenv("ANTHROPIC_API_KEY")
            assert api_key, "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."

            self.client = anthropic.Anthropic(api_key=api_key)
            thinking_info = f" (thinking_budget={thinking_budget})" if thinking_budget else ""
            log.info(f"Anthropic client initialized for response evaluation{thinking_info}")
        else:
            raise NotImplementedError(f"Provider '{provider}' is not implemented.")

    def generate(
        self,
        conversation: Conversation,
    ) -> GenerationResult:
        """Generate a response for the given conversation using the API model.

        Returns:
            GenerationResult with text, optional thinking content, token usage, and whether thinking was used.
        """
        if self.provider == "openai":
            # Forward whichever token limit the config provided so both APIs work.
            token_kwargs = {}
            if self.max_tokens is not None:
                token_kwargs["max_tokens"] = self.max_tokens
            if self.max_completion_tokens is not None:
                token_kwargs["max_completion_tokens"] = self.max_completion_tokens

            # Add reasoning effort for reasoning models (o-series, gpt-5)
            if self.effort is not None:
                token_kwargs["reasoning_effort"] = self.effort

            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                **token_kwargs,
            )

            # Extract token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            # OpenAI reasoning tokens are in completion_tokens_details
            reasoning_tokens = None
            if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", None)

            return GenerationResult(
                text=response.choices[0].message.content,
                thinking=None,  # OpenAI doesn't expose reasoning content
                used_thinking=self.effort is not None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=reasoning_tokens,
            )
        elif self.provider == "anthropic":
            # Anthropic uses max_tokens (required parameter)
            max_tokens = self.max_tokens or self.max_completion_tokens

            # Extract system message if present (Anthropic handles it separately)
            system_message = None
            messages = []
            for msg in conversation:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)

            # Build request kwargs
            request_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system_message:
                request_kwargs["system"] = system_message

            # Add extended thinking for Claude 4.5+ models
            if self.thinking_budget is not None:
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }

            response = self.client.messages.create(**request_kwargs)

            # Extract thinking and text from response
            thinking_content = None
            text_content = None

            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                elif block.type == "text":
                    text_content = block.text

            # Extract token usage
            usage = response.usage
            input_tokens = usage.input_tokens if usage else None
            output_tokens = usage.output_tokens if usage else None
            thinking_tokens = None

            return GenerationResult(
                text=text_content or "",
                thinking=thinking_content,
                used_thinking=self.thinking_budget is not None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
            )
        else:
            raise NotImplementedError(f"Provider '{self.provider}' is not implemented.")
