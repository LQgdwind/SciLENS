import os
import json
from typing import Any, AsyncIterator
from openai import AsyncOpenAI

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)
from tool import Tool

DEFAULT_MODEL_NAME = "gpt-4o"

class SummarizeTool(Tool):
    def __init__(self, name: str = "SummarizeTool"):
        assert name == "SummarizeTool"

        self.model_name = os.getenv("SUMMARY_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "SummarizeTool"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to summarize academic papers or long text segments.
        Input format: {"content": "The long text or paper abstract you want summarized."}

        The tool returns a concise summary of the provided text.
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        return self.make_response(content=content, channel=channel)

    def make_response(self, content: Content, *, channel: str | None = None, **kwargs) -> Message:
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=f"{tool_name}")
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text

        try:
            args = json.loads(raw_text)
            content_to_summarize = args.get("content")
            if not content_to_summarize:
                yield self._make_response("Error: Missing 'content' parameter in JSON", channel)
                return

        except json.JSONDecodeError:
            yield self._make_response("Error: Invalid JSON format", channel)
            return

        try:

            messages = [
                {"role": "system", "content": "You are a helpful research assistant. Summarize the following text concisely."},
                {"role": "user", "content": content_to_summarize}
            ]

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )

            summary_text = response.choices[0].message.content

        except Exception as e:

            print(f"[SummarizeTool Error] {e}")
            yield self._make_response(f"Summarize API Error: {str(e)}", channel)
            return

        yield self._make_response(summary_text, channel=channel)

    def reset_state(self):
        api_key = os.getenv("SUMMARY_API_KEY")
        base_url = os.getenv("SUMMARY_API_BASE")

        if not api_key:
            print("[Warning] SUMMARY_API_KEY environment variable is not set. Tool may fail.")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
