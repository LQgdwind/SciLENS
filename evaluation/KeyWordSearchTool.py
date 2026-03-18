import json
from typing import Any, AsyncIterator
from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)
from tool import Tool

from utils.db_tool import PaperManager

class KeywordSearchTool(Tool):
    def __init__(self, name: str = "KeywordSearch"):
        assert name == "KeywordSearch"
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "KeywordSearch"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to search for papers based on keywords.
        Input format: {"keywords": ["machine learning", "transformers"]}
        The tool returns a list of papers containing ALL provided keywords.
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
            keywords_value = args.get("keywords")
            if not keywords_value:
                yield self._make_response("Error: Missing 'keywords' parameter", channel)
                return
        except json.JSONDecodeError:
            yield self._make_response("Error: Invalid JSON format", channel)
            return

        try:

            results = self.papermanager.search_by_keywords(keywords_value, limit=10)
        except Exception as e:
            yield self._make_response(f"Database Error: {str(e)}", channel)
            return

        if not results:
            yield self._make_response(f"No papers found matching keywords: {keywords_value}", channel)
            return

        outputs = []
        for p in results:

            title = p.get('title', 'N/A')
            year = p.get('year', 'N/A')
            citation = p.get('n_citation', 0)

            outputs.append(f"[{year}] {title} (Citations: {citation})")

        final_output = "Found Papers:\n" + "\n".join(outputs)
        yield self._make_response(final_output, channel=channel)

    def reset_state(self):

        self.papermanager = PaperManager()
