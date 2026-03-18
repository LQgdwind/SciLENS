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

class PaperInfo(Tool):
    def __init__(self, name: str = "PaperInfo"):
        assert name == "PaperInfo"
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "PaperInfo"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to retrieve the full metadata (abstract, authors, venue, year, citation count) of a specific paper.
        Input format:
        1. By ID (Recommended): {"id": "unique_paper_id"}
        2. By Title: {"title": "Exact Title of the Paper"}

        The tool returns the detailed JSON object of the paper.
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

            paper_id = args.get("id")
            title = args.get("title")

            if paper_id:
                identifier = paper_id
                search_by = "id"
            elif title:
                identifier = title
                search_by = "title"
            else:
                yield self._make_response("Error: Missing 'id' or 'title' parameter", channel)
                return

        except json.JSONDecodeError:
            yield self._make_response("Error: Invalid JSON format", channel)
            return

        try:

            result = self.papermanager.get_paper_info(identifier, by=search_by)
        except Exception as e:
            yield self._make_response(f"Database Error: {str(e)}", channel)
            return

        if not result:
            yield self._make_response(f"Paper not found by {search_by}: '{identifier}'", channel)
            return

        if '_id' in result:
            del result['_id']

        paper_json = json.dumps(result, ensure_ascii=False, indent=2)
        response_str = f"Paper Details for '{identifier}':\n\n{paper_json}"

        yield self._make_response(response_str, channel=channel)

    def reset_state(self):
        self.papermanager = PaperManager()
