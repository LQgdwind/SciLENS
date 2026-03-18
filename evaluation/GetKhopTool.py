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

class GetKhopTool(Tool):
    def __init__(self, name: str = "GetKhopTool"):
        assert name == "GetKhopTool"
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "GetKhopTool"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:

        return """
        Use this tool to retrieve the K-hop citation network originating from a specific paper (Recursive/Multi-hop Citations).
        Input format:
        1. By ID (Recommended): {"id": "unique_paper_id", "k": "number_of_hops"}
        2. By Title: {"title": "Exact Title of the Paper", "k": "number_of_hops"}

        The tool returns the details of all papers within the K-hop citation graph starting from the input paper. Note that the results are capped at a maximum of 500 nodes to prevent data overload.
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

            k_val = args.get("k", 2)
            try:
                hop_numbers = int(k_val)
            except ValueError:
                yield self._make_response("Error: 'k' must be a valid number", channel)
                return

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
            results = self.papermanager.get_k_hop_references(identifier, by=search_by, k=hop_numbers)
        except Exception as e:
            yield self._make_response(f"Database Error: {str(e)}", channel)
            return

        if not results:
            yield self._make_response(f"No references found (or paper not found) for {search_by}: '{identifier}'", channel)
            return

        from collections import defaultdict

        grouped_papers = defaultdict(list)

        for x in results:

            if '_id' in x:
                del x['_id']

            hop = x.get('hop', 'Unknown')
            grouped_papers[hop].append(x)

        response_lines = []

        sorted_hops = sorted([h for h in grouped_papers.keys() if isinstance(h, int)])
        sorted_hops += [h for h in grouped_papers.keys() if not isinstance(h, int)]

        for hop in sorted_hops:
            if hop == 0:
                header = f"=== Source Paper (Hop 0) ==="
            else:
                header = f"=== Hop-{hop} ==="

            response_lines.append(header)

            for paper in grouped_papers[hop]:
                paper_json = json.dumps(paper, ensure_ascii=False)
                response_lines.append(paper_json)

            response_lines.append("")

        final_response_str = "\n".join(response_lines)
        yield self._make_response(final_response_str, channel=channel)

    def reset_state(self):
        self.papermanager = PaperManager()
