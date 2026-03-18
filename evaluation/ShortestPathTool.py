import json
from typing import AsyncIterator
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

class ShortestPathTool(Tool):
    def __init__(self, name: str = "ShortestPathTool"):
        assert name == "ShortestPathTool"
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "ShortestPathTool"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to find the shortest citation path from paper A to paper B (A -> ... -> B).

        Input format (JSON):
        1. By ID (Recommended):
           {"start_id": "source_paper_id", "end_id": "target_paper_id", "max_depth": 4}
        2. By Title:
           {"start_title": "Title of Paper A", "end_title": "Title of Paper B", "max_depth": 4}

        The tool returns a list of papers representing the path. If no path is found, it returns a message stating so.
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

            if "start_id" in args:
                start_identifier = args["start_id"]
                start_by = "id"
            elif "start_title" in args:
                start_identifier = args["start_title"]
                start_by = "title"
            else:
                yield self._make_response("Error: Missing 'start_id' or 'start_title'", channel)
                return

            if "end_id" in args:
                end_identifier = args["end_id"]
                end_by = "id"
            elif "end_title" in args:
                end_identifier = args["end_title"]
                end_by = "title"
            else:
                yield self._make_response("Error: Missing 'end_id' or 'end_title'", channel)
                return
            if start_by != end_by:
                yield self._make_response("Error: Please provide both identifiers as IDs or both as Titles.", channel)
                return

            max_depth = int(args.get("max_depth", 3))

            if max_depth > 5:
                max_depth = 5

        except json.JSONDecodeError:
            yield self._make_response("Error: Invalid JSON format", channel)
            return
        except ValueError:
             yield self._make_response("Error: 'max_depth' must be a number", channel)
             return

        try:

            path_papers = self.papermanager.find_shortest_path(
                start_identifier=start_identifier,
                end_identifier=end_identifier,
                max_depth=max_depth,
                by=start_by
            )
        except Exception as e:

            error_str = str(e)
            if "memory" in error_str.lower() or "40099" in error_str:
                 yield self._make_response(f"System Error: The citation graph is too large (Memory Limit Exceeded). Please try again with a smaller 'max_depth' (e.g., 2).", channel)
            else:
                 yield self._make_response(f"Database Error: {error_str}", channel)
            return

        if not path_papers:

            msg = (
                f"No path found between '{start_identifier}' and '{end_identifier}' within depth {max_depth}.\n\n"
                f"Tips for the Assistant:\n"
                f"1. If these are seminal papers (e.g., Transformer, ResNet) with thousands of citations, the graph might be too complex to fully traverse.\n"
                f"2. The path might not exist in the 'forward citation' direction (A cites B).\n"
                f"3. Try identifying a common intermediate paper and searching in two steps."
            )
            yield self._make_response(msg, channel)
            return

        path_nodes = []
        for paper in path_papers:
            paper_id = paper.get("_id", "Unknown")
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "N/A")
            path_nodes.append(f"[{paper_id}] {title} ({year})")

        path_str = " -> ".join(path_nodes)

        response_text = (
            f"Found a path with {len(path_papers)} steps (Depth {len(path_papers)-1}):\n\n"
            f"{path_str}"
        )

        yield self._make_response(response_text, channel=channel)

    def reset_state(self):

        self.papermanager = PaperManager()
