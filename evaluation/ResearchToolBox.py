import json
from typing import AsyncIterator
from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)

from tool import Tool

try:
    from EmbeddingSearchTool import EmbeddingSearchTool
except ImportError:
    print("Warning: EmbeddingSearchTool.py not found.")

try:
    from GetNeighborTool import ReferenceSearch
except ImportError:
    print("Warning: GetNeighborTool.py not found.")

try:
    from KeyWordSearchTool import KeywordSearchTool
except ImportError:
    print("Warning: KeyWordSearchTool.py not found.")

try:
    from PaperInfoTool import PaperInfo
except ImportError:
    print("Warning: PaperInfoTool.py not found.")

try:
    from ShortestPathTool import ShortestPathTool
except ImportError:
    print("Warning: ShortestPathTool.py not found.")

try:
    from GetKhopTool import GetKhopTool
except ImportError:
    print("Warning: GetKhopTool.py not found.")

try:
    from GetInDegreeTool import GetInDegreeTool
except ImportError:
    print("Warning: GetInDegreeTool.py not found.")

try:
    from SummarizeTool import SummarizeTool
except ImportError:
    print("Warning: SummarizeTool.py not found.")

try:
    from PlotTools import LineChartTool, BarChartTool, PieChartTool, ScatterPlotTool
except ImportError:
    print("Warning: PlotTools.py not found.")

class ResearchToolbox(Tool):
    def __init__(self, name: str = "ResearchToolbox"):
        assert name == "ResearchToolbox"
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "ResearchToolbox"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:

        return """
        This is a powerful toolbox for academic paper research.
        You must provide a JSON input with:
        1. "tool_name": The name of the specific tool.
        2. "tool_args": The arguments for that tool.

        Available Tools:

        1. EmbeddingSearch
           - Description: Search papers by semantic meaning (vector similarity). Best for broad topic exploration.
           - Args: {"query": "natural language processing reasoning"}

        2. KeywordSearch
           - Description: Search papers by exact keywords. Best for finding specific topics.
           - Args: {"keywords": ["Large Language Models", "Chain of Thought"]}

        3. ReferenceSearch
           - Description: Find all papers cited by a specific paper (outgoing citations).
           - Args (Option 1): {"id": "unique_paper_id"}
           - Args (Option 2): {"title": "Exact Title of the Paper"}

        4. PaperInfo
           - Description: Retrieve full metadata (abstract, authors, venue, etc.) of a specific paper.
           - Args (Option 1): {"id": "unique_paper_id"}
           - Args (Option 2): {"title": "Exact Title of the Paper"}

        5. ShortestPath
           - Description: Find the shortest citation path from paper A to paper B.
           - Args (Option 1): {"start_id": "id_A", "end_id": "id_B", "max_depth": 3}
           - Args (Option 2): {"start_title": "Title A", "end_title": "Title B", "max_depth": 3}

        6. GetKhop
           - Description: Retrieve the K-hop citation network originating from a paper (Recursive citations).
           - Args (Option 1): {"id": "unique_paper_id", "k": 2}
           - Args (Option 2): {"title": "Exact Title", "k": 2}

        7. GetInDegree
           - Description: Find papers that cite a specific paper (Incoming citations / Cited By).
           - Args (Option 1): {"id": "unique_paper_id"}
           - Args (Option 2): {"title": "Exact Title"}

        8. Summarize
            - Description: Generate a concise summary of a paper abstract or any long text segment.
            - Args: {"content": "The text content you want summarized..."}

        9. LineChart
            - Description: Create line charts for trends over time (e.g., citation trends).
            - Args: {"title": "...", "x_label": "Year", "y_label": "Citations", "data": {"labels": [...], "datasets": [...]}}

        10. BarChart
            - Description: Create bar charts for comparing categories (e.g., Top N papers).
            - Args: {"title": "...", "x_label": "...", "y_label": "...", "data": {"labels": [...], "values": [...]}}

        11. PieChart
            - Description: Create pie charts for showing proportions (e.g., field distribution).
            - Args: {"title": "...", "data": {"labels": [...], "values": [...]}}

        12. ScatterPlot
            - Description: Create scatter plots for correlation analysis (e.g., citations vs year).
            - Args: {"title": "...", "x_label": "...", "y_label": "...", "data": {"points": [{"x": ..., "y": ..., "label": "..."}]}}
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def reset_state(self):

        self.tools = {}

        try:
            self.tools["EmbeddingSearch"] = EmbeddingSearchTool()
            print("[Toolbox] EmbeddingSearch loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load EmbeddingSearch: {e}")

        try:
            self.tools["ReferenceSearch"] = ReferenceSearch()
            print("[Toolbox] ReferenceSearch loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load ReferenceSearch: {e}")

        try:
            self.tools["KeywordSearch"] = KeywordSearchTool()
            print("[Toolbox] KeywordSearch loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load KeywordSearch: {e}")

        try:
            self.tools["PaperInfo"] = PaperInfo()
            print("[Toolbox] PaperInfo loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load PaperInfo: {e}")

        try:
            self.tools["ShortestPath"] = ShortestPathTool()
            print("[Toolbox] ShortestPath loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load ShortestPath: {e}")

        try:
            self.tools["GetKhop"] = GetKhopTool()
            print("[Toolbox] GetKhop loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load GetKhop: {e}")

        try:
            self.tools["GetInDegree"] = GetInDegreeTool()
            print("[Toolbox] GetInDegree loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load GetInDegree: {e}")

        try:

            self.tools["Summarize"] = SummarizeTool()
            print("[Toolbox] SummarizeTool loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load SummarizeTool: {e}")

        try:
            self.tools["LineChart"] = LineChartTool()
            print("[Toolbox] LineChart loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load LineChart: {e}")

        try:
            self.tools["BarChart"] = BarChartTool()
            print("[Toolbox] BarChart loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load BarChart: {e}")

        try:
            self.tools["PieChart"] = PieChartTool()
            print("[Toolbox] PieChart loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load PieChart: {e}")

        try:
            self.tools["ScatterPlot"] = ScatterPlotTool()
            print("[Toolbox] ScatterPlot loaded.")
        except Exception as e:
            print(f"[Toolbox] Failed to load ScatterPlot: {e}")

    def _make_error_response(self, error_msg: str, channel: str | None = None) -> Message:
        author = Author(role=Role.TOOL, name=self.name)
        return Message(
            author=author,
            content=[TextContent(text=f"Toolbox Error: {error_msg}")]
        ).with_recipient("assistant").with_channel(channel)

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text

        try:
            payload = json.loads(raw_text)
            tool_name = payload.get("tool_name")
            tool_args = payload.get("tool_args")

            if not tool_name or tool_args is None:
                yield self._make_error_response("Missing 'tool_name' or 'tool_args'.", channel)
                return

        except json.JSONDecodeError:
            yield self._make_error_response("Invalid JSON format.", channel)
            return

        target_tool = self.tools.get(tool_name)

        if not target_tool:

            yield self._make_error_response(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}", channel)
            return

        sub_tool_payload = json.dumps(tool_args, ensure_ascii=False)

        sub_message = Message(
            author=Author(role=Role.ASSISTANT, name="gpt-4"),
            content=[TextContent(text=sub_tool_payload)]
        ).with_channel(channel)

        try:
            async for sub_response in target_tool._process(sub_message):
                yield sub_response
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield self._make_error_response(f"Error executing tool '{tool_name}': {str(e)}", channel)
