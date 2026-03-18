import json
import os
from typing import AsyncIterator, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from datetime import datetime

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)
from tool import Tool

class ChartStyle:
    PRIMARY_COLOR = '#2E86AB'
    SECONDARY_COLOR = '#A23B72'
    ACCENT_COLOR = '#F18F01'
    SUCCESS_COLOR = '#06A77D'

    COLOR_PALETTE = [
        '#2E86AB', '#A23B72', '#F18F01', '#06A77D',
        '#C73E1D', '#6A4C93', '#3D5A80', '#98C1D9',
    ]

    PIE_COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
    ]

    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10
    LEGEND_FONT_SIZE = 10
    FIGURE_SIZE = (12, 7)
    DPI = 300
    BAR_WIDTH = 0.5
    BAR_ALPHA = 0.85
    BAR_EDGE_WIDTH = 1.2
    GRID_ALPHA = 0.25
    GRID_LINESTYLE = '--'
    GRID_LINEWIDTH = 0.5

    @staticmethod
    def apply_style():
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': '#F8F9FA',
            'axes.edgecolor': '#DEE2E6',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': ChartStyle.GRID_ALPHA,
            'grid.linestyle': ChartStyle.GRID_LINESTYLE,
            'grid.linewidth': ChartStyle.GRID_LINEWIDTH,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.labelsize': ChartStyle.LABEL_FONT_SIZE,
            'axes.titlesize': ChartStyle.TITLE_FONT_SIZE,
            'xtick.labelsize': ChartStyle.TICK_FONT_SIZE,
            'ytick.labelsize': ChartStyle.TICK_FONT_SIZE,
            'legend.fontsize': ChartStyle.LEGEND_FONT_SIZE,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '#DEE2E6',
        })

ChartStyle.apply_style()

class BasePlotTool(Tool):

    DEFAULT_SAVE_DIR = Path("./generated_charts")

    def __init__(self, output_dir: Optional[str] = None):

        if output_dir is not None:
            self._save_dir = Path(output_dir)
        else:
            self._save_dir = self.DEFAULT_SAVE_DIR

        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._chart_counter = 0

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    def set_output_dir(self, output_dir: str):
        self._save_dir = Path(output_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

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

    def _generate_filename(self, chart_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._chart_counter += 1
        return f"{chart_type}_{timestamp}_{self._chart_counter}.png"

    def _save_figure(self, fig, chart_type: str) -> str:
        filename = self._generate_filename(chart_type)
        filepath = self._save_dir / filename
        fig.savefig(filepath, dpi=ChartStyle.DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        return str(filepath)

    def _add_watermark(self, ax, position='bottom'):
        watermark_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if position == 'bottom':
            ax.text(0.99, 0.01, watermark_text,
                    transform=ax.transAxes,
                    fontsize=7, color='gray', alpha=0.5,
                    ha='right', va='bottom')

    def _format_response(self, chart_type: str, params: dict, filepath: str) -> str:
        response_parts = [
            f"✓ {chart_type.upper()} Chart Generated Successfully",
            f"�� Title: {params.get('title', 'Untitled')}",
            f"�� Saved to: {filepath}",
            ""
        ]
        return "\n".join(response_parts)

    def reset_state(self):
        self._chart_counter = 0

class LineChartTool(BasePlotTool):

    def __init__(self, name: str = "LineChart", output_dir: Optional[str] = None):
        assert name == "LineChart"
        super().__init__(output_dir=output_dir)
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "LineChart"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to create LINE CHARTS for visualizing trends over time or sequences.

        Input Format:
        {
            "title": "Chart Title",
            "x_label": "X-axis Label",
            "y_label": "Y-axis Label",
            "data": {
                "labels": [2018, 2019, 2020, 2021, 2022],
                "datasets": [
                    {"label": "Series A", "values": [10, 25, 40, 65, 90]}
                ]
            }
        }
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _create_line_chart(self, params: dict) -> str:
        fig, ax = plt.subplots(figsize=ChartStyle.FIGURE_SIZE)

        data = params["data"]
        labels = data["labels"]
        datasets = data["datasets"]

        for idx, dataset in enumerate(datasets):
            color = ChartStyle.COLOR_PALETTE[idx % len(ChartStyle.COLOR_PALETTE)]
            ax.plot(labels, dataset["values"],
                    marker='o',
                    label=dataset["label"],
                    linewidth=2.5,
                    color=color,
                    markersize=8,
                    markeredgecolor='white',
                    markeredgewidth=2,
                    alpha=0.9)

        ax.set_title(params["title"], fontsize=ChartStyle.TITLE_FONT_SIZE,
                     fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel(params.get("x_label", "X Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                      fontweight='semibold', color='#34495E')
        ax.set_ylabel(params.get("y_label", "Y Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                      fontweight='semibold', color='#34495E')
        ax.grid(True, alpha=ChartStyle.GRID_ALPHA, linestyle=ChartStyle.GRID_LINESTYLE,
                linewidth=ChartStyle.GRID_LINEWIDTH, color='gray')
        ax.set_axisbelow(True)

        legend = ax.legend(loc='best', framealpha=0.95, edgecolor='#BDC3C7',
                           fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        ax.tick_params(axis='both', which='major',
                       labelsize=ChartStyle.TICK_FONT_SIZE, colors='#34495E')
        self._add_watermark(ax)
        plt.tight_layout()
        return self._save_figure(fig, "line")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text
        try:
            params = json.loads(raw_text)
            if "title" not in params:
                yield self._make_response("Error: Missing 'title'", channel); return
            if "data" not in params:
                yield self._make_response("Error: Missing 'data'", channel); return
            data = params["data"]
            if "labels" not in data or "datasets" not in data:
                yield self._make_response("Error: 'data' must contain 'labels' and 'datasets'", channel); return
            if not isinstance(data["datasets"], list) or len(data["datasets"]) == 0:
                yield self._make_response("Error: 'datasets' must be a non-empty list", channel); return
            num_labels = len(data["labels"])
            for idx, dataset in enumerate(data["datasets"]):
                if "label" not in dataset or "values" not in dataset:
                    yield self._make_response(f"Error: dataset[{idx}] missing 'label' or 'values'", channel); return
                if len(dataset["values"]) != num_labels:
                    yield self._make_response(f"Error: dataset[{idx}] values length mismatch", channel); return
            filepath = self._create_line_chart(params)
            yield self._make_response(self._format_response("line", params, filepath), channel)
        except json.JSONDecodeError as e:
            yield self._make_response(f"Error: Invalid JSON - {str(e)}", channel)
        except Exception as e:
            yield self._make_response(f"Error: {str(e)}", channel)

class BarChartTool(BasePlotTool):

    def __init__(self, name: str = "BarChart", output_dir: Optional[str] = None):
        assert name == "BarChart"
        super().__init__(output_dir=output_dir)
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "BarChart"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to create BAR CHARTS for comparing values across categories.

        Input Format:
        {
            "title": "Top 5 Most Cited Papers",
            "x_label": "Paper Title",
            "y_label": "Citation Count",
            "data": {
                "labels": ["Paper A", "Paper B", "Paper C"],
                "values": [1500, 1200, 980]
            },
            "orientation": "vertical"
        }
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _create_bar_chart(self, params: dict) -> str:
        fig, ax = plt.subplots(figsize=ChartStyle.FIGURE_SIZE)
        data = params["data"]
        labels = data["labels"]
        values = data["values"]
        orientation = params.get("orientation", "vertical")
        colors = [ChartStyle.COLOR_PALETTE[i % len(ChartStyle.COLOR_PALETTE)]
                  for i in range(len(values))]

        if orientation == "horizontal":
            bars = ax.barh(labels, values, height=ChartStyle.BAR_WIDTH,
                           color=colors, alpha=ChartStyle.BAR_ALPHA,
                           edgecolor='white', linewidth=ChartStyle.BAR_EDGE_WIDTH)
            ax.set_xlabel(params.get("y_label", "Values"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                          fontweight='semibold', color='#34495E')
            ax.set_ylabel(params.get("x_label", "Categories"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                          fontweight='semibold', color='#34495E')
            for bar, value in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{value:,.0f}', ha='left', va='center',
                        fontsize=ChartStyle.TICK_FONT_SIZE, fontweight='bold', color='#2C3E50')
        else:
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, values, width=ChartStyle.BAR_WIDTH,
                          color=colors, alpha=ChartStyle.BAR_ALPHA,
                          edgecolor='white', linewidth=ChartStyle.BAR_EDGE_WIDTH)
            ax.set_xlabel(params.get("x_label", "Categories"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                          fontweight='semibold', color='#34495E')
            ax.set_ylabel(params.get("y_label", "Values"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                          fontweight='semibold', color='#34495E')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(values) * 0.01,
                        f'{value:,.0f}', ha='center', va='bottom',
                        fontsize=ChartStyle.TICK_FONT_SIZE, fontweight='bold', color='#2C3E50')

        ax.set_title(params["title"], fontsize=ChartStyle.TITLE_FONT_SIZE,
                     fontweight='bold', pad=20, color='#2C3E50')
        if orientation == "vertical":
            ax.yaxis.grid(True, alpha=ChartStyle.GRID_ALPHA,
                          linestyle=ChartStyle.GRID_LINESTYLE, linewidth=ChartStyle.GRID_LINEWIDTH)
            ax.xaxis.grid(False)
        else:
            ax.xaxis.grid(True, alpha=ChartStyle.GRID_ALPHA,
                          linestyle=ChartStyle.GRID_LINESTYLE, linewidth=ChartStyle.GRID_LINEWIDTH)
            ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major',
                       labelsize=ChartStyle.TICK_FONT_SIZE, colors='#34495E')
        self._add_watermark(ax)
        plt.tight_layout()
        return self._save_figure(fig, "bar")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text
        try:
            params = json.loads(raw_text)
            if "title" not in params:
                yield self._make_response("Error: Missing 'title'", channel); return
            if "data" not in params:
                yield self._make_response("Error: Missing 'data'", channel); return
            data = params["data"]
            if "labels" not in data or "values" not in data:
                yield self._make_response("Error: 'data' must contain 'labels' and 'values'", channel); return
            if len(data["labels"]) != len(data["values"]):
                yield self._make_response("Error: 'labels' and 'values' must have same length", channel); return
            if "orientation" not in params:
                params["orientation"] = "vertical"
            filepath = self._create_bar_chart(params)
            yield self._make_response(self._format_response("bar", params, filepath), channel)
        except json.JSONDecodeError as e:
            yield self._make_response(f"Error: Invalid JSON - {str(e)}", channel)
        except Exception as e:
            yield self._make_response(f"Error: {str(e)}", channel)

class PieChartTool(BasePlotTool):

    def __init__(self, name: str = "PieChart", output_dir: Optional[str] = None):
        assert name == "PieChart"
        super().__init__(output_dir=output_dir)
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "PieChart"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to create PIE CHARTS for showing proportions and distributions.

        Input Format:
        {
            "title": "Research Field Distribution",
            "data": {
                "labels": ["NLP", "CV", "ML", "Robotics"],
                "values": [35, 28, 22, 15]
            }
        }
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _create_pie_chart(self, params: dict) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        data = params["data"]
        labels = data["labels"]
        values = data["values"]
        colors = [ChartStyle.PIE_COLORS[i % len(ChartStyle.PIE_COLORS)]
                  for i in range(len(labels))]
        max_value = max(values)
        explode = [0.05 if v == max_value else 0 for v in values]

        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors, explode=explode, startangle=90,
            textprops={'fontsize': ChartStyle.TICK_FONT_SIZE},
            pctdistance=0.85, shadow=True
        )
        for text in texts:
            text.set_fontsize(ChartStyle.TICK_FONT_SIZE)
            text.set_fontweight('semibold')
            text.set_color('#2C3E50')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(ChartStyle.TICK_FONT_SIZE)

        ax.set_title(params["title"], fontsize=ChartStyle.TITLE_FONT_SIZE,
                     fontweight='bold', pad=20, color='#2C3E50')
        ax.axis('equal')
        ax.legend(wedges, labels, title="Categories",
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=ChartStyle.LEGEND_FONT_SIZE,
                  framealpha=0.95, edgecolor='#BDC3C7')
        fig.text(0.99, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=7, color='gray', alpha=0.5, ha='right', va='bottom')
        plt.tight_layout()
        return self._save_figure(fig, "pie")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text
        try:
            params = json.loads(raw_text)
            if "title" not in params:
                yield self._make_response("Error: Missing 'title'", channel); return
            if "data" not in params:
                yield self._make_response("Error: Missing 'data'", channel); return
            data = params["data"]
            if "labels" not in data or "values" not in data:
                yield self._make_response("Error: 'data' must contain 'labels' and 'values'", channel); return
            if len(data["labels"]) != len(data["values"]):
                yield self._make_response("Error: 'labels' and 'values' must have same length", channel); return
            if any(v <= 0 for v in data["values"]):
                yield self._make_response("Error: All values must be positive", channel); return
            filepath = self._create_pie_chart(params)
            yield self._make_response(self._format_response("pie", params, filepath), channel)
        except json.JSONDecodeError as e:
            yield self._make_response(f"Error: Invalid JSON - {str(e)}", channel)
        except Exception as e:
            yield self._make_response(f"Error: {str(e)}", channel)

class ScatterPlotTool(BasePlotTool):

    def __init__(self, name: str = "ScatterPlot", output_dir: Optional[str] = None):
        assert name == "ScatterPlot"
        super().__init__(output_dir=output_dir)
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "ScatterPlot"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to create SCATTER PLOTS for showing correlations between two variables.

        Input Format:
        {
            "title": "Citation Count vs Publication Year",
            "x_label": "Publication Year",
            "y_label": "Citation Count",
            "data": {
                "points": [
                    {"x": 2018, "y": 45, "label": "Paper A"},
                    {"x": 2019, "y": 67, "label": "Paper B"}
                ]
            }
        }
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _create_scatter_plot(self, params: dict) -> str:
        fig, ax = plt.subplots(figsize=ChartStyle.FIGURE_SIZE)
        data = params["data"]
        points = data["points"]
        x_coords = [p["x"] for p in points]
        y_coords = [p["y"] for p in points]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(points)))

        ax.scatter(x_coords, y_coords, s=200, alpha=0.7, c=colors,
                   edgecolors='white', linewidth=2.5)

        for point in points:
            if "label" in point:
                ax.annotate(point["label"], (point["x"], point["y"]),
                            xytext=(8, 8), textcoords='offset points',
                            fontsize=ChartStyle.TICK_FONT_SIZE, fontweight='semibold',
                            color='#2C3E50',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                      edgecolor='gray', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                            color='gray', lw=1.5))

        if len(x_coords) > 1:
            z = np.polyfit(x_coords, y_coords, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x_coords), max(x_coords), 100)
            ax.plot(x_trend, p(x_trend), linestyle='--',
                    color=ChartStyle.SECONDARY_COLOR, linewidth=2,
                    alpha=0.6, label='Trend Line')

        ax.set_title(params["title"], fontsize=ChartStyle.TITLE_FONT_SIZE,
                     fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel(params.get("x_label", "X Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                      fontweight='semibold', color='#34495E')
        ax.set_ylabel(params.get("y_label", "Y Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                      fontweight='semibold', color='#34495E')
        ax.grid(True, alpha=ChartStyle.GRID_ALPHA, linestyle=ChartStyle.GRID_LINESTYLE,
                linewidth=ChartStyle.GRID_LINEWIDTH, color='gray')
        ax.set_axisbelow(True)
        if len(x_coords) > 1:
            legend = ax.legend(loc='best', framealpha=0.95, edgecolor='#BDC3C7',
                               fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
        ax.tick_params(axis='both', which='major',
                       labelsize=ChartStyle.TICK_FONT_SIZE, colors='#34495E')
        self._add_watermark(ax)
        plt.tight_layout()
        return self._save_figure(fig, "scatter")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text
        try:
            params = json.loads(raw_text)
            if "title" not in params:
                yield self._make_response("Error: Missing 'title'", channel); return
            if "data" not in params:
                yield self._make_response("Error: Missing 'data'", channel); return
            data = params["data"]
            if "points" not in data:
                yield self._make_response("Error: 'data' must contain 'points'", channel); return
            if not isinstance(data["points"], list) or len(data["points"]) == 0:
                yield self._make_response("Error: 'points' must be a non-empty list", channel); return
            for idx, point in enumerate(data["points"]):
                if "x" not in point or "y" not in point:
                    yield self._make_response(f"Error: point[{idx}] missing 'x' or 'y'", channel); return
                if not isinstance(point["x"], (int, float)) or not isinstance(point["y"], (int, float)):
                    yield self._make_response(f"Error: point[{idx}] coordinates must be numeric", channel); return
            filepath = self._create_scatter_plot(params)
            yield self._make_response(self._format_response("scatter", params, filepath), channel)
        except json.JSONDecodeError as e:
            yield self._make_response(f"Error: Invalid JSON - {str(e)}", channel)
        except Exception as e:
            yield self._make_response(f"Error: {str(e)}", channel)

class ComboChartTool(BasePlotTool):

    def __init__(self, name: str = "ComboChart", output_dir: Optional[str] = None):
        assert name == "ComboChart"
        super().__init__(output_dir=output_dir)
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "ComboChart"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        Use this tool to create COMBO CHARTS (Bar + Line) for comparing two different metrics.

        Input Format:
        {
            "title": "Publication Count vs Average Citations",
            "x_label": "Year",
            "y1_label": "Publication Count",
            "y2_label": "Average Citations",
            "data": {
                "labels": [2018, 2019, 2020, 2021, 2022],
                "bars": {"label": "Publications", "values": [10, 15, 20, 25, 30]},
                "line": {"label": "Avg Citations",  "values": [45, 52, 48, 60, 65]}
            }
        }
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _create_combo_chart(self, params: dict) -> str:
        fig, ax1 = plt.subplots(figsize=ChartStyle.FIGURE_SIZE)
        data = params["data"]
        labels = data["labels"]
        bars_data = data["bars"]
        line_data = data["line"]

        color1 = ChartStyle.PRIMARY_COLOR
        x_pos = np.arange(len(labels))
        bars = ax1.bar(x_pos, bars_data["values"],
                       width=ChartStyle.BAR_WIDTH, alpha=0.7,
                       color=color1, edgecolor='white',
                       linewidth=ChartStyle.BAR_EDGE_WIDTH,
                       label=bars_data["label"])

        ax1.set_xlabel(params.get("x_label", "X Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                       fontweight='semibold', color='#34495E')
        ax1.set_ylabel(params.get("y1_label", "Y1 Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                       fontweight='semibold', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=ChartStyle.TICK_FONT_SIZE)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels)

        ax2 = ax1.twinx()
        color2 = ChartStyle.ACCENT_COLOR
        ax2.plot(x_pos, line_data["values"], color=color2,
                 marker='o', markersize=10, linewidth=3,
                 markeredgecolor='white', markeredgewidth=2,
                 label=line_data["label"])

        ax2.set_ylabel(params.get("y2_label", "Y2 Axis"), fontsize=ChartStyle.LABEL_FONT_SIZE,
                       fontweight='semibold', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=ChartStyle.TICK_FONT_SIZE)

        ax1.set_title(params["title"], fontsize=ChartStyle.TITLE_FONT_SIZE,
                      fontweight='bold', pad=20, color='#2C3E50')
        ax1.grid(True, alpha=ChartStyle.GRID_ALPHA,
                 linestyle=ChartStyle.GRID_LINESTYLE, linewidth=ChartStyle.GRID_LINEWIDTH)
        ax1.set_axisbelow(True)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                            loc='upper left', framealpha=0.95,
                            edgecolor='#BDC3C7', fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')

        fig.text(0.99, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=7, color='gray', alpha=0.5, ha='right', va='bottom')
        fig.tight_layout()
        return self._save_figure(fig, "combo")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text
        try:
            params = json.loads(raw_text)
            if "title" not in params:
                yield self._make_response("Error: Missing 'title'", channel); return
            if "data" not in params:
                yield self._make_response("Error: Missing 'data'", channel); return
            data = params["data"]
            for field in ["labels", "bars", "line"]:
                if field not in data:
                    yield self._make_response(f"Error: 'data' must contain '{field}'", channel); return
            if "label" not in data["bars"] or "values" not in data["bars"]:
                yield self._make_response("Error: 'bars' must contain 'label' and 'values'", channel); return
            if "label" not in data["line"] or "values" not in data["line"]:
                yield self._make_response("Error: 'line' must contain 'label' and 'values'", channel); return
            num_labels = len(data["labels"])
            if len(data["bars"]["values"]) != num_labels or len(data["line"]["values"]) != num_labels:
                yield self._make_response("Error: All data arrays must have the same length", channel); return
            filepath = self._create_combo_chart(params)
            yield self._make_response(self._format_response("combo", params, filepath), channel)
        except json.JSONDecodeError as e:
            yield self._make_response(f"Error: Invalid JSON - {str(e)}", channel)
        except Exception as e:
            yield self._make_response(f"Error: {str(e)}", channel)
