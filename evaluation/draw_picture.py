import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PlotToolNew import LineChartTool, BarChartTool, PieChartTool, ScatterPlotTool
from openai_harmony import Message, Author, Role, TextContent
import logging
from openai import AsyncOpenAI
import os
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMChartDataConverter:

    def __init__(self, api_key=None, base_url=None, model="qwen3-max"):
        api_key = api_key or os.getenv("API_KEY")
        base_url = base_url or os.getenv("BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model

        self.chart_formats = {
            "BarChart": {
                "title": "Chart Title",
                "x_label": "X-axis Label",
                "y_label": "Y-axis Label",
                "data": {"labels": ["A", "B", "C"], "values": [10, 20, 30]},
                "orientation": "vertical"
            },
            "LineChart": {
                "title": "Chart Title",
                "x_label": "X-axis Label",
                "y_label": "Y-axis Label",
                "data": {
                    "labels": [2020, 2021, 2022],
                    "datasets": [{"label": "Series 1", "values": [10, 20, 30]}]
                }
            },
            "PieChart": {
                "title": "Chart Title",
                "data": {"labels": ["A", "B", "C"], "values": [30, 40, 30]}
            },
            "ScatterPlot": {
                "title": "Chart Title",
                "x_label": "X-axis Label",
                "y_label": "Y-axis Label",
                "data": {
                    "points": [
                        {"x": 1, "y": 10, "label": "Point A"},
                        {"x": 2, "y": 20, "label": "Point B"}
                    ]
                }
            }
        }

    def get_conversion_prompt(self, chart_type, raw_data, question=""):
        standard_format = self.chart_formats.get(chart_type, {})
        format_str = json.dumps(standard_format, indent=2)
        raw_data_str = json.dumps(raw_data, indent=2)
        prompt = f"""You are a data format converter. Convert the given raw chart data into the standard format for a {chart_type}.

Standard Format for {chart_type}:
{format_str}

Raw Data to Convert:
{raw_data_str}

Context (Question):
{question if question else "No additional context"}

Instructions:
1. Extract all relevant data from the raw data
2. Map it to the standard format exactly as shown above
3. For the title, use a concise version of the question (max 60 chars) or create a descriptive title
4. Ensure all data arrays have matching lengths
5. Preserve all numerical values accurately
6. If the raw data has labels/categories, use them; don't create fake data
7. If data is missing or incomplete, return null

Important Rules:
- For BarChart: labels and values must have the SAME length
- For LineChart: each dataset's values must match the length of labels
- For PieChart: labels and values must have the SAME length
- For ScatterPlot: each point must have x, y, and optionally label

Output Requirements:
- Return ONLY valid JSON in the standard format
- No explanations, no markdown code blocks, just pure JSON
- If conversion is impossible, return: {{"error": "reason"}}

Convert now:"""
        return prompt

    async def convert_with_llm(self, chart_type, chart_data, question="", max_retries=2):
        prompt = self.get_conversion_prompt(chart_type, chart_data, question)
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise data format converter. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                result_text = response.choices[0].message.content.strip()
                if result_text.startswith("```"):
                    lines = result_text.split('\n')
                    result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_text
                    result_text = result_text.replace("```json", "").replace("```", "").strip()
                converted_data = json.loads(result_text)
                if "error" in converted_data:
                    logger.warning(f"LLM conversion error: {converted_data['error']}")
                    return None
                if self.validate_converted_data(chart_type, converted_data):
                    return converted_data
                else:
                    logger.warning(f"Validation failed for {chart_type} (attempt {attempt + 1})")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Error in LLM conversion (attempt {attempt + 1}): {e}")
        return None

    def validate_converted_data(self, chart_type, data):
        try:
            if chart_type == "BarChart":
                if "data" not in data: return False
                labels = data["data"].get("labels", [])
                values = data["data"].get("values", [])
                return len(labels) == len(values) and len(labels) > 0
            elif chart_type == "LineChart":
                if "data" not in data: return False
                labels = data["data"].get("labels", [])
                datasets = data["data"].get("datasets", [])
                if len(labels) == 0 or len(datasets) == 0: return False
                return all(len(ds.get("values", [])) == len(labels) for ds in datasets)
            elif chart_type == "PieChart":
                if "data" not in data: return False
                labels = data["data"].get("labels", [])
                values = data["data"].get("values", [])
                return len(labels) == len(values) and len(labels) > 0
            elif chart_type == "ScatterPlot":
                if "data" not in data: return False
                points = data["data"].get("points", [])
                if len(points) == 0: return False
                return all("x" in p and "y" in p for p in points)
            return True
        except Exception:
            return False

class DatasetChartGenerator:

    TOOL_CLASSES = {
        "BarChart": BarChartTool,
        "LineChart": LineChartTool,
        "PieChart": PieChartTool,
        "ScatterPlot": ScatterPlotTool,
    }

    def __init__(self, dataset_path, use_llm=True,
                 api_key=None, base_url=None,
                 debug=False, output_dir="./generated_charts",
                 max_concurrency=5):
        self.dataset_path = Path(dataset_path)
        self.use_llm = use_llm
        self.debug = debug
        self.output_dir = Path(output_dir)
        self.max_concurrency = max_concurrency

        if debug:
            logger.setLevel(logging.DEBUG)

        if use_llm:
            self.llm_converter = LLMChartDataConverter(api_key=api_key, base_url=base_url)
        else:
            self.llm_converter = None

        self.stats = {
            "total_records": 0, "total_qas": 0,
            "charts_generated": 0, "conversion_failures": 0,
            "generation_failures": 0, "by_type": {}, "error_types": {}
        }
        self.failed_cases: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _make_qa_dir(self, index: int, chart_type: str, subgraph_id: str) -> Path:
        safe_id = str(subgraph_id).replace("/", "_").replace(" ", "_")[:50]
        dir_name = f"{index:04d}_{chart_type}_{safe_id}"
        qa_dir = self.output_dir / dir_name
        qa_dir.mkdir(parents=True, exist_ok=True)
        return qa_dir

    def _save_qa_metadata(self, qa_dir: Path, qa: Dict[str, Any],
                          converted_data: Optional[Dict[str, Any]]):
        (qa_dir / "question.txt").write_text(
            qa.get("question", ""), encoding="utf-8")
        (qa_dir / "answer.txt").write_text(
            qa.get("answer", ""), encoding="utf-8")

        with open(qa_dir / "chart_data_original.json", "w", encoding="utf-8") as f:
            json.dump(qa["chart_data"], f, indent=2, ensure_ascii=False)

        if converted_data is not None:
            with open(qa_dir / "chart_data_converted.json", "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)

        metadata = {
            "chart_type": qa["chart_type"],
            "subgraph_id": qa.get("subgraph_id", ""),
            "source": qa.get("source", ""),
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "has_converted_data": converted_data is not None,
        }
        with open(qa_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_dataset(self, limit=None):
        records = []
        if not self.dataset_path.exists():
            logger.error(f"Dataset file not found: {self.dataset_path}")
            return records
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if limit and len(records) >= limit: break
                line = line.strip()
                if not line: continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
        logger.info(f"Loaded {len(records)} records from dataset")
        return records

    def extract_qas(self, record):
        qas = []
        if "qas" in record:
            for qa_item in record["qas"]:
                if "qa" in qa_item:
                    qa = qa_item["qa"]
                    if "chart_type" in qa and "chart_data" in qa:
                        qas.append({
                            "question": qa.get("question", ""),
                            "answer": qa.get("answer", ""),
                            "chart_type": qa["chart_type"],
                            "chart_data": qa["chart_data"],
                            "subgraph_id": record.get("subgraph_id", "unknown"),
                            "source": "qa"
                        })
                if "qa_history" in qa_item:
                    for idx, hist_qa in enumerate(qa_item["qa_history"]):
                        if "chart_type" in hist_qa and "chart_data" in hist_qa:
                            qas.append({
                                "question": hist_qa.get("question", ""),
                                "answer": hist_qa.get("answer", ""),
                                "chart_type": hist_qa["chart_type"],
                                "chart_data": hist_qa["chart_data"],
                                "subgraph_id": record.get("subgraph_id", "unknown"),
                                "source": f"qa_history_{idx}"
                            })
        return qas

    async def generate_chart(self, qa: Dict[str, Any], index: int) -> bool:
        chart_type = qa["chart_type"]
        subgraph_id = qa.get("subgraph_id", "unknown")

        qa_dir = self._make_qa_dir(index, chart_type, subgraph_id)

        try:

            converted_data = None
            if self.use_llm and self.llm_converter:
                converted_data = await self.llm_converter.convert_with_llm(
                    chart_type, qa["chart_data"], qa.get("question", ""))

            self._save_qa_metadata(qa_dir, qa, converted_data)

            if converted_data is None:
                logger.warning(f"[{index}] Conversion failed → {qa_dir.name}")
                (qa_dir / "STATUS_CONVERSION_FAILED").touch()
                async with self._lock:
                    self.stats["conversion_failures"] += 1
                return False

            tool_cls = self.TOOL_CLASSES.get(chart_type)
            if not tool_cls:
                logger.error(f"No tool for chart type: {chart_type}")
                return False

            tool = tool_cls(name=chart_type, output_dir=str(qa_dir))

            content = TextContent(text=json.dumps(converted_data))
            author = Author(role=Role.USER, name="dataset_generator")
            message = Message(author=author, content=[content])

            chart_generated = False
            async for response in tool._process(message):
                response_text = response.content[0].text
                if "successfully" in response_text.lower():
                    chart_generated = True
                    (qa_dir / "STATUS_SUCCESS").touch()
                    logger.info(f"[{index}] ✓ → {qa_dir.name}")
                else:
                    logger.warning(f"[{index}] Tool: {response_text[:120]}")
                    (qa_dir / "STATUS_GENERATION_FAILED").touch()

            async with self._lock:
                if not chart_generated:
                    self.stats["generation_failures"] += 1

            return chart_generated

        except Exception as e:
            logger.error(f"[{index}] Exception: {e}")
            (qa_dir / "STATUS_EXCEPTION").touch()
            async with self._lock:
                self.stats["error_types"][str(e)] =\
                    self.stats["error_types"].get(str(e), 0) + 1
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    async def _process_single_qa(self, idx, total, qa, semaphore):
        async with semaphore:
            chart_type = qa["chart_type"]
            logger.info(f"[{idx}/{total}] {chart_type} | {qa.get('subgraph_id', '?')}")
            success = await self.generate_chart(qa, idx)
            async with self._lock:
                if success:
                    self.stats["charts_generated"] += 1
                    self.stats["by_type"][chart_type] =\
                        self.stats["by_type"].get(chart_type, 0) + 1

    async def generate_all_charts(self, limit=None):
        records = self.load_dataset(limit)
        self.stats["total_records"] = len(records)
        if not records:
            logger.error("No records loaded."); return

        all_qas = []
        for record in records:
            all_qas.extend(self.extract_qas(record))
        self.stats["total_qas"] = len(all_qas)
        logger.info(f"Found {len(all_qas)} charts to generate")

        semaphore = asyncio.Semaphore(self.max_concurrency)
        total = len(all_qas)
        tasks = [
            asyncio.create_task(self._process_single_qa(idx, total, qa, semaphore))
            for idx, qa in enumerate(all_qas, 1)
        ]

        done_count = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                async with self._lock:
                    rate = (self.stats['charts_generated'] / done_count * 100) if done_count else 0
                    logger.info(
                        f"Progress {done_count}/{total} | "
                        f"OK {self.stats['charts_generated']} ({rate:.1f}%) | "
                        f"ConvFail {self.stats['conversion_failures']} | "
                        f"GenFail {self.stats['generation_failures']}")

        self.print_final_stats()

    def print_final_stats(self):
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"  Records   : {self.stats['total_records']}")
        logger.info(f"  QAs       : {self.stats['total_qas']}")
        logger.info(f"  Charts OK : {self.stats['charts_generated']}")
        logger.info(f"  Conv Fail : {self.stats['conversion_failures']}")
        logger.info(f"  Gen  Fail : {self.stats['generation_failures']}")
        if self.stats['total_qas'] > 0:
            rate = self.stats['charts_generated'] / self.stats['total_qas'] * 100
            logger.info(f"  Rate      : {rate:.2f}%")
        logger.info("  By type:")
        for t, c in sorted(self.stats['by_type'].items()):
            logger.info(f"    {t}: {c}")
        logger.info(f"\n  Output: {self.output_dir}")
        logger.info("=" * 60)

async def main():
    dataset_path = "<REDACTED_PATH>"

    generator = DatasetChartGenerator(
        dataset_path=dataset_path,
        use_llm=True,
        debug=False,
        output_dir="./generated_charts_2",
        max_concurrency=5
    )

    await generator.generate_all_charts()

if __name__ == "__main__":
    asyncio.run(main())
