import sys, os
import time, json, uuid
import re
import asyncio
import copy
import traceback
import random
from typing import Any, Dict, List, Optional
from datetime import date
from openai import OpenAI, APIConnectionError, APIError, RateLimitError

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from message_processor import parse_arguments

try:
    from ResearchToolBox import ResearchToolbox
    from openai_harmony import Message as HarmonyMessage, TextContent, Author, Role
except ImportError:
    print("❌ Error: ResearchToolBox.py or openai_harmony not found!")
    pass

Message = dict[str, str]
MessageList = list[Message]

ACADEMIC_SYS_PROMPT = """You are an advanced Qwen assistant integrated into a tool-calling environment.

# Capabilities and constraints

- You are a large language model with a knowledge cutoff at 2025-04.
- You have strong reasoning ability and should perform deliberate, step-by-step thinking before answering.
- You can access external tools via the built-in tool-calling mechanism to obtain up-to-date information, verify facts, and support complex reasoning.
- When you lack enough information, or when questions involve recent events or specific factual details, you MUST actively consider using tools rather than guessing.

# Tools

You have access to the following tools. Choose the most appropriate one based on your need:

1. **KeywordSearch** (Coarse-grained Retrieval):
   - **Use Case**: To find papers in a general domain or topic.
   - **Input**: A list of **1-3** general, high-level domain keywords (e.g., "Graph Neural Networks").

2. **EmbeddingSearch** (Fine-grained Semantic Match):
   - **Use Case**: To find specific papers, methodologies, or findings based on natural language descriptions.
   - **Input**: A detailed natural language query describing the paper's content.

3. **ReferenceSearch** (Outgoing Citations):
   - **Use Case**: Find papers cited BY a specific paper (Papers that the target paper references).
   - **Input**: {"id": "unique_paper_id"} OR {"title": "Exact Title"}

4. **PaperInfo** (Metadata Retrieval):
   - **Use Case**: Retrieve full metadata (Abstract, Venue, Year, Authors) by ID or exact title.
   - **Input**: {"id": "unique_paper_id"} OR {"title": "Exact Title"}

5. **ShortestPath** (Connection Discovery):
   - **Use Case**: Find the shortest citation path connecting two papers (Paper A -> ... -> Paper B).
   - **Input**:
     - Option 1: {"start_id": "ID_A", "end_id": "ID_B", "max_depth": 3}
     - Option 2: {"start_title": "Title A", "end_title": "Title B", "max_depth": 3}

6. **GetKhop** (Citation Network Expansion):
   - **Use Case**: Retrieve the K-hop citation network originating from a paper (Recursive citations).
   - **Input**: {"id": "unique_paper_id", "k": 2} OR {"title": "Exact Title", "k": 2}

7. **GetInDegree** (Incoming Citations / Cited By):
   - **Use Case**: Find papers that cite a specific paper (Who cites this paper?).
   - **Input**: {"id": "unique_paper_id"} OR {"title": "Exact Title"}

8. **Summarize** (Content Summarization):
   - **Use Case**: Generate a concise summary of a paper abstract or any long text segment.
   - **Input**: {"content": "The text content you want summarized..."}

9. **LineChart** (Trend Visualization):
   - **Use Case**: Show trends over time (e.g., citation growth, publication trends).
   - **Input**: {"title": "...", "x_label": "Year", "y_label": "Citations", "data": {"labels": [2019, 2020, ...], "datasets": [{"label": "Paper A", "values": [10, 25, ...]}, ...]}}

10. **BarChart** (Category Comparison):
    - **Use Case**: Compare values across categories (e.g., Top N papers, venue comparison).
    - **Input**: {"title": "...", "x_label": "...", "y_label": "...", "data": {"labels": ["A", "B", ...], "values": [100, 85, ...]}, "orientation": "vertical"}

11. **PieChart** (Proportion Display):
    - **Use Case**: Show distribution and proportions (e.g., field distribution, collaboration ratios).
    - **Input**: {"title": "...", "data": {"labels": ["NLP", "CV", ...], "values": [35, 28, ...]}}

12. **ScatterPlot** (Correlation Analysis):
    - **Use Case**: Analyze relationships between two variables (e.g., citations vs year).
    - **Input**: {"title": "...", "x_label": "...", "y_label": "...", "data": {"points": [{"x": 2019, "y": 45, "label": "Paper A"}, ...]}}

# Tool calling behavior
- The system routes your tool calls automatically.
- You do NOT need to manually construct any special markup like <｜DSML｜function_calls>; follow standard function calling.
- **Strategy**:
  - Start with broad searches (Keyword/Embedding) if the topic is unknown.
  - Use graph tools (ShortestPath, GetKhop, ReferenceSearch, GetInDegree) to trace intellectual lineages.
  - Use Summarize to digest long abstracts if needed.
  - Use visualization tools to present data-driven insights and comparisons.

# Answer style
- Be accurate, concise, and logically coherent.
- If information is uncertain, explicitly mention the uncertainty.
- When appropriate, use visualization tools to enhance your explanations.
- **CRITICAL**: When you have gathered sufficient information and are ready to provide the final response, you MUST wrap your final answer in <answer>...</answer> tags. The system will NOT terminate the conversation until it detects this tag.
"""

class Qwen3EvaluatorTool:
    def __init__(
        self,
        url: str = "",
        model: str = "Qwen3-8B",
        api_key: str = "EMPTY",
        system_message: Optional[str] = None,
        max_tokens: int = 131072,
        enable_thinking: bool = True,
        **kwargs
    ):
        today = date.today().isoformat()
        default_sys = ACADEMIC_SYS_PROMPT
        self.system_message = system_message or default_sys

        self.model = model
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

        self.vllm_ports = [8004, 8005, 8006, 8007]

        self.clients = []
        for port in self.vllm_ports:
            client = OpenAI(
                base_url=f"<REDACTED_URL>",
                api_key="EMPTY"
            )
            self.clients.append(client)
            print(f"✅ Initialized vLLM client for port {port}")

        print(">>> Initializing ResearchToolbox (Connecting Ray & Mongo)...")
        try:
            self.toolbox = ResearchToolbox()
            print("✅ ResearchToolbox Ready.")
        except Exception as e:
            print(f"❌ ResearchToolbox Init Failed: {e}")
            self.toolbox = None

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "EmbeddingSearch",
                    "description": "Semantic search using vector embeddings. Input detailed natural language query.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "KeywordSearch",
                    "description": "Coarse-grained search. Input 1-3 general domain keywords ONLY.",
                    "parameters": {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["keywords"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ReferenceSearch",
                    "description": "Find outgoing citations (papers cited by this paper).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique paper ID"},
                            "title": {"type": "string", "description": "Exact title"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "PaperInfo",
                    "description": "Retrieve full metadata (Abstract, Authors, Venue) by ID or exact title.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique paper ID"},
                            "title": {"type": "string", "description": "Exact title"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ShortestPath",
                    "description": "Find shortest citation path between two papers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_id": {"type": "string"},
                            "end_id": {"type": "string"},
                            "start_title": {"type": "string"},
                            "end_title": {"type": "string"},
                            "max_depth": {"type": "integer", "default": 3}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "GetKhop",
                    "description": "Retrieve K-hop citation network.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "k": {"type": "integer", "default": 2}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "GetInDegree",
                    "description": "Find papers that cite this paper (Incoming citations).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Summarize",
                    "description": "Summarize text.",
                    "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "LineChart",
                    "description": "Create line chart for trend visualization (e.g., citation trends over years).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Chart title"},
                            "x_label": {"type": "string", "description": "X-axis label"},
                            "y_label": {"type": "string", "description": "Y-axis label"},
                            "data": {
                                "type": "object",
                                "properties": {
                                    "labels": {"type": "array", "items": {"type": ["integer", "string"]}, "description": "X-axis values"},
                                    "datasets": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "values": {"type": "array", "items": {"type": "number"}}
                                            },
                                            "required": ["label", "values"]
                                        }
                                    }
                                },
                                "required": ["labels", "datasets"]
                            }
                        },
                        "required": ["title", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "BarChart",
                    "description": "Create bar chart for comparing categories (e.g., Top N papers).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "x_label": {"type": "string"},
                            "y_label": {"type": "string"},
                            "data": {
                                "type": "object",
                                "properties": {
                                    "labels": {"type": "array", "items": {"type": "string"}},
                                    "values": {"type": "array", "items": {"type": "number"}}
                                },
                                "required": ["labels", "values"]
                            },
                            "orientation": {"type": "string", "enum": ["vertical", "horizontal"], "default": "vertical"}
                        },
                        "required": ["title", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "PieChart",
                    "description": "Create pie chart for showing proportions (e.g., field distribution).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "data": {
                                "type": "object",
                                "properties": {
                                    "labels": {"type": "array", "items": {"type": "string"}},
                                    "values": {"type": "array", "items": {"type": "number"}}
                                },
                                "required": ["labels", "values"]
                            }
                        },
                        "required": ["title", "data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ScatterPlot",
                    "description": "Create scatter plot for correlation analysis (e.g., citations vs year).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "x_label": {"type": "string"},
                            "y_label": {"type": "string"},
                            "data": {
                                "type": "object",
                                "properties": {
                                    "points": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "x": {"type": "number"},
                                                "y": {"type": "number"},
                                                "label": {"type": "string"}
                                            },
                                            "required": ["x", "y"]
                                        }
                                    }
                                },
                                "required": ["points"]
                            }
                        },
                        "required": ["title", "data"]
                    }
                }
            }
        ]

    def _get_pruned_messages(self, messages: List[Dict[str, Any]], keep_last: int = 5) -> List[Dict[str, Any]]:
        pruned_messages = copy.deepcopy(messages)

        tool_indices = [i for i, msg in enumerate(pruned_messages) if msg.get("role") == "tool"]

        if len(tool_indices) <= keep_last:
            return pruned_messages

        cutoff_count = len(tool_indices) - keep_last
        indices_to_prune = tool_indices[:cutoff_count]

        for idx in indices_to_prune:
            pruned_messages[idx]["content"] = ""

        return pruned_messages

    def _run_toolbox_sync(self, tool_name: str, tool_args: Dict):
        if not self.toolbox:
            return "Error: ResearchToolbox not initialized."

        async def run():
            payload = {
                "tool_name": tool_name,
                "tool_args": tool_args
            }
            payload_json = json.dumps(payload, ensure_ascii=False)

            msg = HarmonyMessage(
                author=Author(role=Role.ASSISTANT, name="gpt-4"),
                content=[TextContent(text=payload_json)]
            )

            full_response = ""
            try:
                async for chunk in self.toolbox._process(msg):
                    if chunk.content and hasattr(chunk.content[0], 'text'):
                        full_response += chunk.content[0].text
                    else:
                        full_response += str(chunk)
            except Exception as e:
                full_response = f"Toolbox Runtime Error: {e}"

            return full_response

        try:
            return asyncio.run(run())
        except Exception as e:
            return f"Async Execution Error: {str(e)}"

    def _tool_args_valid(self, tool_calls) -> bool:
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name")
            args_str = func.get("arguments", "")

            if isinstance(args_str, str):
                args, is_valid_dict = parse_arguments(args_str)
                if not is_valid_dict:
                    print(f"[WARN] Invalid JSON arguments for tool {name}: {args_str}")
                    return False
            elif not isinstance(args_str, dict):
                print(f"[WARN] Unknown arguments type for tool {name}: {type(args_str)}")
                return False
        return True

    def _execute_tool_calls(self, tool_calls, initial_prompt, session_id):
        tool_outputs = []
        for tc in tool_calls:
            name = tc['function']['name']
            args_str = tc['function']['arguments']

            try:

                if isinstance(args_str, str):
                    args, is_valid_dict = parse_arguments(args_str)

                    if not is_valid_dict:
                        print(f"[WARN] Failed to parse arguments for {name}, keep raw string")

                        if name == "KeywordSearch":
                            args = {"keywords": [args_str]}
                        else:
                            args = {}

                elif isinstance(args_str, dict):

                    args = args_str
                else:
                    print(f"[ERROR] Unknown arguments type: {type(args_str)}")
                    args = {}

            except Exception as e:
                print(f"[ERROR] Parse exception for {name}: {e}")
                traceback.print_exc()
                args = {}

            if not isinstance(args, dict):
                print(f"[WARN] args is not dict, converting to empty dict")
                args = {}

            tool_result = self._run_toolbox_sync(name, args)

            if len(tool_result) > 20000:
                tool_result = tool_result[:20000] + "\n... [Output Truncated due to length]"

            tool_outputs.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tc.get("id")
            })
        return tool_outputs

    def _call_response_api(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:

        client = random.choice(self.clients)

        for attempt in range(1, 10):
            try:

                extra_body = {}
                if self.enable_thinking:
                    extra_body = {
                        "enable_reasoning": True,
                        "reasoning_parser": "deepseek_r1"
                    }

                temperature = 0.6 if self.enable_thinking else 0.7
                top_p = 0.95 if self.enable_thinking else 0.8

                stream = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    tools=self.tools,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body if extra_body else None
                )

                content = []
                thinking_content = []
                tool_calls_dict = {}

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        thinking_content.append(delta.reasoning_content)

                    if delta.content:
                        content.append(delta.content)

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_dict:
                                tool_calls_dict[idx] = tc.to_dict()
                                tool_calls_dict[idx]['function']['arguments'] = ""
                            if tc.function.arguments:
                                tool_calls_dict[idx]['function']['arguments'] += tc.function.arguments

                result = {
                    "role": "assistant",
                    "content": "".join(content),
                    "tool_calls": [v for k,v in sorted(tool_calls_dict.items())] if tool_calls_dict else []
                }

                if thinking_content:
                    result["thinking"] = "".join(thinking_content)

                return result

            except Exception as e:

                msg = str(e)
                if "maximum context length" in msg:
                    print(f"[API Error] Fatal context length error, no retry: {e}")
                    raise RuntimeError("Context length exceeded")
                print(f"[API Error] attempt {attempt}: {e}")
                time.sleep(1)

        raise RuntimeError("API Failed after retries")

    def _parse_tool_calls_from_text(self, text: str) -> List[Dict]:
        import re

        tool_calls = []
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:

                func_match = re.match(r'([^\n{]+)', match.strip())
                if not func_match:
                    continue
                func_name = func_match.group(1).strip()

                json_match = re.search(r'\{.*\}', match, re.DOTALL)
                if not json_match:
                    continue
                args_str = json_match.group(0)

                tool_calls.append({
                    'id': f"tool-call-{uuid.uuid4()}",
                    'type': 'function',
                    'function': {
                        'name': func_name,
                        'arguments': args_str
                    }
                })
            except Exception as e:
                print(f"[WARN] Failed to parse tool_call: {e}")
                continue

        return tool_calls

    def _parse_thinking_from_content(self, content: str) -> tuple[str, str]:

        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, content, re.DOTALL)
        if match:
            thinking = match.group(1).strip()

            final_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
            return thinking, final_content

        return "", content

    def get_resp(self, message_list, top_p=-1, temperature=-1):
            if self.system_message:
                has_sys = any(m['role'] == 'system' for m in message_list)
                if not has_sys:
                    message_list.insert(0, {"role": "system", "content": self.system_message})

            max_rounds = 800
            round_count = 0

            consecutive_retries = 0
            max_retries = 5

            initial_prompt = message_list[-1]["content"] if message_list else ""

            while True:

                round_count += 1
                if round_count > max_rounds:
                    print("[WARN] Max rounds reached.")
                    break

                try:

                    messages_to_send = self._get_pruned_messages(message_list, keep_last=5)

                    assistant_message = self._call_response_api(messages=messages_to_send)
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")

                    raise

                final_content = assistant_message.get("content", "") or ""

                if "thinking" not in assistant_message and final_content:
                    thinking, clean_content = self._parse_thinking_from_content(final_content)
                    if thinking:
                        assistant_message["thinking"] = thinking
                        assistant_message["content"] = clean_content
                        final_content = clean_content

                tool_calls = assistant_message.get("tool_calls", []) or []

                if tool_calls:
                    if not self._tool_args_valid(tool_calls):

                        if consecutive_retries < max_retries:
                            consecutive_retries += 1
                            print(f"[WARN] Invalid tool arguments, retrying round {consecutive_retries}/{max_retries}...")

                            continue
                        else:
                            print(f"[ERROR] Max retries reached for invalid tool arguments, force using current response.")

                if not tool_calls:
                    if "thinking" in assistant_message:
                        extracted_calls = self._parse_tool_calls_from_text(assistant_message["thinking"])
                        if extracted_calls:
                            print(f"[INFO] Extracted {len(extracted_calls)} tool calls from thinking")
                            tool_calls = extracted_calls

                    if not tool_calls and final_content:
                        extracted_calls = self._parse_tool_calls_from_text(final_content)
                        if extracted_calls:
                            print(f"[INFO] Extracted {len(extracted_calls)} tool calls from content")
                            tool_calls = extracted_calls

                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls

                has_answer_tag = "<answer>" in final_content
                has_tools = len(tool_calls) > 0

                if not has_tools and not has_answer_tag:
                    if consecutive_retries < max_retries:
                        consecutive_retries += 1
                        print(f"\n[WARN] Invalid response detected (No <answer> and no tools). Retrying {consecutive_retries}/{max_retries}...")
                        continue
                    else:
                        print(f"\n[ERROR] Max retries ({max_retries}) reached. Forced to accept current response.")

                consecutive_retries = 0
                message_list.append(assistant_message)

                if tool_calls:
                    print(f"[INFO] Executing {len(tool_calls)} tool calls...")
                    tool_outputs = self._execute_tool_calls(tool_calls, initial_prompt, str(uuid.uuid4()))
                    message_list.extend(tool_outputs)
                    continue

                if has_answer_tag:
                    print(f"[INFO] <answer> tag detected. Finishing conversation.")
                    break

                print(f"[INFO] No tool calls and max retries reached without <answer>. Ending loop.")
                break

            return assistant_message.get("content", "").strip(), message_list

    def __call__(self, message_list, **kwargs):
        return self.get_resp(message_list)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3 Model Evaluation Script")
    parser.add_argument("--mode", default="single", choices=["single", "eval"],
                       help="Evaluation mode: single test or batch evaluation")
    parser.add_argument("--qa_path", default="",
                       help="Path to test set (JSONL format with question/answer pairs)")
    parser.add_argument("--out_path", default="qwen3_eval_results.jsonl",
                       help="Output path for evaluation results")
    parser.add_argument("--enable_thinking", action="store_true", default=True,
                       help="Enable thinking mode (default: True)")
    args = parser.parse_args()

    evaluator = Qwen3EvaluatorTool(enable_thinking=args.enable_thinking)

    if args.mode == "single":

        test_question = "Search for papers about 'BERT' and 'GPT', then create a chart comparing their citation trends from 2018 to 2023."
        print(f"[TEST] Question: {test_question}")
        print(f"[TEST] Thinking mode: {'Enabled' if args.enable_thinking else 'Disabled'}")
        print("")

        try:
            answer, history = evaluator([{"role": "user", "content": test_question}])

            print("\n" + "="*60)
            print("EVALUATION RESULT")
            print("="*60)
            print(answer)
            print("\n" + "="*60)
            print("FULL CONVERSATION TRACE")
            print("="*60)
            print(json.dumps(history, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"[ERROR] Single test failed: {e}")
            traceback.print_exc()

    else:

        if not args.qa_path:
            raise ValueError("--qa_path is required for batch evaluation mode")

        print(f"[EVAL] Starting batch evaluation...")
        print(f"[EVAL] Test set: {args.qa_path}")
        print(f"[EVAL] Output: {args.out_path}")
        print(f"[EVAL] Thinking mode: {'Enabled' if args.enable_thinking else 'Disabled'}")
        print("")

        total_cases = 0
        evaluated_cases = 0

        with open(args.qa_path, "r", encoding="utf-8") as fin,\
             open(args.out_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                total_cases += 1

                question = ""
                gold_answer = ""
                model_answer = ""
                conversations = []
                error_msg = ""

                try:
                    item = json.loads(line)

                    if "qas" in item and isinstance(item["qas"], list) and len(item["qas"]) > 0:
                        qa_wrapper = item["qas"][0].get("qa", {})
                        question = qa_wrapper.get("question", "")
                        gold_answer = qa_wrapper.get("answer", "")

                    if not question:
                        question = item.get("question", "")
                        gold_answer = item.get("answer", "")

                except Exception as e:
                    print(f"[WARN] Skip invalid JSON at line {total_cases}: {e}")
                    continue

                if not question:
                    print(f"[WARN] Empty question at line {total_cases}, skipping")
                    continue

                message_list = [{"role": "user", "content": question}]

                try:
                    model_answer, conversations = evaluator(message_list)

                except Exception as e:
                    print(f"[ERROR] Evaluation failed for case {total_cases}: {e}")
                    error_msg = str(e)
                    conversations = message_list
                    model_answer = "[Evaluation Failed]"

                if (not model_answer) or (model_answer == ""):
                    last_assistant_msg = None
                    for msg in reversed(conversations):
                        if msg.get("role") == "assistant":
                            content = (msg.get("content") or "").strip()
                            if content:
                                last_assistant_msg = content
                                break

                    if last_assistant_msg:
                        print(f"[INFO] Using last assistant message as model_answer for case {total_cases}")
                        model_answer = last_assistant_msg

                if not model_answer or model_answer == "[Evaluation Failed]":
                    print(f"[WARN] Case {total_cases} returned empty/failed response")

                eval_record = {
                    "case_id": total_cases,
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "conversations": conversations,
                    "error": error_msg,
                    "status": "success" if (model_answer and model_answer != "[Evaluation Failed]") else "failed"
                }

                fout.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                evaluated_cases += 1

                if total_cases % 10 == 0:
                    print(f"[PROGRESS] Evaluated {total_cases} cases...")

        print("")
        print("="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print(f"Total cases: {total_cases}")
        print(f"Evaluated: {evaluated_cases}")
        print(f"Output: {args.out_path}")
        print("="*60)
