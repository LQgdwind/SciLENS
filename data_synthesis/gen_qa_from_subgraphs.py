import os
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from copy import deepcopy
from tqdm import tqdm

from openai import OpenAI, RateLimitError, APIError

TEXT_SUBGRAPHS = None
OUTPUT_QA = None

LLM_API_KEY = os.environ.get("MATRIX_API_KEY", "<REDACTED_API_KEY>")
LLM_BASE_URL = "<REDACTED_URL>"

LLM_MODEL = ""

MAX_TURNS = 3
MAX_QA_PER_SUBGRAPH = 4
MAX_SUBGRAPHS: Optional[int] = None

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

def call_llm(messages,
             purpose: str = "",
             temperature: float = 0.3,
             max_tokens: int = 4096,
             max_retries: int = 4,
             backoff_base: float = 0.5) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[LLM] ({purpose}) attempt {attempt}/{max_retries} ...")
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
            )
            choices = getattr(resp, "choices", None)
            if not choices:
                print(f"[WARN] call_llm({purpose}): resp.choices is empty.")
                return ""
            msg = choices[0].message
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                print(f"[WARN] call_llm({purpose}): message.content is not str.")
                return ""
            print(f"[LLM] ({purpose}) success, content length={len(content)}")
            return content.strip()

        except RateLimitError as e:
            print(f"[WARN] call_llm({purpose}) RateLimitError: {repr(e)}")
            if attempt < max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"[LLM] ({purpose}) sleep {sleep_time:.1f}s before retry")
                time.sleep(sleep_time)
                continue
            else:
                return ""

        except APIError as e:
            print(f"[WARN] call_llm({purpose}) APIError: {repr(e)}")
            if attempt < max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"[LLM] ({purpose}) sleep {sleep_time:.1f}s before retry")
                time.sleep(sleep_time)
                continue
            else:
                return ""

        except Exception as e:
            print(f"[WARN] call_llm({purpose}) unexpected error: {repr(e)}")
            if attempt < max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"[LLM] ({purpose}) sleep {sleep_time:.1f}s before retry")
                time.sleep(sleep_time)
                continue
            else:
                return ""

def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            t = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(t)
    except Exception:
        return None

@dataclass
class KGEdge:
    s: str
    p: str
    o: str

@dataclass
class PaperSubgraph:
    subgraph_id: Any
    entities: Dict[str, Dict[str, Any]]
    properties: Dict[str, Dict[str, Any]]
    edges: List[KGEdge]

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "PaperSubgraph":
        sg_id = raw.get("subgraph_id")
        entities = raw.get("entities", {})
        properties = raw.get("properties", {})
        edges_raw = raw.get("edges", [])
        edges = [KGEdge(e["s"], e["p"], e["o"]) for e in edges_raw]
        return cls(subgraph_id=sg_id, entities=entities, properties=properties, edges=edges)

@dataclass
class AgentMemory:
    qa: Dict[str, str] = field(default_factory=lambda: {"question": None, "answer": None})
    statements: List[str] = field(default_factory=list)
    used_edges: List[KGEdge] = field(default_factory=list)
    used_entities: List[str] = field(default_factory=list)
    edit_history: List[str] = field(default_factory=list)
    qa_history: List[Dict[str, Any]] = field(default_factory=list)

    def statements_repr(self) -> str:
        return "\n".join(self.statements)

    def dict(self) -> Dict[str, Any]:
        return dict(
            qa=self.qa,
            statements=self.statements,
            used_edges=[dict(s=e.s, p=e.p, o=e.o) for e in self.used_edges],
            used_entities=self.used_entities,
            edit_history=self.edit_history,
            qa_history=self.qa_history,
        )

class PaperKGAsearchPrompts:

    base_qa_prefix = """You are an agent for constructing question–answer pairs from a small citation subgraph of scientific papers.

You are given a target paper with a title and abstract, plus some citation edges.

Each edge is printed as:
    paper_id --[relation / relation_summary]--> paper_id

You must NOT use external knowledge beyond the paper summaries and edges shown.

Task:
- Propose ONE clear scientific question in English about this target paper or closely related papers.
- The question must be solvable and have a unique answer from the given summaries and edges.
- Prefer questions that check scientific claims, experimental findings, or relationships between cited works.
- The question should NOT be yes/no; use wh-questions.
- Provide:
  - "question": the question,
  - "answer": a short answer，and it should be an unambiguous phrase.
  - "statement": a declarative sentence describing the fact behind the QA.

Target paper:
ID: {pid}
Summary:
{summary}

Related citation edges:
{edges_block}

Output JSON only:
"""

    base_qa_json_hint = '{"question": "...", "answer": "...", "statement": "..."}'

    link_qa_prefix = """You are an agent for constructing question–answer pairs from a small citation subgraph of scientific papers.

You are given two sets from the same subgraph:

Set A (candidate answer papers):
```json
{entities_a}
```

Set B (context: papers and citation edges):
```json
{context_b}
```

Use ONLY this information.

Task:
- Construct ONE question–answer pair that connects information from A and B.
- The question context should mainly involve papers in set B.
- The answer must be one paper from set A (mention by its ID or a short description).
- Ensure the question is clear, solvable, and has a unique answer from the given information.
- Provide one supporting statement that links the answer with the context.

Output JSON only:
"""

    link_qa_json_hint = '{"question": "...", "answer": "...", "statement": "..."}'

    compose_qa_prefix = """You are an agent for composing more challenging scientific questions.
You are given two question–answer pairs:

First QA (primary):
{qa_a}

Second QA (auxiliary, providing extra papers/context):
{qa_b}

Supporting statements:
{statements}

Goal:
- Write ONE new question whose answer is EXACTLY the same as the first QA's answer.
- The new question MUST mention at least one key paper or fact from the second QA.
- Do NOT simply copy or lightly rephrase the first question.
- Do NOT change the answer.
- The question must be clear and solvable from the supporting statements.

Output JSON only:
"""
    compose_qa_json_hint = '{"question": "...", "answer": "...", "note": "..."}'

    qa_valid_check_prefix = """Check if the following question–answer pair is valid under the given statements.

A QA is valid if:
1) It is a single, coherent question (not two independent questions).
2) The provided answer is the only correct answer.
3) The answer can be derived from the statements.

Question:
{question}

Statements:
{statements}

Think briefly, then output JSON only:
"""

    qa_valid_check_json_hint = '{"judgement": "yes"} or {"judgement": "no"}'

    direct_gen_check = """Answer the following question based ONLY on the supporting statements.

Question:
{question}

Supporting statements:
{statements}

Provide your final answer enclosed in <answer> </answer> tags and nothing else."""

    llm_judge = """You are an evaluation assistant. Determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Supporting statements:
{statements}

Labeled Answer: {gt_answer}
Predicted Answer: {pred_answer}

Respond with "Correct" if they are equivalent, or "Incorrect" otherwise. Nothing else."""

class PaperKGAsearchAgent:
    def __init__(self, sg: PaperSubgraph, max_turns: int = MAX_TURNS):
        self.sg = sg
        self.max_turns = max_turns

    def construct_base_qa(self) -> Optional[Dict[str, Any]]:
        if not self.sg.entities:
            return None
        root_pid = random.choice(list(self.sg.entities.keys()))
        root_info = self.sg.entities[root_pid]
        summary = (root_info.get("summary") or "").strip()
        if not summary:
            summary = f"(No summary available for {root_pid}.)"

        rel_edges = [e for e in self.sg.edges if e.s == root_pid or e.o == root_pid]
        random.shuffle(rel_edges)
        rel_edges = rel_edges[: min(6, len(rel_edges))]

        edges_lines = []
        for e in rel_edges:
            p_info = self.sg.properties.get(e.p, {})
            p_sum = (p_info.get("summary") or "").strip()
            p_sum_short = p_sum.split("\n")[0][:160] if p_sum else ""
            edges_lines.append(f"{e.s} --[{e.p} / {p_sum_short}]--> {e.o}")
        edges_block = "\n".join(edges_lines) if edges_lines else "(no edges)"

        prefix = PaperKGAsearchPrompts.base_qa_prefix.format(
            pid=root_pid,
            summary=summary,
            edges_block=edges_block,
        )
        prompt = prefix + "\n" + PaperKGAsearchPrompts.base_qa_json_hint

        content = call_llm(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            purpose="base_qa",
        )
        data = parse_json_block(content)
        if not data:
            return None
        if not all(k in data for k in ("question", "answer", "statement")):
            return None
        data["_root_pid"] = root_pid
        data["_used_edges"] = rel_edges
        return data

    def construct_link_qa(self, memory: AgentMemory) -> Optional[Dict[str, Any]]:
        all_pids = set(self.sg.entities.keys())
        used_pids = set(memory.used_entities)
        candidate_pids = list(all_pids - used_pids)
        if len(candidate_pids) < 1:
            return None

        random.shuffle(candidate_pids)
        pids_a = candidate_pids[: min(3, len(candidate_pids))]
        entities_a = []
        for pid in pids_a:
            ent = self.sg.entities.get(pid, {})
            entities_a.append({
                "paper_id": pid,
                "summary": ent.get("summary", ""),
            })

        edges_pool = [e for e in self.sg.edges if e not in memory.used_edges]
        if len(edges_pool) == 0:
            return None
        random.shuffle(edges_pool)
        edges_b = edges_pool[: min(5, len(edges_pool))]
        context_entities: Dict[str, Any] = {}
        for e in edges_b:
            for pid in [e.s, e.o]:
                if pid not in context_entities and pid in self.sg.entities:
                    context_entities[pid] = {
                        "paper_id": pid,
                        "summary": self.sg.entities[pid].get("summary", ""),
                    }
        context_b = {
            "papers": list(context_entities.values()),
            "edges": [dict(s=e.s, p=e.p, o=e.o) for e in edges_b],
        }

        prefix = PaperKGAsearchPrompts.link_qa_prefix.format(
            entities_a=json.dumps(entities_a, ensure_ascii=False, indent=2),
            context_b=json.dumps(context_b, ensure_ascii=False, indent=2),
        )
        prompt = prefix + "\n" + PaperKGAsearchPrompts.link_qa_json_hint

        content = call_llm(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            purpose="link_qa",
        )
        data = parse_json_block(content)
        if not data or not all(k in data for k in ("question", "answer", "statement")):
            return None
        data["_used_edges"] = edges_b
        return data

    def compose_qa(self,
                   qa_a: Dict[str, Any],
                   qa_b: Dict[str, Any],
                   memory: AgentMemory) -> Optional[Dict[str, Any]]:
        qa_a_json = json.dumps(
            {"question": qa_a["question"], "answer": qa_a["answer"]},
            ensure_ascii=False,
        )
        qa_b_json = json.dumps(
            {"question": qa_b["question"], "answer": qa_b["answer"]},
            ensure_ascii=False,
        )
        stmts = memory.statements_repr() + "\n" + qa_b["statement"]

        prefix = PaperKGAsearchPrompts.compose_qa_prefix.format(
            qa_a=qa_a_json,
            qa_b=qa_b_json,
            statements=stmts,
        )
        prompt = prefix + "\n" + PaperKGAsearchPrompts.compose_qa_json_hint

        content = call_llm(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            purpose="compose_qa",
        )
        data = parse_json_block(content)
        if not data or not all(k in data for k in ("question", "answer", "note")):
            return None
        if data["answer"] != qa_a["answer"]:
            return None

        q_old = qa_a["question"].strip()
        q_new = data["question"].strip()
        if len(q_new) < len(q_old) + 5:
            return None

        return data

    def check_qa_valid(self, memory: AgentMemory) -> bool:
        stmts_full = memory.statements_repr()

        prefix = PaperKGAsearchPrompts.qa_valid_check_prefix.format(
            question=memory.qa["question"],
            statements=stmts_full,
        )
        prompt = prefix + "\n" + PaperKGAsearchPrompts.qa_valid_check_json_hint

        content = call_llm(
            [
                {"role": "system", "content": "You are a careful evaluator."},
                {"role": "user", "content": prompt},
            ],
            purpose="qa_valid_check",
        )

        data = parse_json_block(content)
        if data and "judgement" in data:
            return "yes" in str(data["judgement"]).lower()

        low = content.lower()
        if "yes" in low and "no" not in low:
            return True
        if "no" in low and "yes" not in low:
            return False
        return False

    def direct_generate(self, memory: AgentMemory, n: int = 4) -> List[Optional[str]]:
        prompt = PaperKGAsearchPrompts.direct_gen_check.format(
            question=memory.qa["question"],
            statements=memory.statements_repr(),
        )
        answers = []
        for i in range(n):
            content = call_llm(
                [
                    {"role": "system", "content": "You must answer strictly based on the statements."},
                    {"role": "user", "content": prompt},
                ],
                purpose=f"direct_gen_{i+1}",
            )
            ans = None
            if "<answer>" in content and "</answer>" in content:
                ans = content.split("<answer>")[-1].split("</answer>")[0].strip()
            answers.append(ans)
        return answers

    def judge_answers(self, memory: AgentMemory, preds: List[Optional[str]]) -> List[bool]:
        results = []
        for i,pa in enumerate(preds):
            if pa is None:
                results.append(False)
                continue
            prompt = PaperKGAsearchPrompts.llm_judge.format(
                question=memory.qa["question"],
                statements=memory.statements_repr(),
                gt_answer=memory.qa["answer"],
                pred_answer=str(pa),
            )
            content = call_llm(
                [
                    {"role": "system", "content": "You are a precise judge."},
                    {"role": "user", "content": prompt},
                ],
                purpose=f"judge_{i+1}",
            )
            correct = "Correct" in content
            results.append(correct)
        return results

    def run(self) -> Optional[AgentMemory]:
        if not self.sg.edges or not self.sg.entities:
            return None

        memory = AgentMemory()

        base_qa = self.construct_base_qa()
        if not base_qa:
            return None

        memory.qa["question"] = base_qa["question"]
        memory.qa["answer"] = base_qa["answer"]
        memory.statements.append(base_qa["statement"])
        memory.used_edges.extend(base_qa["_used_edges"])
        memory.used_entities.append(base_qa["_root_pid"])
        memory.edit_history.append(
            f"Create base QA on {base_qa['_root_pid']}: Q={base_qa['question']} A={base_qa['answer']}"
        )
        memory.qa_history.append(
            dict(question=base_qa["question"], answer=base_qa["answer"], direct_gen_acc=None)
        )

        ready_to_exit = False

        for turn in range(1, self.max_turns + 1):
            link_qa = self.construct_link_qa(memory)
            if not link_qa:
                break

            combined = self.compose_qa(memory.qa, link_qa, memory)
            if not combined:
                continue

            memory_new = deepcopy(memory)
            q_new = combined["question"]
            memory_new.qa["question"] = q_new
            memory_new.qa["answer"] = memory.qa["answer"]
            memory_new.statements.append(link_qa["statement"])
            memory_new.used_edges.extend(link_qa["_used_edges"])
            memory_new.edit_history.append(
                f"Turn {turn}: add link QA and compose. Note: {combined['note']}"
            )

            valid = self.check_qa_valid(memory_new)
            memory_new.edit_history.append(f"QA valid check judgement={valid}")

            answers = self.direct_generate(memory_new, n=4)
            corrects = self.judge_answers(memory_new, answers)

            memory_new.qa_history.append(
                dict(
                    question=memory_new.qa["question"],
                    answer=memory_new.qa["answer"],
                    direct_gen_acc=f"{sum(corrects)}/{len(corrects)}",
                )
            )
            memory_new.edit_history.append(
                f"[INFO] direct gen answers & corrects: {[(a, c) for a, c in zip(answers, corrects)]}"
            )

            if not any(corrects):
                ready_to_exit = True
                memory_new.edit_history.append(
                    f"[INFO] turn {turn}: model fails all direct generations, ready_to_exit=True"
                )

            memory = memory_new
            if ready_to_exit:
                break

        return memory

def load_paper_subgraphs(path: Path, limit: Optional[int] = None) -> List[PaperSubgraph]:
    sgs: List[PaperSubgraph] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sgs.append(PaperSubgraph.from_raw(raw))
    return sgs

def main():
    ap = argparse.ArgumentParser(description="Generate QA from paper subgraphs (Asearcher-style)")
    ap.add_argument("--work_dir", default="../KG")
    ap.add_argument("--max_subgraphs", type=int, default=None)
    ap.add_argument("--max_qa_per_subgraph", type=int, default=4)
    ap.add_argument("--max_turns", type=int, default=3)

    ap.add_argument("--llm_model", type=str, default="",
                    help="LLM model name for OpenAI-compatible API")
    ap.add_argument("--llm_base_url", type=str, default="<REDACTED_URL>",
                    help="Base URL for OpenAI-compatible API")
    ap.add_argument("--llm_api_key", type=str, default=None,
                    help="API key for OpenAI-compatible API (default: env MATRIX_API_KEY)")

    ap.add_argument("--subgraph_file", type=str, default="paper_subgraphs_textual.jsonl",
                    help="Subgraph input file name under work_dir")
    ap.add_argument("--output_file", type=str, default=None,
                    help="Output QA file name under work_dir; "
                         "if not set, use {model}_paper_qa_from_subgraphs.jsonl")

    args = ap.parse_args()

    global TEXT_SUBGRAPHS, OUTPUT_QA, MAX_QA_PER_SUBGRAPH, MAX_TURNS
    global LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, client

    LLM_MODEL = args.llm_model
    LLM_BASE_URL = args.llm_base_url
    if args.llm_api_key is not None:
        LLM_API_KEY = args.llm_api_key

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    TEXT_SUBGRAPHS = Path(os.path.join(args.work_dir, args.subgraph_file))

    if args.output_file is not None:
        output_filename = args.output_file
    else:
        safe_model_name = "".join(
            c if (c.isalnum() or c in ("-", "_")) else "_" for c in LLM_MODEL
        )
        output_filename = f"{safe_model_name}_paper_qa_from_subgraphs.jsonl"
    OUTPUT_QA = Path(os.path.join(args.work_dir, output_filename))

    MAX_QA_PER_SUBGRAPH = args.max_qa_per_subgraph
    MAX_TURNS = args.max_turns

    subgraphs = load_paper_subgraphs(TEXT_SUBGRAPHS, args.max_subgraphs)
    print(f"[MAIN] Loaded {len(subgraphs)} paper subgraphs from {TEXT_SUBGRAPHS}")
    print(f"[MAIN] Using LLM model: {LLM_MODEL}")
    print(f"[MAIN] Output QA file: {OUTPUT_QA}")

    with OUTPUT_QA.open("w", encoding="utf-8") as fout:
        for i, sg in enumerate(subgraphs):
            print(f"\n=== Processing subgraph {sg.subgraph_id} ({i+1}/{len(subgraphs)}) ===")
            qa_list = []
            for j in range(MAX_QA_PER_SUBGRAPH):
                print(f"[MAIN] --- Subgraph {sg.subgraph_id}, QA #{j+1}/{MAX_QA_PER_SUBGRAPH} ---")
                agent = PaperKGAsearchAgent(sg, max_turns=MAX_TURNS)
                memory = agent.run()
                if memory is None:
                    qa_list.append({"success": False, "reason": "agent_failed"})
                else:
                    qa_list.append(memory.dict())

            record = {
                "subgraph_id": sg.subgraph_id,
                "qas": qa_list,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[MAIN] Done. QA written to {OUTPUT_QA}")

if __name__ == "__main__":
    main()
