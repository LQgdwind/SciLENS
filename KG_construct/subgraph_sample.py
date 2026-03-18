import os
import json
import random
import argparse
from typing import Dict, List, Any, Set
from tqdm import tqdm

def sample_centers_from_adj_out(
    adj_out_path: str,
    max_subgraphs: int,
    sample_prob: float,
) -> Dict[str, List[str]]:
    centers = {}
    with open(adj_out_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="scan adj_out"):
            if len(centers) >= max_subgraphs:
                break
            if random.random() > sample_prob:
                continue
            s, ts_str = line.strip().split("\t")
            neighbors = ts_str.split(",") if ts_str else []
            if not neighbors:
                continue
            centers[s] = neighbors
    return centers

def load_summaries_for_nodes(
    summary_path: str,
    needed_ids: Set[str],
    max_summary_len: int = 1200,
) -> Dict[str, Dict[str, Any]]:
    meta = {}
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="scan summaries"):
            if not needed_ids:
                break
            p = json.loads(line)
            pid = p.get("id")
            if pid in needed_ids:
                title = p.get("title", "") or ""
                abstract = p.get("abstract", "") or ""
                summary = (title + "\n" + abstract).strip()
                if len(summary) > max_summary_len:
                    summary = summary[:max_summary_len]
                if not summary:
                    summary = f"(No summary for paper {pid}.)"
                meta[pid] = {"summary": summary}
                needed_ids.remove(pid)
    return meta

def build_paper_subgraphs_from_adj_out(
    work_dir: str,
    max_subgraphs: int = 50000,
    max_neighbors_each_side: int = 8,
    sample_prob: float = 1e-3,
    max_summary_len: int = 1200,
):
    adj_out_path = os.path.join(work_dir, "adj_out.tsv")
    summary_path = os.path.join(work_dir, "paper_summary.jsonl")
    out_path = os.path.join(work_dir, "0104_paper_subgraphs_textual.jsonl")

    centers = sample_centers_from_adj_out(
        adj_out_path=adj_out_path,
        max_subgraphs=max_subgraphs,
        sample_prob=sample_prob,
    )
    print(f"[INFO] sampled {len(centers)} centers from adj_out")

    needed_ids: Set[str] = set(centers.keys())
    for s, neighs in centers.items():

        if len(neighs) > max_neighbors_each_side:
            centers[s] = random.sample(neighs, max_neighbors_each_side)
        needed_ids.update(centers[s])

    print(f"[INFO] total needed node ids: {len(needed_ids)}")

    meta = load_summaries_for_nodes(
        summary_path=summary_path,
        needed_ids=needed_ids,
        max_summary_len=max_summary_len,
    )
    print(f"[INFO] loaded meta for {len(meta)} nodes")

    properties = {
        "cites": {
            "summary": "A citation relation in the scientific literature: a paper cites another paper."
        }
    }

    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for center, neighs in centers.items():
            if center not in meta:
                continue
            entities: Dict[str, Dict[str, Any]] = {}

            entities[center] = meta[center]

            for t in neighs:
                if t in meta:
                    entities[t] = meta[t]
            if len(entities) < 2:
                continue

            edges = []
            for t in neighs:
                if t in entities:
                    edges.append({"s": center, "p": "cites", "o": t})
            if not edges:
                continue

            record = {
                "subgraph_id": center,
                "entities": entities,
                "properties": properties,
                "edges": edges,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[INFO] built {count} subgraphs into {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="../KG")
    ap.add_argument("--max_subgraphs", type=int, default=250000)
    ap.add_argument("--max_neighbors_each_side", type=int, default=8)
    ap.add_argument("--sample_prob", type=float, default=0.05)
    ap.add_argument("--max_summary_len", type=int, default=1200)
    args = ap.parse_args()

    build_paper_subgraphs_from_adj_out(
        work_dir=args.work_dir,
        max_subgraphs=args.max_subgraphs,
        max_neighbors_each_side=args.max_neighbors_each_side,
        sample_prob=args.sample_prob,
        max_summary_len=args.max_summary_len,
    )

if __name__ == "__main__":
    main()
