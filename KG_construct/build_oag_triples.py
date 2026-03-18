import os, json, argparse
from collections import defaultdict
from tqdm import tqdm

def stage2_generate_triples(work_dir, min_degree=2, min_citations=10):

    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    edge_parts = [f for f in os.listdir(work_dir) if f.startswith("edges_part_")]
    print(f"发现边文件 {len(edge_parts)} 个")

    for fn in tqdm(edge_parts, desc="统计度"):
        with open(os.path.join(work_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                s,t = line.strip().split("\t")
                outdeg[s]+=1
                indeg[t]+=1

    valid_nodes = set()
    for pid in set(list(indeg.keys())+list(outdeg.keys())):
        if outdeg[pid]+indeg[pid] >= min_degree and indeg[pid] >= min_citations:
            valid_nodes.add(pid)
    print(f"有效节点 {len(valid_nodes):,}")

    triple_path = os.path.join(work_dir,"citation_triples.txt")
    out = open(triple_path,"w",encoding="utf-8")
    kept_edges = 0
    for fn in tqdm(edge_parts, desc="生成三元组"):
        with open(os.path.join(work_dir, fn), "r", encoding="utf-8") as f:
            for line in f:
                s,t = line.strip().split("\t")
                if s in valid_nodes and t in valid_nodes:
                    out.write(f"{s}\tcites\t{t}\n")
                    kept_edges +=1
    out.close()
    stats = {
        "valid_nodes": len(valid_nodes),
        "triples": kept_edges
    }
    with open(os.path.join(work_dir,"stats_stage2.json"),"w",encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    print("Stage2 完成", stats)

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="OAG 百亿级三元组过滤阶段")
    ap.add_argument("--work_dir", default="KG")
    ap.add_argument("--min_degree", type=int, default=40)
    ap.add_argument("--min_citations", type=int, default=10)
    a = ap.parse_args()
    stage2_generate_triples(a.work_dir, a.min_degree, a.min_citations)
