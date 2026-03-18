import os, argparse, json, hashlib
from collections import defaultdict
from tqdm import tqdm
import glob

def node_bucket(pid: str, num_buckets: int) -> int:

    h = hashlib.md5(pid.encode()).hexdigest()
    return int(h[:8], 16) % num_buckets

def build_adj_out_on_disk(work_dir: str,
                          edge_pattern: str = "citation_triples.txt",
                          num_buckets: int = 256,
                          rel_filter: str = "cites"):
    edge_files = []
    if "*" in edge_pattern:
        edge_files = sorted(glob.glob(os.path.join(work_dir, edge_pattern)))
    else:
        edge_files = [os.path.join(work_dir, edge_pattern)]
    edge_files = [p for p in edge_files if os.path.exists(p)]
    if not edge_files:
        raise FileNotFoundError(f"No edge files match {edge_pattern} under {work_dir}")
    print(f"[INFO] edge files: {edge_files}")

    tmp_dir = os.path.join(work_dir, "adj_out_buckets")
    os.makedirs(tmp_dir, exist_ok=True)

    print("[INFO] Phase1: split edges into buckets ...")
    bucket_files = [open(os.path.join(tmp_dir, f"bucket_{i}.tsv"), "w", encoding="utf-8")
                    for i in range(num_buckets)]

    for ef in edge_files:
        with open(ef, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"split {os.path.basename(ef)}"):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    s, rel, t = parts
                    if rel_filter and rel != rel_filter:
                        continue
                elif len(parts) == 2:
                    s, t = parts
                else:
                    continue
                b = node_bucket(s, num_buckets)
                bucket_files[b].write(f"{s}\t{t}\n")

    for bf in bucket_files:
        bf.close()

    print("[INFO] Phase2: aggregate each bucket ...")
    adj_out_parts = []
    for i in range(num_buckets):
        bucket_path = os.path.join(tmp_dir, f"bucket_{i}.tsv")
        if not os.path.exists(bucket_path):
            continue

        adj = defaultdict(list)
        with open(bucket_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"agg bucket_{i}", leave=False):
                s, t = line.strip().split("\t")
                adj[s].append(t)

        part_path = os.path.join(tmp_dir, f"adj_out_part_{i}.tsv")
        with open(part_path, "w", encoding="utf-8") as pf:
            for s, ts in adj.items():
                pf.write(f"{s}\t{','.join(ts)}\n")
        adj_out_parts.append(part_path)

        os.remove(bucket_path)

    out_path = os.path.join(work_dir, "adj_out.tsv")
    with open(out_path, "w", encoding="utf-8") as fout:
        for part in adj_out_parts:
            with open(part, "r", encoding="utf-8") as pf:
                for line in pf:
                    fout.write(line)
    print(f"[INFO] adj_out written to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="../KG")
    ap.add_argument("--edge_pattern", default="citation_triples.txt")
    ap.add_argument("--num_buckets", type=int, default=256)
    args = ap.parse_args()
    build_adj_out_on_disk(args.work_dir, args.edge_pattern, args.num_buckets)

if __name__ == "__main__":
    main()
