import os, json, argparse
from tqdm import tqdm

def build_light_paper_summary(work_dir: str,
                              in_file: str = "paper_info.jsonl",
                              out_file: str = "paper_summary.jsonl"):
    in_path = os.path.join(work_dir, in_file)
    out_path = os.path.join(work_dir, out_file)

    with open(in_path, "r", encoding="utf-8") as fin,\
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="summary"):
            if not line.strip():
                continue
            p = json.loads(line)
            pid = p.get("id")
            if not pid:
                continue
            rec = {
                "id": pid,
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] light summary written to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="../KG")
    ap.add_argument("--in_file", default="paper_info.jsonl")
    ap.add_argument("--out_file", default="paper_summary.jsonl")
    args = ap.parse_args()
    build_light_paper_summary(args.work_dir, args.in_file, args.out_file)

if __name__ == "__main__":
    main()
