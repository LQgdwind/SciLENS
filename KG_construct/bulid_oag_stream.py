import os, json, gc, hashlib, argparse
from tqdm import tqdm

class FastPaperDeduplicator:
    def __init__(self, work_dir):
        self.doi_index = {}
        self.title_hash_index = {}
        self.id_map_f = open(os.path.join(work_dir, "id_mapping.tsv"), "w", encoding="utf-8")
        self.paper_f = open(os.path.join(work_dir, "paper_info.jsonl"), "a", encoding="utf-8")

    @staticmethod
    def norm_doi(doi):
        if not doi:
            return None
        doi = doi.lower().strip()
        for p in ("<REDACTED_URL>","<REDACTED_URL>","<REDACTED_URL>","<REDACTED_URL>","doi:"):
            if doi.startswith(p): doi = doi[len(p):]
        return doi.strip() or None

    @staticmethod
    def title_hash(title):
        if not title: return None
        import re
        t = re.sub(r"[^\w\s]"," ", title.lower())
        t = re.sub(r"\s+"," ", t).strip()
        if len(t)<10: return None
        return hashlib.md5(t.encode()).hexdigest()

    def add_paper(self, pid, info):
        doi = self.norm_doi(info.get("doi"))
        th = self.title_hash(info.get("title"))
        cid = None
        if doi and doi in self.doi_index:
            cid = self.doi_index[doi]
        elif th and th in self.title_hash_index:
            cid = self.title_hash_index[th]

        if not cid:
            cid = pid
            if doi: self.doi_index[doi] = cid
            if th: self.title_hash_index[th] = cid
            self.paper_f.write(json.dumps(info, ensure_ascii=False)+"\n")

        self.id_map_f.write(f"{pid}\t{cid}\n")
        return cid

    def close(self):
        self.id_map_f.close()
        self.paper_f.close()
        self.doi_index.clear(); self.title_hash_index.clear()

def build_stage1(data_dir, work_dir, file_range):
    os.makedirs(work_dir, exist_ok=True)
    dedup = FastPaperDeduplicator(work_dir)
    stats = {"files":0, "papers":0, "edges":0}

    for i in file_range:
        path = os.path.join(data_dir, f"v5_oag_publication_{i}.json")
        if not os.path.exists(path):
            print(f"⚠ missing {path}")
            continue

        stats["files"] += 1
        edge_path = os.path.join(work_dir, f"edges_part_{i}.tsv")
        ef = open(edge_path, "w", encoding="utf-8")

        with open(path, "r", encoding="utf-8") as f, tqdm(desc=f"file {i}") as bar:
            for line in f:
                bar.update(1)
                try:
                    p = json.loads(line)
                except Exception:
                    continue
                pid = p.get("id")
                if not pid:
                    continue
                info = {
                    "id": pid,
                    "title": p.get("title",""),
                    "year": p.get("year"),
                    "venue": p.get("venue"),
                    "venue_id": p.get("venue_id"),
                    "doi": p.get("doi"),
                    "n_citation": int(p.get("n_citation",0)),
                    "keywords": p.get("keywords", []),
                    "abstract": p.get("abstract",""),
                    "authors": [
                        {"name": a.get("name"), "id":a.get("id"), "org":a.get("org"), "org_id":a.get("org_id")}
                        for a in p.get("authors",[]) if a.get("name")
                    ]
                }
                cid = dedup.add_paper(pid, info)
                refs = p.get("references", [])
                for rid in refs:
                    ef.write(f"{cid}\t{rid}\n")
                    stats["edges"] += 1
                stats["papers"] += 1

        ef.close()
        gc.collect()

    dedup.close()
    with open(os.path.join(work_dir, "stats_stage1.json"),"w",encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    print("Stage1 完成", stats)

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="OAG 百亿级流式建图阶段1")
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--output_dir", default="KG")
    ap.add_argument("--file_range", type=str, default="1-16")
    a = ap.parse_args()
    if "-" in a.file_range:
        s,e = map(int,a.file_range.split("-")); fr = range(s,e+1)
    else:
        fr = [int(x) for x in a.file_range.split(",")]
    build_stage1(a.data_dir, a.output_dir, fr)
