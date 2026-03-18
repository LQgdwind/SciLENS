import os
import sys
import glob
import time
import orjson
import pymongo
import subprocess
import argparse
import multiprocessing
from multiprocessing import Queue
from collections import defaultdict
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

CONFIG = {

    "MONGO_URI": "mongodb://localhost:27017/",
    "DB_NAME": "academic_db",
    "COLLECTION": "papers",

    "FILES": {

        "PAPER_INFO": "<REDACTED_PATH>",

        "MAPPING_DIR": "<REDACTED_PATH>",
        "MAPPING_PATTERN": "id_mapping_*.jsonl",

        "CITATION_TRIPLES": "<REDACTED_PATH>",
    },

    "WORKERS_IMPORT": 48,
    "WORKERS_UPDATE": 32,
    "WORKERS_CITATION": 80,
    "BATCH_SIZE": 20000,
    "TOTAL_PAPERS_EST": 12_000_000,
}

WRITE_CONCERN = pymongo.write_concern.WriteConcern(w=0, j=False)

def get_db_collection(client):
    return client[CONFIG["DB_NAME"]].get_collection(
        CONFIG["COLLECTION"], write_concern=WRITE_CONCERN
    )

def worker_import_papers(chunk_start, chunk_end, worker_id, progress_queue):
    client = MongoClient(CONFIG["MONGO_URI"], maxPoolSize=4)
    collection = get_db_collection(client)
    docs_buffer = []

    try:
        with open(CONFIG["FILES"]["PAPER_INFO"], 'rb') as f:
            f.seek(chunk_start)
            if chunk_start != 0:
                f.readline()

            while f.tell() < chunk_end:
                line = f.readline()
                if not line: break
                try:
                    item = orjson.loads(line)

                    if 'id' in item: item['_id'] = item.pop('id')

                    item['n_citation'] = int(item.get('n_citation', 0))
                    item['year'] = int(item.get('year', 0))

                    docs_buffer.append(item)
                    if len(docs_buffer) >= CONFIG["BATCH_SIZE"]:
                        collection.insert_many(docs_buffer, ordered=False)
                        progress_queue.put(len(docs_buffer))
                        docs_buffer = []
                except Exception: continue

            if docs_buffer:
                collection.insert_many(docs_buffer, ordered=False)
                progress_queue.put(len(docs_buffer))
    except Exception as e:
        print(f"[Import Worker {worker_id}] Error: {e}")
    finally:
        client.close()
        progress_queue.put(None)

def run_step_1_import_papers():
    print(f"\n🚀 [Step 1] 导入论文基础数据...")
    fpath = CONFIG["FILES"]["PAPER_INFO"]
    if not os.path.exists(fpath):
        print(f"❌ 文件未找到: {fpath}，请检查路径配置！"); return

    with MongoClient(CONFIG["MONGO_URI"]) as c:
        print("🧹 清理旧集合 (Drop Collection)...")
        c[CONFIG["DB_NAME"]].drop_collection(CONFIG["COLLECTION"])

    file_size = os.path.getsize(fpath)
    num_workers = CONFIG["WORKERS_IMPORT"]
    chunk_size = file_size // num_workers
    queue = Queue()
    procs = []

    for i in range(num_workers):
        start = i * chunk_size
        end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
        p = multiprocessing.Process(target=worker_import_papers, args=(start, end, i, queue))
        p.start(); procs.append(p)

    finished = 0
    with tqdm(total=CONFIG["TOTAL_PAPERS_EST"], unit="docs", unit_scale=True, desc="Importing") as pbar:
        while finished < num_workers:
            res = queue.get()
            if res is None: finished += 1
            else: pbar.update(res)

    for p in procs: p.join()
    print("✅ Step 1 完成。")

def worker_update_ids(file_list, worker_id, progress_queue):
    client = MongoClient(CONFIG["MONGO_URI"], maxPoolSize=4)
    collection = get_db_collection(client)
    ops_buffer = []

    try:
        for file_path in file_list:
            with open(file_path, 'rb') as f:
                for line in f:
                    try:
                        item = orjson.loads(line)

                        op = UpdateOne(
                            {'_id': item['ref_id']},
                            {'$set': {'faiss_id': item['_id']}}
                        )
                        ops_buffer.append(op)

                        if len(ops_buffer) >= CONFIG["BATCH_SIZE"]:
                            collection.bulk_write(ops_buffer, ordered=False)
                            progress_queue.put(len(ops_buffer))
                            ops_buffer = []
                    except Exception: continue
            if ops_buffer:
                collection.bulk_write(ops_buffer, ordered=False)
                progress_queue.put(len(ops_buffer))
                ops_buffer = []
    except Exception as e:
        print(f"[ID Worker {worker_id}] Error: {e}")
    finally:
        client.close(); progress_queue.put(None)

def run_step_2_update_ids():
    print(f"\n🚀 [Step 2] 更新 Faiss ID (Mapping)...")
    search_path = os.path.join(CONFIG["FILES"]["MAPPING_DIR"], CONFIG["FILES"]["MAPPING_PATTERN"])
    all_files = glob.glob(search_path)
    if not all_files: print(f"❌ 未找到 Mapping 文件: {search_path}，跳过。"); return

    num_workers = min(CONFIG["WORKERS_UPDATE"], len(all_files))
    file_chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(all_files): file_chunks[i % num_workers].append(f)

    queue = Queue()
    procs = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker_update_ids, args=(file_chunks[i], i, queue))
        p.start(); procs.append(p)

    finished = 0
    with tqdm(unit="docs", unit_scale=True, desc="Updating IDs") as pbar:
        while finished < num_workers:
            res = queue.get()
            if res is None: finished += 1
            else: pbar.update(res)

    for p in procs: p.join()
    print("✅ Step 2 完成。")

def worker_import_citations(subset_data, worker_id):
    client = MongoClient(CONFIG["MONGO_URI"], maxPoolSize=4)
    collection = get_db_collection(client)
    ops_buffer = []
    try:
        for src_id, ref_list in subset_data:
            op = UpdateOne({"_id": src_id}, {"$set": {"references": ref_list}})
            ops_buffer.append(op)
            if len(ops_buffer) >= CONFIG["BATCH_SIZE"]:
                collection.bulk_write(ops_buffer, ordered=False)
                ops_buffer = []
        if ops_buffer: collection.bulk_write(ops_buffer, ordered=False)
    except Exception as e:
        print(f"[Cit Worker {worker_id}] Error: {e}")
    finally:
        client.close()

def run_step_3_citations():
    print(f"\n🚀 [Step 3] 导入引用关系...")
    fpath = CONFIG["FILES"]["CITATION_TRIPLES"]
    if not os.path.exists(fpath): print(f"❌ 引用文件不存在: {fpath}，跳过。"); return

    print("⏳ [Memory] 正在全量加载引用关系到内存 (利用2TB RAM优势)...")
    citation_map = defaultdict(list)
    with open(fpath, 'r') as f:
        for line in tqdm(f, desc="Reading Disk", unit=" lines", mininterval=2.0):
            parts = line.strip().split()
            if len(parts) >= 3:
                citation_map[parts[0]].append(parts[2])

    all_items = list(citation_map.items())
    del citation_map

    print(f"🔥 正在启动 {CONFIG['WORKERS_CITATION']} 个进程写入 MongoDB...")
    chunk_size = len(all_items) // CONFIG['WORKERS_CITATION'] + 1
    chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]

    procs = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=worker_import_citations, args=(chunk, i))
        p.start(); procs.append(p)

    for p in procs: p.join()
    print("✅ Step 3 完成。")

def run_step_4_indexes():
    print(f"\n🚀 [Step 4] 构建最终索引...")

    client = MongoClient(CONFIG["MONGO_URI"], socketTimeoutMS=None)
    col = client[CONFIG["DB_NAME"]][CONFIG["COLLECTION"]]

    print("🧹 为了保险起见，清理旧索引...")
    col.drop_indexes()

    print("⏳ [1/4] Creating Index: faiss_id (Unique)...")
    col.create_index([("faiss_id", pymongo.ASCENDING)], unique=True, name="idx_faiss_id")

    print("⏳ [2/4] Creating Index: title (Ascending)...")
    col.create_index([("title", pymongo.ASCENDING)], name="idx_title_exact")

    print("⏳ [3/4] Creating Index: Text (Keywords + Title)...")
    col.create_index(
        [("keywords", "text"), ("title", "text")],
        weights={"keywords": 10, "title": 5},
        name="idx_full_text"
    )

    print("⏳ [4/4] Creating Index: Sort (Citation + Year)...")
    col.create_index([("n_citation", -1), ("year", -1)], name="idx_sort_citation")

    print("\n✅ 所有索引构建完毕。当前索引列表:")
    for name, info in col.index_information().items():
        print(f"   - {name}: {info['key']}")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学术数据库终极构建脚本 (v3.0)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["import", "mapping", "citation", "index", "all"],
                        help="选择要执行的步骤: import=导论文, mapping=导ID, citation=导引用, index=建索引")
    args = parser.parse_args()

    start_global = time.time()

    if args.step in ["import", "all"]:
        run_step_1_import_papers()

    if args.step in ["mapping", "all"]:
        run_step_2_update_ids()

    if args.step in ["citation", "all"]:
        run_step_3_citations()

    if args.step in ["index", "all"]:
        run_step_4_indexes()

    print(f"\n🎉🎉🎉 所有任务处理完毕！总耗时: {(time.time() - start_global)/60:.2f} 分钟")
