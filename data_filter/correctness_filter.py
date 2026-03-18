import os

os.environ["MALLOC_ARENA_MAX"] = "2"

import sys
import subprocess
import time
import json
import re
import asyncio
import random
import glob
import threading
import shutil
import psutil
import traceback

def install_and_import(package, import_name=None):
    if import_name is None: import_name = package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

required_packages = [
    ("openai", "openai"), ("httpx", "httpx"), ("tqdm", "tqdm"),
    ("duckdb", "duckdb"), ("huggingface_hub", "huggingface_hub"),
    ("psutil", "psutil")
]
for pkg, name in required_packages: install_and_import(pkg, name)

import duckdb
import httpx
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from huggingface_hub import snapshot_download

RAM_DISK = "/ramdata"
DB_PATH = os.path.join(RAM_DISK, "oag_search.duckdb")
DEBUG_LOG_FILE = "verification_debug.log"

ARXIV_DIR = os.path.join(RAM_DISK, "raw_arxiv")
PUBMED_DIR = os.path.join(RAM_DISK, "raw_pubmed")
INPUT_DIR = os.path.join(RAM_DISK, "QA4_0")
OUTPUT_DIR = os.path.join(RAM_DISK, "QA4_1")

DUCKDB_MEMORY_LIMIT = "500GB"
VOTE_TOTAL = 4
VOTE_PASS = 4
CONCURRENCY_LIMIT = 64

API_KEY = "<REDACTED_API_KEY>"
BASE_URL = "<REDACTED_URL>"

MODELS = [
  "claude-3-5-sonnet-20241022",
  "claude-3-7-sonnet-20250219",
  "claude-opus-4-1-20250805",
  "claude-opus-4-20250514",
  "claude-sonnet-4-20250514",
  "claude-sonnet-4-5-20250929",
  "gemini-2.0-flash",
  "gpt-4-turbo-2024-04-09",
  "gpt-4.1-2025-04-14",
  "gpt-4.1-mini-2025-04-14",
  "gpt-4o-2024-08-06",
  "gpt-4o-2024-11-20",
  "gpt-4o-mini-2024-07-18",
  "gpt-5-2025-08-07",
  "gpt-5-chat-2025-08-07",
  "gpt-5-mini-2025-08-07",
  "gpt-5.1",
  "gpt-5.1-chat",
  "o3-2025-04-16",
  "o3-mini-2025-01-31",
  "gpt-5.2",
  "gpt-5.2-chat"
]

db_conn = None
db_access_sem = asyncio.Semaphore(20)

STATS = {
    "total_processed": 0,
    "kept": 0,
    "dropped_logic": 0,
    "dropped_api_error": 0,
    "api_calls_success": 0,
    "api_calls_failed": 0
}
stats_lock = threading.Lock()

def log_debug(msg):

    print(f"📝 {msg}")
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def prepare_data():
    print("⬇️  Verifying Data on RAM Disk...")
    if not os.path.exists(ARXIV_DIR):
        try:
            snapshot_download(repo_id="librarian-bots/arxiv-metadata-snapshot", repo_type="dataset",
                              local_dir=ARXIV_DIR, allow_patterns="*.parquet", local_dir_use_symlinks=False)
        except Exception: pass
    if not os.path.exists(PUBMED_DIR):
        try:
            snapshot_download(repo_id="ccdv/pubmed-summarization", repo_type="dataset",
                              local_dir=PUBMED_DIR, allow_patterns="*.parquet", local_dir_use_symlinks=False)
        except Exception: pass

def init_db():
    global db_conn
    prepare_data()

    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            os.remove(DB_PATH + ".wal")
        except: pass

    print(f"🚀 Initializing DuckDB at {DB_PATH}...")
    db_conn = duckdb.connect(DB_PATH, config={'allow_unsigned_extensions': 'true'})
    db_conn.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    db_conn.execute(f"PRAGMA threads={os.cpu_count()};")

    try:
        arxiv_files = os.path.join(ARXIV_DIR, "**", "*.parquet")
        db_conn.execute(f"CREATE TABLE arxiv AS SELECT * FROM read_parquet('{arxiv_files}', union_by_name=true);")
        db_conn.execute("PRAGMA create_fts_index('arxiv', 'title', 'abstract', stemmer='porter');")
        print("   ✅ ArXiv Index Built.")
    except Exception as e: print(f"   ⚠️ ArXiv Error: {e}")

    try:
        pubmed_files = os.path.join(PUBMED_DIR, "**", "*.parquet")
        db_conn.execute(f"CREATE TABLE pubmed AS SELECT * FROM read_parquet('{pubmed_files}', union_by_name=true);")

        cols = [r[0] for r in db_conn.execute("DESCRIBE pubmed").fetchall()]
        print(f"   ℹ️  PubMed Columns: {cols}")

        index_col = None
        if 'article' in cols: index_col = 'article'
        elif 'abstract' in cols: index_col = 'abstract'

        if index_col:
            db_conn.execute(f"PRAGMA create_fts_index('pubmed', '{index_col}', stemmer='porter');")
            print(f"   ✅ PubMed Index Built on '{index_col}'.")
        else:
            print("   ❌ PubMed Index Failed: No suitable column found.")

    except Exception as e: print(f"   ⚠️ PubMed Error: {e}")

def search_db_sync(table_name, query):
    if not db_conn: return "DB Not Init"
    try:
        cursor = db_conn.cursor()
        try: cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        except: return f"Table {table_name} not loaded."

        safe_query = query.replace("'", " ").replace('"', " ").strip()
        if not safe_query: return "Empty Query"

        text_col = 'abstract'
        if table_name == 'pubmed':
            try:
                cols = [r[0] for r in cursor.execute(f"DESCRIBE {table_name}").fetchall()]
                text_col = 'article' if 'article' in cols else 'abstract'
            except: pass

        sql = f"""
            SELECT score, {text_col}
            FROM (
                SELECT *, fts_main_{table_name}.match_bm25({text_col}, '{safe_query}') AS score
                FROM {table_name}
            )
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT 2
        """
        results = cursor.execute(sql).fetchall()
        cursor.close()

        if not results: return f"No results in {table_name}."

        formatted = []
        for r in results:
            content = str(r[1])[:800].replace('\n', ' ')
            formatted.append(f"Content: {content}...\n")
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Search Error: {str(e)}"

http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=200), timeout=60.0)
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

async def unified_search_tool(query):
    async with db_access_sem:
        task_arxiv = asyncio.to_thread(search_db_sync, 'arxiv', query)
        task_pubmed = asyncio.to_thread(search_db_sync, 'pubmed', query)
        res_arxiv, res_pubmed = await asyncio.gather(task_arxiv, task_pubmed)
    return f"<<<< ARXIV >>>>\n{res_arxiv}\n\n<<<< PUBMED >>>>\n{res_pubmed}"

def clean_json_content(content):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match: return match.group(1)
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match: return match.group(1)
    except: pass
    return content

async def run_agent_verify(model_name, question, answer):
    system_prompt = """You are a Universal Academic Auditor.
Goal: Verify if the QA pair is factually supported by academic literature and the answer is unique.
Tools: Use [SEARCH: keywords] to query ArXiv/PubMed.
Output: [FINISH] followed by JSON: { "is_correct": bool, "is_unique": bool, "reason": "str" }"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}]

    for _ in range(4):
        try:
            resp = await client.chat.completions.create(model=model_name, messages=messages)
            content = resp.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": content})

            if "[SEARCH:" in content:
                match = re.search(r"\[SEARCH:\s*(.*?)\]", content)
                if match:
                    q = match.group(1)
                    res = await unified_search_tool(q)
                    messages.append({"role": "user", "content": f"[TOOL]:\n{res}"})
            elif "[FINISH]" in content:
                json_str = clean_json_content(content)
                res_json = json.loads(json_str)
                res_json['model'] = model_name

                with stats_lock: STATS["api_calls_success"] += 1
                return res_json
        except Exception as e:

            with stats_lock: STATS["api_calls_failed"] += 1
            return {"model": model_name, "error": f"{type(e).__name__}: {str(e)}", "is_api_error": True}

    return {"model": model_name, "error": "Max turns reached", "is_api_error": False}

async def process_single_qa(qa_item, semaphore):
    async with semaphore:
        q = qa_item.get('qa', {}).get('question') or qa_item.get('question')
        a = qa_item.get('qa', {}).get('answer') or qa_item.get('answer')
        if not q or not a: return None

        with stats_lock: STATS["total_processed"] += 1

        selected_models = [random.choice(MODELS) for _ in range(VOTE_TOTAL)]
        tasks = [run_agent_verify(m, q, a) for m in selected_models]
        results = await asyncio.gather(*tasks)

        votes_yes = 0
        details = []
        has_api_error = False

        for res in results:
            if res.get("is_api_error"):
                has_api_error = True

                log_debug(f"🚨 API ERROR [{res['model']}]: {res['error']}")
                details.append({"m": res.get('model'), "err": res['error']})
                continue

            if "error" in res:
                details.append({"m": res.get('model'), "err": res['error']})
                continue

            is_pass = res.get('is_correct') is True and res.get('is_unique') is True
            if is_pass: votes_yes += 1
            details.append({"m": res.get('model'), "pass": is_pass, "r": str(res.get('reason'))[:50]})

        if votes_yes >= VOTE_PASS:
            with stats_lock: STATS["kept"] += 1
            qa_item['verification'] = {"score": f"{votes_yes}/{VOTE_TOTAL}", "details": details}
            return qa_item
        else:
            with stats_lock:
                if has_api_error:
                    STATS["dropped_api_error"] += 1
                else:
                    STATS["dropped_logic"] += 1

            if random.random() < 0.05:
                if has_api_error:
                    log_debug(f"❌ DROPPED (API ERROR): {q[:30]}...")
                else:
                    log_debug(f"🗑️ DROPPED (LOGIC): {q[:30]}... Votes: {votes_yes}/{VOTE_TOTAL}")
            return None

async def process_file(file_path, semaphore):
    fname = os.path.basename(file_path)
    out_path = os.path.join(OUTPUT_DIR, fname)
    qa_list = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "qas" in obj: qa_list.extend(obj["qas"])
                        else: qa_list.append(obj)
                    except: pass
    except: return 0

    if not qa_list: return 0

    tasks = [process_single_qa(item, semaphore) for item in qa_list]
    kept = []

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Proc {fname}", leave=False):
        res = await f
        if res: kept.append(res)

    with open(out_path, 'w', encoding='utf-8') as f:
        if kept:
            for item in kept:
                f.write(json.dumps({"qas": [item]}, ensure_ascii=False) + "\n")

    return len(kept)

async def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    with open(DEBUG_LOG_FILE, "w") as f: f.write("Starting Diagnostic Run...\n")

    init_db()

    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl")) + glob.glob(os.path.join(INPUT_DIR, "*.json"))
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"🔥 Started. Logs -> {DEBUG_LOG_FILE}")
    total = 0
    for fp in files:
        total += await process_file(fp, sem)

    print("\n" + "="*50)
    print("📊 FINAL STATISTICS")
    print(f"   Total Processed: {STATS['total_processed']}")
    print(f"   ✅ Kept:          {STATS['kept']}")
    print(f"   🗑️ Dropped (Logic):{STATS['dropped_logic']} (Models said NO)")
    print(f"   🚨 Dropped (API):  {STATS['dropped_api_error']} (API Failed)")
    print("-" * 20)
    print(f"   API Calls Success: {STATS['api_calls_success']}")
    print(f"   API Calls Failed:  {STATS['api_calls_failed']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
