import os
import json
import glob
import asyncio
import random
import httpx
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

API_KEY = "<REDACTED_API_KEY>"
BASE_URL = "<REDACTED_URL>"

INPUT_DIR = "/ramdata/QA3_0"
OUTPUT_DIR = "/ramdata/QA3_1"

CONCURRENCY_LIMIT = 1024

EASY_THRESHOLD = 2

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

http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=CONCURRENCY_LIMIT * 2, max_connections=CONCURRENCY_LIMIT * 3),
    timeout=60.0
)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

async def get_model_answer(model_name, question):
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:

        return ""

async def judge_answer(judge_model, question, ground_truth, model_pred):
    if not model_pred:
        return False

    prompt = f"""
Judge if the Prediction matches the Ground Truth.
Question: {question}
Ground Truth: {ground_truth}
Prediction: {model_pred}
Reply YES or NO.
"""
    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}]
        )
        judgment = response.choices[0].message.content.strip().upper()
        return "YES" in judgment
    except Exception:
        return False

async def process_single_qa(qa_item, semaphore):
    async with semaphore:
        qa_content = qa_item.get('qa', qa_item)
        question = qa_content.get('question')
        ground_truth = qa_content.get('answer')

        if not question or not ground_truth:
            return None

        gen_models = random.sample(MODELS, 4)

        gen_tasks = [get_model_answer(m, question) for m in gen_models]
        predictions = await asyncio.gather(*gen_tasks)

        judge_tasks = []
        for i, pred in enumerate(predictions):
            if pred:

                available_judges = [m for m in MODELS if m != gen_models[i]]
                judge_model = random.choice(available_judges) if available_judges else random.choice(MODELS)

                judge_tasks.append(judge_answer(judge_model, question, ground_truth, pred))
            else:
                judge_tasks.append(asyncio.sleep(0, result=False))

        results = await asyncio.gather(*judge_tasks)
        correct_count = sum(results)

        if correct_count >= EASY_THRESHOLD:
            return None
        else:
            return qa_item

async def process_file(file_path, semaphore):
    file_name = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, file_name)

    qa_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if "qas" in obj and isinstance(obj["qas"], list):
                        qa_list.extend(obj["qas"])
                    elif "question" in obj:
                        qa_list.append(obj)
                except:
                    continue
    except Exception as e:
        print(f"Read Error {file_name}: {e}")
        return 0, 0

    if not qa_list:
        return 0, 0

    tasks = [process_single_qa(item, semaphore) for item in qa_list]

    kept_items = []
    dropped_count = 0

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Proc {file_name}", leave=False):
        result_item = await f
        if result_item:
            kept_items.append(result_item)
        else:
            dropped_count += 1

    if kept_items:
        with open(output_path, 'w', encoding='utf-8') as f:

            buffer = []
            for item in kept_items:
                wrapper = {"qas": [item]}
                buffer.append(json.dumps(wrapper, ensure_ascii=False))
            f.write("\n".join(buffer) + "\n")

    return len(kept_items), dropped_count

async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    json_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl")) + glob.glob(os.path.join(INPUT_DIR, "*.json"))

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"🚀 High-Speed Clean Mode | Concurrency: {CONCURRENCY_LIMIT} | Files: {len(json_files)}")

    total_kept = 0
    total_dropped = 0

    for file_path in json_files:
        k, d = await process_file(file_path, semaphore)
        total_kept += k
        total_dropped += d
        print(f" > {os.path.basename(file_path)}: Keep {k} / Drop {d}")

    print(f"Done. Kept: {total_kept}, Dropped: {total_dropped}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
