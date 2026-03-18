from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import faiss
import numpy as np
import uvicorn
from typing import List, Tuple
import os
import sys
from datetime import datetime
import logging
import asyncio
from contextlib import asynccontextmanager
import psutil
import time

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    import json

EMB_DIM = 4096
IVFPQ_INDEX_PATH = "/ramdata/faiss_ivfpq/merged_ivfpq_from_flat.faiss"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9000

MAX_CONCURRENT = 800
TIMEOUT_SECONDS = 15

logging.basicConfig(
    level=logging.WARNING,
    format=f'[{SERVER_PORT}] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class IVFPQSearchEngine:

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None
        self.ivfpq_index = None
        self.start_time = datetime.now()
        self.total_requests = 0
        self.failed_requests = 0

        logger.info("=" * 70)
        logger.info("Initializing IVFPQ (550-instance cluster, 10k concurrent)")
        logger.info("=" * 70)

        self._load_index()
        self._optimize_index()

        logger.info("✅ IVFPQ Engine Ready")
        logger.info(f"   Vectors: {self.get_total_vectors():,}")
        logger.info(f"   Max concurrent: {MAX_CONCURRENT}")
        logger.info("=" * 70)

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"IVFPQ index not found: {self.index_path}")

        load_start = time.time()
        self.index = faiss.read_index(self.index_path)
        load_time = time.time() - load_start

        logger.info(f"Loaded in {load_time:.2f}s")

        if isinstance(self.index, faiss.IndexIDMap):
            self.ivfpq_index = faiss.downcast_index(self.index.index)
        else:
            self.ivfpq_index = self.index

    def _optimize_index(self):
        if self.ivfpq_index is None:
            return

        if hasattr(self.ivfpq_index, 'nprobe'):
            self.ivfpq_index.nprobe = 32

        cpu_count = psutil.cpu_count(logical=False)
        num_threads = max(1, min(2, cpu_count // 50))
        faiss.omp_set_num_threads(num_threads)

        for _ in range(2):
            warmup_vec = np.random.randn(1, EMB_DIM).astype('float32')
            faiss.normalize_L2(warmup_vec)
            self.index.search(warmup_vec, 5)

    def search(self, vector: List[float], k: int) -> List[Tuple[float, int]]:
        self.total_requests += 1

        if self.index is None:
            self.failed_requests += 1
            return []

        k = min(k, 100)

        try:
            xq = np.array([vector], dtype='float32')

            if xq.shape[1] != EMB_DIM:
                self.failed_requests += 1
                return []

            faiss.normalize_L2(xq)
            D, I = self.index.search(xq, k)

            results = []
            for dist, fid in zip(D[0], I[0]):
                if fid != -1:
                    results.append((float(dist), int(fid)))

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.failed_requests += 1
            return []

    def get_total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def get_uptime(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def get_stats(self) -> dict:
        success_rate = 0
        if self.total_requests > 0:
            success_rate = round((self.total_requests - self.failed_requests) / self.total_requests * 100, 2)

        stats = {
            "status": "healthy",
            "server_port": SERVER_PORT,
            "index_type": "IVFPQ",
            "cluster_size": 550,
            "total_vectors": self.get_total_vectors(),
            "uptime_seconds": round(self.get_uptime(), 2),
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate
        }

        if self.ivfpq_index and hasattr(self.ivfpq_index, 'nprobe'):
            stats["ivfpq_nprobe"] = self.ivfpq_index.nprobe

        process = psutil.Process()
        stats["memory_mb"] = round(process.memory_info().rss / 1024**2, 2)

        if os.path.exists(self.index_path):
            stats["index_size_mb"] = round(os.path.getsize(self.index_path) / 1024**2, 2)

        return stats

search_engine: IVFPQSearchEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine

    try:
        search_engine = IVFPQSearchEngine(IVFPQ_INDEX_PATH)
        logger.info("🚀 Service ready")
    except Exception as e:
        logger.error(f"❌ Init failed: {e}")
        raise

    yield
    logger.info("🛑 Shutdown")

app = FastAPI(
    title=f"FAISS IVFPQ (Port {SERVER_PORT})",
    description="550-instance cluster, 10k concurrent support",
    version="9.0.0",
    lifespan=lifespan
)

active_requests = {"count": 0}

@app.middleware("http")
async def concurrency_limiter(request: Request, call_next):
    path = request.url.path

    if path in ["/", "/health", "/stats", "/docs", "/openapi.json"]:
        return await call_next(request)

    active_requests["count"] += 1

    if active_requests["count"] > MAX_CONCURRENT:
        active_requests["count"] -= 1
        return JSONResponse(
            status_code=503,
            content={
                "error": "Instance busy",
                "current": active_requests["count"],
                "max": MAX_CONCURRENT,
                "hint": "550 instances available, auto-retry"
            }
        )

    try:
        response = await asyncio.wait_for(call_next(request), timeout=TIMEOUT_SECONDS)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Timeout"})
    finally:
        active_requests["count"] = max(0, active_requests["count"] - 1)

@app.post("/search")
async def search(request: Request):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    try:
        start_time = time.time()
        body_bytes = await request.body()

        if USE_ORJSON:
            data = orjson.loads(body_bytes)
        else:
            data = json.loads(body_bytes)

        vector = data.get("vector")
        k = data.get("k", 5)

        if vector is None:
            raise HTTPException(status_code=400, detail="Missing vector")

        if not isinstance(vector, list) or len(vector) != EMB_DIM:
            raise HTTPException(status_code=400, detail=f"Invalid vector dimension")

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, search_engine.search, vector, k)

        query_time_ms = (time.time() - start_time) * 1000

        response_data = {
            "results": results,
            "query_time_ms": round(query_time_ms, 2),
            "timestamp": datetime.now().isoformat(),
            "server_port": SERVER_PORT
        }

        if USE_ORJSON:
            return Response(content=orjson.dumps(response_data), media_type="application/json")
        else:
            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "FAISS IVFPQ",
        "version": "9.0.0",
        "cluster_size": 550,
        "max_concurrent_per_instance": MAX_CONCURRENT,
        "cluster_capacity": 550 * MAX_CONCURRENT,
        "server_port": SERVER_PORT
    }

@app.get("/health")
async def health():
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    stats = search_engine.get_stats()
    return {
        "status": "healthy",
        "cluster_size": 550,
        "total_vectors": stats["total_vectors"],
        "uptime": stats["uptime_seconds"],
        "port": SERVER_PORT,
        "memory_mb": stats.get("memory_mb", 0)
    }

@app.get("/stats")
async def stats():
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    stats_data = search_engine.get_stats()
    stats_data["current_concurrent"] = active_requests["count"]
    stats_data["max_concurrent"] = MAX_CONCURRENT
    return stats_data

if __name__ == "__main__":
    if not os.path.exists(IVFPQ_INDEX_PATH):
        logger.error(f"❌ Index not found: {IVFPQ_INDEX_PATH}")
        sys.exit(1)

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="critical",
        access_log=False,
        workers=1,
        limit_concurrency=MAX_CONCURRENT,
        timeout_keep_alive=120,
        backlog=10000,
        timeout_graceful_shutdown=30,
    )
