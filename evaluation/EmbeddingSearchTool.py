import json
import asyncio
import httpx
import logging
import random
from pymongo import MongoClient
from openai import OpenAI
from typing import List, Tuple, AsyncIterator
from openai_harmony import Message, TextContent, ToolNamespaceConfig, Author, Role
from tool import Tool

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    print("[WARNING] orjson not installed. Install for 10x faster serialization: pip install orjson")

EMB_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
EMB_DIM = 4096
API_PORTS = [8000, 8001, 8002, 8003]

FAISS_SERVICE_PORTS = list(range(9000, 9550))

logger = logging.getLogger(__name__)

class FAISSClient:

    def __init__(self, base_ports: List[int] = FAISS_SERVICE_PORTS):
        self.ports = base_ports
        self.base_url = "<REDACTED_URL>"

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=15.0,
                write=10.0,
                pool=10.0
            ),
            limits=httpx.Limits(
                max_connections=20000,
                max_keepalive_connections=5000,
                keepalive_expiry=120.0
            ),
            http2=True,
            transport=httpx.AsyncHTTPTransport(retries=0)
        )

        print(f"[FAISSClient] Ready: {len(self.ports)} instances (ports {self.ports[0]}-{self.ports[-1]})")
        if USE_ORJSON:
            print(f"[FAISSClient] Using orjson")

    def _get_next_port(self) -> int:
        return random.choice(self.ports)

    def _serialize_payload(self, vector: List[float], k: int) -> bytes:
        payload = {"vector": vector, "k": k}
        if USE_ORJSON:
            return orjson.dumps(payload)
        else:
            return json.dumps(payload).encode('utf-8')

    def _deserialize_response(self, content: bytes) -> dict:
        if USE_ORJSON:
            return orjson.loads(content)
        else:
            return json.loads(content.decode('utf-8'))

    async def search(self, vector: List[float], k: int = 5, max_retries: int = 10) -> Tuple[List[Tuple[float, int]], str]:

        tried_ports = []
        error_messages = []
        payload_bytes = self._serialize_payload(vector, k)

        for attempt in range(max_retries):
            port = self._get_next_port()
            tried_ports.append(port)
            search_url = f"{self.base_url}:{port}/search"

            try:
                response = await self.client.post(
                    search_url,
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = self._deserialize_response(response.content)

                return data["results"], ""

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:

                    await asyncio.sleep(0.001)
                    continue
                error_messages.append(f"{port}:HTTP{e.response.status_code}")

            except httpx.TimeoutException:
                error_messages.append(f"{port}:Timeout")
                await asyncio.sleep(0.005)

            except httpx.ConnectError:

                await asyncio.sleep(0.001)
                continue

            except Exception as e:
                error_messages.append(f"{port}:{str(e)[:30]}")
                await asyncio.sleep(0.001)

        unique_ports = list(set(tried_ports))
        error_summary = f"Search failed after {max_retries} retries across {len(unique_ports)} instances. Sample errors: {'; '.join(error_messages[:3])}"
        return [], error_summary

    async def health_check(self) -> dict:
        sample_size = min(40, len(self.ports))
        sample_ports = random.sample(self.ports, sample_size)

        results = []
        for port in sample_ports:
            try:
                response = await self.client.get(
                    f"{self.base_url}:{port}/health",
                    timeout=5.0
                )
                response.raise_for_status()
                results.append({
                    "port": port,
                    "status": "healthy",
                    "data": response.json()
                })
            except Exception as e:
                results.append({
                    "port": port,
                    "status": "unhealthy",
                    "error": str(e)[:50]
                })

        healthy_count = sum(1 for r in results if r["status"] == "healthy")
        estimated_healthy = int(healthy_count * len(self.ports) / sample_size)

        return {
            "total_instances": len(self.ports),
            "sampled_instances": sample_size,
            "sampled_healthy": healthy_count,
            "estimated_healthy": estimated_healthy,
            "health_rate": round(100 * healthy_count / sample_size, 2),
            "estimated_qps": estimated_healthy * 30,
            "cluster_capacity": estimated_healthy * 800,
            "sample_details": results[:10]
        }

    async def close(self):
        await self.client.aclose()

class EmbeddingSearchTool(Tool):
    def __init__(self, name: str = "EmbeddingSearch"):
        assert name == "EmbeddingSearch"
        self.faiss_client = None
        self.reset_state()

    @classmethod
    def get_tool_name(cls) -> str:
        return "EmbeddingSearch"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
Use this tool to execute semantic search using vector embeddings. The query will be searched within a corpus of 12 million papers.

Input format: {"query": "your search text"}
Output: Top-k papers with Title, Abstract, Year, and ID.

This tool uses IVFPQ index optimized for high concurrency:
- 800 load-balanced instances
- Support for 10,000 concurrent requests
- ~30-50ms query latency
- 90-95% recall rate
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def reset_state(self):
        print("[EmbeddingSearchTool] Initializing...")

        if self.faiss_client is not None:
            try:
                asyncio.create_task(self.faiss_client.close())
            except:
                pass

        self.faiss_client = FAISSClient(FAISS_SERVICE_PORTS)

        try:
            self.mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client['academic_db']
            self.collection = self.db['papers']

            self.mongo_client.server_info()
            print("[EmbeddingSearchTool] ✅ MongoDB connected")

        except Exception as e:
            print(f"[EmbeddingSearchTool] ⚠️  MongoDB connection failed: {e}")
            self.collection = None

        self.api_ports = API_PORTS
        self.current_port_index = 0
        self.clients = [
            OpenAI(api_key="EMPTY", base_url=f"<REDACTED_URL>")
            for port in self.api_ports
        ]
        print(f"[EmbeddingSearchTool] ✅ Initialized {len(self.clients)} embedding API clients")

    def _get_next_client(self) -> OpenAI:
        client = self.clients[self.current_port_index]
        self.current_port_index = (self.current_port_index + 1) % len(self.clients)
        return client

    def _get_qwen_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        text = text.replace("\n", " ")

        for attempt in range(max_retries):
            client = self._get_next_client()
            used_port_idx = (self.current_port_index - 1) % len(self.api_ports)

            try:
                resp = client.embeddings.create(
                    model=EMB_MODEL_NAME,
                    input=text
                )
                return resp.data[0].embedding

            except Exception as e:
                print(f"[EmbeddingSearchTool] Embedding API error on port {self.api_ports[used_port_idx]} (attempt {attempt+1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
                else:
                    raise e

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        raw_text = message.content[0].text

        try:
            outer = json.loads(raw_text)

            if isinstance(outer, dict) and "tool_args" in outer:
                args = outer["tool_args"] or {}
            else:
                args = outer

            query_text = args.get("query")
            if query_text is None or query_text == "":
                yield self._make_response("❌ Error: Missing 'query' parameter.", channel)
                return
        except Exception as e:
            yield self._make_response(f"❌ Error: Invalid JSON - {str(e)}", channel)
            return

        if isinstance(query_text, list):

            query_text = " ".join(str(x) for x in query_text)
        elif not isinstance(query_text, str):
            query_text = str(query_text)

        try:
            query_vec = self._get_qwen_embedding(query_text)
        except Exception as e:
            yield self._make_response(f"❌ Failed to generate embedding: {str(e)}", channel)
            return

        try:
            search_results, status_message = await self.faiss_client.search(query_vec, k=5)

            if status_message:
                friendly_message = f"""🔍 Search Service Status: {status_message}

This is a temporary condition. The system has 800 IVFPQ instances for high availability.

You can:
1. Acknowledge this status to the user
2. Suggest trying the query again (it will automatically route to a different instance)
3. Use other available tools (KeywordSearch, PaperInfo, etc.) as alternatives

The search service will be available shortly."""

                yield self._make_response(friendly_message, channel)
                return

            if not search_results:
                yield self._make_response("ℹ️ No results found for this query.", channel)
                return

        except Exception as e:
            error_message = f"⚠️ Search Service Error: {str(e)}"
            yield self._make_response(error_message, channel)
            return

        doc_ids = [fid for dist, fid in search_results]

        response_parts = []
        if self.collection is not None:
            try:
                cursor = self.collection.find(
                    {"faiss_id": {"$in": doc_ids}},
                    {"_id": 1, "title": 1, "year": 1, "abstract": 1, "venue": 1, "faiss_id": 1}
                )
                doc_map = {doc['faiss_id']: doc for doc in cursor}
            except Exception as e:
                print(f"[EmbeddingSearchTool] MongoDB query error: {e}")
                doc_map = {}
        else:
            doc_map = {}

        for rank, (dist, fid) in enumerate(search_results):
            doc = doc_map.get(fid)
            if doc:
                real_id = doc.get("_id")
                clean_doc = {
                    "id": str(real_id),
                    "title": doc.get("title", "No Title"),
                    "year": doc.get("year", "N/A"),
                    "venue": doc.get("venue", "N/A"),
                    "abstract": doc.get("abstract", "")[:500] + "..." if doc.get("abstract") else "",
                }
                doc_str = json.dumps(clean_doc, ensure_ascii=False)
                response_parts.append(f"📄 Rank {rank+1} (Similarity: {1-dist:.4f}):\n{doc_str}")
            else:
                response_parts.append(f"📄 Rank {rank+1} (ID {fid}): Metadata not found.")

        if not response_parts:
            response_parts.append("No relevant papers found.")

        yield self._make_response("\n\n".join(response_parts), channel)

    def _make_response(self, text, channel):
        return Message(
            author=Author(role=Role.TOOL, name=self.get_tool_name()),
            content=[TextContent(text=text)]
        ).with_recipient("assistant").with_channel(channel)

    async def cleanup(self):
        if self.faiss_client:
            await self.faiss_client.close()
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()

async def check_faiss_cluster():
    client = FAISSClient()
    health = await client.health_check()

    print("\n" + "=" * 70)
    print("FAISS IVFPQ Service Cluster Health Check")
    print("=" * 70)
    print(json.dumps(health, indent=2, ensure_ascii=False))
    print("=" * 70 + "\n")

    await client.close()
    return health

if __name__ == "__main__":
    asyncio.run(check_faiss_cluster())
