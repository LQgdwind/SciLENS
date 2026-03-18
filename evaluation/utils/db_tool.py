import pymongo
from pymongo import MongoClient
from nltk.stem import PorterStemmer
import json
import re

class PaperManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="academic_db", collection_name="papers"):
        self.client = MongoClient(
            uri,
            maxPoolSize=200,
            connectTimeoutMS=2000,
            socketTimeoutMS=10000,
            serverSelectionTimeoutMS=3000
        )
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.stemmer = PorterStemmer()

    def _to_int(self, val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return val

    def search_by_keywords(self, keywords, limit=10, skip=0):
        if isinstance(keywords, str):
            keywords = [keywords]

        stemmed_terms = []
        for k in keywords:
            if not k.strip(): continue
            stem = self.stemmer.stem(k)
            stemmed_terms.append(f'"{stem}"')

        if not stemmed_terms: return []
        search_str = " ".join(stemmed_terms)

        query = {"$text": {"$search": search_str}}
        projection = {"title": 1, "authors.name": 1, "year": 1, "n_citation": 1, "abstract": 1, "venue": 1}

        try:
            cursor = self.collection.find(query, projection)\
                                    .sort([("n_citation", -1), ("year", -1)])\
                                    .skip(skip)\
                                    .limit(limit)\
                                    .max_time_ms(5000)
            return list(cursor)
        except Exception as e:
            print(f"[Error] Keyword search failed: {e}")
            return []

    def get_paper_detail(self, paper_id):
        return self.collection.find_one({"_id": self._to_int(paper_id)})

    def get_papers_by_ids(self, id_list, fields=None):
        if not id_list:
            return []

        query = {"_id": {"$in": [self._to_int(pid) for pid in id_list]}}

        if fields is None:
            fields = {"title": 1, "authors.name": 1, "year": 1, "n_citation": 1, "venue": 1}

        return list(self.collection.find(query, fields).max_time_ms(3000))

    def get_papers_by_faiss_ids(self, faiss_id_list):
        if not faiss_id_list: return []
        query = {"faiss_id": {"$in": [self._to_int(fid) for fid in faiss_id_list]}}
        return list(self.collection.find(query).max_time_ms(3000))

    def get_references(self, identifier, by="title"):
        source_paper = None

        if by == "id":
            source_paper = self.collection.find_one({"_id": identifier}, {"references": 1})
        elif by == "title":
            source_paper = self.collection.find_one(
                {"$text": {"$search": identifier}},
                {"references": 1, "score": {"$meta": "textScore"}},
                sort=[("score", {"$meta": "textScore"})]
            )
        else:
            return []

        if not source_paper:
            print(f"[Warn] Paper not found: {identifier}")
            return []

        ref_ids = source_paper.get("references", [])
        if not ref_ids:
            return []

        MAX_LIMIT = 3000

        if len(ref_ids) > MAX_LIMIT:
            print(f"[Warn] 引用过多 ({len(ref_ids)}条)，仅保留前 {MAX_LIMIT} 条处理。")

            ref_ids = ref_ids[:MAX_LIMIT]
        all_reference_details = []
        BATCH_SIZE = 1000

        for i in range(0, len(ref_ids), BATCH_SIZE):
            batch_ids = ref_ids[i : i + BATCH_SIZE]

            try:

                batch_docs = list(self.collection.find(
                    {"_id": {"$in": batch_ids}},
                    {"title": 1, "year": 1, "n_citation": 1, "venue": 1}
                ))
                all_reference_details.extend(batch_docs)

            except Exception as e:
                print(f"[Error] Batch {i} query failed: {e}")
                continue

        return all_reference_details

    def get_paper_info(self, identifier, by="title"):
        if by == "id":
            return self.collection.find_one({"_id": self._to_int(identifier)})

        elif by == "title":

            query = {"$text": {"$search": f'"{identifier}"'}}
            res = self.collection.find_one(query, {"_id": 1})
            if res:
                return self.collection.find_one({"_id": res["_id"]})
        return None

    def get_k_hop_references(self, identifier, k=2, by="title", max_nodes=500):
        start_node = self.get_paper_info(identifier, by=by)
        if not start_node: return []

        start_node["hop"] = 0
        results = [start_node]
        visited_ids = {start_node["_id"]}

        current_layer_ids = start_node.get("references", [])
        LAYER_SEARCH_CAP = 2000
        DB_BATCH_SIZE = 1000

        for current_hop in range(1, k + 1):
            if not current_layer_ids or len(results) >= max_nodes:
                break
            unique_candidates = list(dict.fromkeys([
                pid for pid in current_layer_ids if pid not in visited_ids
            ]))

            candidates_to_process = unique_candidates[:LAYER_SEARCH_CAP]

            if not candidates_to_process:
                break

            layer_docs = []

            for i in range(0, len(candidates_to_process), DB_BATCH_SIZE):
                batch_ids = candidates_to_process[i : i + DB_BATCH_SIZE]
                try:
                    pipeline = [
                        {"$match": {"_id": {"$in": batch_ids}}},
                        {"$project": {
                            "title": 1, "year": 1, "n_citation": 1, "venue": 1, "references": 1
                        }}
                    ]
                    batch_results = list(self.collection.aggregate(pipeline))
                    layer_docs.extend(batch_results)
                except Exception as e:
                    print(f"[Warning] Batch query error: {e}")

            layer_docs.sort(key=lambda x: x.get("n_citation", 0), reverse=True)

            remaining_slots = max_nodes - len(results)
            docs_to_keep = layer_docs[:remaining_slots]

            next_layer_ids = []

            for doc in docs_to_keep:
                visited_ids.add(doc["_id"])
                doc["hop"] = current_hop
                results.append(doc)

                if "references" in doc and isinstance(doc["references"], list):
                    next_layer_ids.extend(doc["references"])

            current_layer_ids = next_layer_ids

            if len(results) >= max_nodes:
                break

        return results

    def find_shortest_path(self, start_identifier, end_identifier, max_depth=4, by="title"):
        start_node = self.get_paper_info(start_identifier, by=by)
        end_node = self.get_paper_info(end_identifier, by=by)
        if not start_node or not end_node: return None

        start_id, end_id = start_node["_id"], end_node["_id"]
        if start_id == end_id: return [start_node]
        if end_id in [self._to_int(r) for r in start_node.get("references", [])]:
            return [start_node, end_node]

        front_visited, back_visited = {start_id: None}, {end_id: None}
        front_layer, back_layer = [start_id], [end_id]

        MAX_LAYER = 100000

        for _ in range(max_depth):
            if not front_layer or not back_layer: break

            if len(front_layer) <= len(back_layer):
                cursor = self.collection.find({"_id": {"$in": front_layer}}, {"references": 1}).max_time_ms(2000)
                next_layer = []
                for doc in cursor:
                    pid = doc["_id"]
                    for child in [self._to_int(c) for c in doc.get("references", [])]:
                        if child not in front_visited:
                            front_visited[child] = pid
                            next_layer.append(child)
                            if child in back_visited:
                                return self._reconstruct_bidirectional_path(child, front_visited, back_visited)
                front_layer = next_layer[:MAX_LAYER]
            else:

                cursor = self.collection.find({"references": {"$in": back_layer}}, {"_id": 1, "references": 1}).max_time_ms(2000)
                next_layer = []
                back_set = set(back_layer)
                for doc in cursor:
                    pid = doc["_id"]
                    for r in [self._to_int(ri) for ri in doc.get("references", [])]:
                        if r in back_set:
                            if pid not in back_visited:
                                back_visited[pid] = r
                                next_layer.append(pid)
                                if pid in front_visited:
                                    return self._reconstruct_bidirectional_path(pid, front_visited, back_visited)
                            break
                back_layer = next_layer[:MAX_LAYER]
        return None

    def _reconstruct_bidirectional_path(self, meet_node, front_map, back_map):
        path_front = []
        curr = meet_node
        while curr is not None:
            path_front.append(curr)
            curr = front_map.get(curr)
        path_front.reverse()

        path_back = []
        curr = back_map.get(meet_node)
        while curr is not None:
            path_back.append(curr)
            curr = back_map.get(curr)

        full_ids = path_front + path_back
        unordered_docs = self.get_papers_by_ids(full_ids)
        doc_map = {doc["_id"]: doc for doc in unordered_docs}
        return [doc_map[pid] for pid in full_ids if pid in doc_map]

    def get_in_degree(self, identifier, by="title", limit=50):
        target_id = identifier
        if by == "title":
            info = self.get_paper_info(identifier, by="title")
            if not info:
                print(f"[Warn] Target paper not found for cited_by query: {identifier}")
                return []
            target_id = info["_id"]

        query = {"references": target_id}

        projection = {"title": 1, "year": 1, "n_citation": 1, "venue": 1}

        try:
            cursor = self.collection.find(query, projection)\
                                    .limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"[Error] get_cited_by failed: {e}")
            return []

    def close(self):
        self.client.close()
