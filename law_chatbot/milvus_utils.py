import os
from pymilvus import connections, Collection, utility

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = 19530
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
MILVUS_DB = os.getenv("MILVUS_DB", "default")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "legal_chunks") 

def connect_milvus():
    if connections.has_connection("default"):
        connections.disconnect("default")
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
        db_name=MILVUS_DB,
        timeout=10,
        secure=False
    )

def search_milvus(query, get_embedding, top_k=7):
    connect_milvus()
    embedding = get_embedding(query)
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 8}},
        limit=top_k,
        output_fields=["chunk_id", "content", "parent_id"]
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id": hit.entity.get("chunk_id"),
            "content": hit.entity.get("content"),
            "parent_id": hit.entity.get("parent_id"),
            "score": hit.distance
        })
    return hits
