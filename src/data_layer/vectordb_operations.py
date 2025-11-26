from pinecone import Pinecone
from typing import List, Dict, Optional
from config import config


def batch_chunks(lst, n=96):
    """Split a list into batches of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class VectorDBOperations:
    def __init__(self):
        if not config.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in config")
        if not config.pinecone_index_name:
            raise ValueError("Pinecone index name not configured in config")
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index_name = config.pinecone_index_name

    async def create_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                }
            )
        return self.index_name

    def get_index(self):
        return self.pc.Index(self.index_name)

    async def upsert_records(self, records: List[Dict], namespace: Optional[str] = None):
        await self.create_index()
        dense_index = self.get_index()
        
        validated_records = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Record must be a dictionary, got {type(record)}")
            
            formatted_record = {
                "_id": record.get("id") or record.get("_id"),
                "chunk_text": record.get("text") or record.get("chunk_text")
            }
            
            if not formatted_record["_id"]:
                raise ValueError("Record must have an 'id' or '_id' field")
            if not formatted_record["chunk_text"]:
                raise ValueError("Record must have a 'text' or 'chunk_text' field for embedding")
            
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict) and metadata:
                formatted_record.update(metadata)
            else:
                excluded_keys = {"id", "_id", "text", "chunk_text", "metadata"}
                for key, value in record.items():
                    if key not in excluded_keys:
                        formatted_record[key] = value
            
            validated_records.append(formatted_record)
        
        namespace_param = namespace if namespace else "__default__"
        
        for batch in batch_chunks(validated_records, n=96):
            dense_index.upsert_records(namespace_param, batch)

    async def search_records(self, query: str, top_k: int = 10, namespace: Optional[str] = None, rerank: bool = False):
        dense_index = self.get_index()
        namespace_param = namespace if namespace else "__default__"
        
        try:
            inference_model = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[query],
                parameters={"input_type": "query"}
            )
            
            if hasattr(inference_model, 'data') and inference_model.data:
                query_vector = inference_model.data[0].values
            elif isinstance(inference_model, dict) and 'data' in inference_model:
                query_vector = inference_model['data'][0].get('values', [])
            else:
                query_vector = inference_model.values if hasattr(inference_model, 'values') else []
            
            if not query_vector:
                return []
            
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": namespace_param
            }
            
            if rerank:
                query_params["rerank"] = {
                    "model": "bge-reranker-v2-m3",
                    "top_n": 10,
                    "rank_fields": ["chunk_text"]
                }
            
            results = dense_index.query(**query_params)
            
            if hasattr(results, 'matches'):
                matches = results.matches
            elif isinstance(results, dict):
                matches = results.get("matches", [])
            else:
                matches = []
            
            formatted_results = []
            for match in matches:
                if isinstance(match, dict):
                    match_id = match.get("id")
                    match_score = match.get("score", 0)
                    match_metadata = match.get("metadata", {})
                    if not match_metadata:
                        match_metadata = {k: v for k, v in match.items() if k not in ["id", "score", "values"]}
                else:
                    match_id = getattr(match, "id", None)
                    match_score = getattr(match, "score", 0)
                    match_metadata = getattr(match, "metadata", {})
                    if not match_metadata or not isinstance(match_metadata, dict):
                        match_metadata = {}
                        for attr in ["type", "source_file", "chunk_index", "chunk_text", "text"]:
                            if hasattr(match, attr):
                                match_metadata[attr] = getattr(match, attr)
                
                formatted_results.append({
                    "id": match_id,
                    "score": match_score,
                    "metadata": match_metadata if isinstance(match_metadata, dict) else {}
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in search_records: {e}")
            return []