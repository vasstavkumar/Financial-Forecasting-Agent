from pathlib import Path
import uuid
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data_extraction.text_extraction import TextExtractor
from src.data_layer.vectordb_operations import VectorDBOperations
from src.forecasting_agent.agent.agent import ForecastingAgent

class ProcessRequest:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.vector_db = VectorDBOperations()
        self.forecasting_agent = ForecastingAgent()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )

    async def ingestion_to_vector(self, session_id: str = None):
        session_id = session_id or "default"
        records = []
        
        transcripts_dir = Path("data/transcripts")
        quarterly_dir = Path("data/quaterly")
        
        if transcripts_dir.exists():
            pdf_files = list(transcripts_dir.glob("*.pdf"))
            for pdf_path in pdf_files:
                try:
                    text = await self.text_extractor.extract_pdf_text_pymupdf(session_id, str(pdf_path))
                    if text:
                        chunks = self.text_splitter.split_text(text)
                        for i, chunk in enumerate(chunks):
                            record_id = f"{pdf_path.stem}_chunk_{i}"
                            records.append({
                                "id": record_id,
                                "text": chunk,
                                "type": "transcriptions",
                                "source_file": pdf_path.name,
                                "chunk_index": i
                            })
                except Exception as e:
                    print(f"{session_id}: Error processing {pdf_path}: {e}")
        
        if quarterly_dir.exists():
            excel_files = list(quarterly_dir.glob("*.xlsx"))
            for excel_path in excel_files:
                try:
                    chunks = await self.text_extractor.chunk_excel(
                        session_id, 
                        str(excel_path), 
                        self.text_splitter, 
                        chunk_size=1000
                    )
                    if chunks:
                        for i, chunk in enumerate(chunks):
                            record_id = f"{excel_path.stem}_chunk_{i}"
                            records.append({
                                "id": record_id,
                                "text": chunk,
                                "type": "quarterly_reports",
                                "source_file": excel_path.name,
                                "chunk_index": i
                            })
                except Exception as e:
                    print(f"{session_id}: Error processing {excel_path}: {e}")
        
        if records:
            try:
                await self.vector_db.upsert_records(records)
                print(f"{session_id}: Successfully ingested {len(records)} records to vector database")
                return {"status": "success", "records_ingested": len(records)}
            except Exception as e:
                print(f"{session_id}: Error upserting to vector database: {e}")
                return {"status": "error", "message": str(e)}
        else:
            print(f"{session_id}: No records to ingest")
            return {"status": "no_data", "message": "No data found to ingest"}

    async def process_request(self, query: str, session_id: str = None):
        session_id = session_id or str(uuid.uuid4())
        
        ingestion_result = await self.ingestion_to_vector(session_id)
        
        if ingestion_result.get("status") == "error":
            return {
                "status_code": 500,
                "status_messages": f"Ingestion failed: {ingestion_result.get('message')}"
            }
        
        forecast_result = await self.forecasting_agent.forecasting_call(query, session_id)
        
        return forecast_result
