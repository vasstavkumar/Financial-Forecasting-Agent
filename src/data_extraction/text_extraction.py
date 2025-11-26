import re
import time
import pymupdf4llm
import pandas as pd
from src.data_extraction.structured_data_handler import StructuredDataHandler

class TextExtractor:
    def __init__(self):
        self.structured_data_handler = StructuredDataHandler()

    async def extract_pdf_text_pymupdf(self, session_id, pdf_path: str):
        try:
            start_time = time.time()
            content = pymupdf4llm.to_markdown(pdf_path, show_progress=False)

            cleaned_text = re.sub(r'\n{3,}', '\n', content)
            cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n', cleaned_text)
            cleaned_text = cleaned_text.strip()
            cleaned_text = cleaned_text.replace('-----', '')
            end_time = time.time()
            print(f"{session_id}: Time taken to extract pdf: {end_time - start_time} seconds")

            return cleaned_text.strip()
        except Exception as e:
            print(f"ERROR: {session_id}: Error extracting pdf: {e}")
            return None
        
    async def chunk_excel(self, session_id, excel_path, kb_text_splitter, chunk_size):
        try:
            start_time = time.time()
            chunks_dict = self.structured_data_handler.load_and_chunk_excel(excel_path, chunk_size=chunk_size, kb_text_splitter=kb_text_splitter)
            if not chunks_dict:
                print(f"{session_id}: No chunks found in excel")
                return None
            
            chunks = []
            for sheet_name, sheet_data in chunks_dict.items():
                for data in sheet_data:
                    text = f"Sheet Name: {sheet_name}\n\n"

                    if type(data) == pd.DataFrame:
                        resp = self.structured_data_handler.dataframe_to_markdown(data)
                        if resp:
                            text += resp
                    else:
                        text += data
                    
                    chunks.append(text)
            
            end_time = time.time()
            print(f"{session_id}: Time taken to extract excel: {end_time - start_time} seconds")
            return chunks
        except Exception as e:
            print(f"ERROR: {session_id}: Error extracting excel: {e}")
            return None