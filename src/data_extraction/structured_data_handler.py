import tiktoken
import pandas as pd
from config import config


class StructuredDataHandler:
    def __init__(self):
        chunking_model = config.chunking_model
        self.encoding = tiktoken.encoding_for_model(chunking_model)

    def estimate_tokens(self, text):
        return len(self.encoding.encode(text))

    def chunk_dataframe(self, df, chunk_size):
        chunks = []
        header_str = ','.join([str(i) for i in df.columns])
        header_size = self.estimate_tokens(header_str)
        current_chunk = []
        current_chunk_size = header_size
        
        for _, row in df.iterrows():
            row_str = ','.join(map(str, row.values))
            row_size = self.estimate_tokens(row_str)
            
            if row_size > (chunk_size - header_size):
                if current_chunk:
                    chunks.append(pd.DataFrame(current_chunk, columns=df.columns))
                    current_chunk = []
                    current_chunk_size = header_size
                
                single_row_chunk = pd.DataFrame([row.to_dict()], columns=df.columns)
                chunks.append(single_row_chunk)
                current_chunk = []
                current_chunk_size = header_size
                continue
            
            if current_chunk and (current_chunk_size + row_size) > chunk_size:
                chunks.append(pd.DataFrame(current_chunk, columns=df.columns))
                current_chunk = []
                current_chunk_size = header_size
            
            current_chunk.append(row.to_dict())
            current_chunk_size += row_size
        
        if current_chunk:
            chunks.append(pd.DataFrame(current_chunk, columns=df.columns))
        
        return chunks

    def load_and_chunk_csv(self, file_path, chunk_size):
        df = pd.read_csv(file_path)
        return self.chunk_dataframe(df, chunk_size)

    def dataframe_to_markdown(self, df):
        if df.empty:
            return None
        
        try:
            return df.to_markdown(index=False)
        except:
            markdown = []
            headers = df.columns.tolist()
            markdown.append("| " + " | ".join(str(h) for h in headers) + " |")
            markdown.append("| " + " | ".join(["---" for _ in headers]) + " |")
            
            for _, row in df.iterrows():
                markdown.append("| " + " | ".join(str(v) if pd.notna(v) else "" for v in row) + " |")
            
            return "\n".join(markdown)
        

    def sheet_requires_header(self, df, irregularity_threshold=0.5):
        header_row = df.columns
        unnamed_cols_ratio = sum(header_row.str.contains('Unnamed')) / len(header_row)

        if unnamed_cols_ratio > 0.5:
            return False

        non_empty_rows = df.dropna(how='all')
        if non_empty_rows.empty:
            return False

        col_counts = non_empty_rows.apply(lambda row: row.count(), axis=1)
        if col_counts.empty:
            return False
        
        mode_count = col_counts.mode().iloc[0]
        inconsistency_ratio = (col_counts != mode_count).mean()

        return inconsistency_ratio < irregularity_threshold

    def extract_sheet_as_text(self, df):
        header = ' '.join(df.columns.astype(str)).strip()
        rows = df.fillna('').astype(str).apply(lambda row: ' '.join(row).strip(), axis=1).tolist()
        return "\n".join([header] + rows)


    def load_and_chunk_excel(self, file_path, chunk_size, kb_text_splitter):
        excel_data = pd.ExcelFile(file_path)
        all_chunks = {}

        for sheet_name in excel_data.sheet_names:
            df = excel_data.parse(sheet_name)
            if not df.empty:
                include_header = self.sheet_requires_header(df)

                if include_header:
                    chunks = self.chunk_dataframe(df, chunk_size)
                else:
                    sheet_text = self.extract_sheet_as_text(df)
                    if sheet_text:
                        chunks = kb_text_splitter.split_text(sheet_text)
                    else:
                        chunks = []
                
                if chunks:
                    all_chunks[sheet_name] = chunks

        return all_chunks