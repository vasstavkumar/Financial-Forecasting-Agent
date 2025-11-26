from langchain.tools import tool

from src.data_layer.vectordb_operations import VectorDBOperations
vector_db = VectorDBOperations()

@tool(parse_docstring=True)
async def think(thought: str):
    """
    Scratchpad tool for reasoning step-by-step and breaking down complex problems. Draft your thoughts, how to proceed further, plan, current stage, what to do next, etc.

    Args:
        thought: Detailed reasoning or internal step about the problem. Max 20 words.
    """
    if thought:
        print(f"Thought: {thought}")
    return "Thought logged."

@tool(parse_docstring=True)
async def analyze(analysis: str):
    """
    Evaluation tool to analyze the results of reasoning or actions and decide next steps. Use this mainly after tool calls to put your observation on the tool's output.

    Args:
        analysis: Explanation of how the result is interpreted or assessed. Max 20 words.
    """
    if analysis:
        print(f"Analysis: {analysis}")
    return "Analysis logged."

@tool(parse_docstring=True)
async def financial_data_extractor(query: str, k: int = 10):
    """
    A robust tool designed to understand quarterly financial reports and extract key financial metrics (e.g., Total Revenue, Net Profit, Operating Margin).

    Args:
        query: The query to search for financial data.
        k: The number of results to be returned.
    """
    try:
        results = await vector_db.search_records(
            query=query,
            top_k=k,
            namespace=None
        )

        contexts = []
        for result in results:
            if isinstance(result, dict) and result.get('score', 0) > 0.1:
                metadata = result.get('metadata', {})
                if isinstance(metadata, dict) and metadata.get('type') == 'quarterly_reports':
                    chunk_content = metadata.get('chunk_text') or metadata.get('text') or ''
                    if not chunk_content:
                        chunk_content = "[Content embedded in vector]"
                    contexts.append(f"Source: {metadata.get('source_file', 'Unknown')}\nContent: {chunk_content}")
                if len(contexts) >= 5:
                    break

        context = '\n-------\n'.join(contexts)
        return context if context else "No relevant quarterly financial data found."

    except Exception as e:
        print(f"There was an error in the financial_data_extractor tool: {e}")
        return "There was an error extracting financial data."

@tool(parse_docstring=True)
async def qualitative_analysis(query: str, k: int = 10):
    """
    A RAG-based tool that performs semantic search and analysis across 2-3 past earnings call transcripts to identify recurring themes, management sentiment, and forward-looking statements.

    Args:
        query: The query to search for qualitative analysis.
        k: The number of results to be returned.
    """
    try:
        results = await vector_db.search_records(
            query=query,
            top_k=k,
            namespace=None
        )

        contexts = []
        for result in results:
            if isinstance(result, dict) and result.get('score', 0) > 0.1:
                metadata = result.get('metadata', {})
                if isinstance(metadata, dict) and metadata.get('type') == 'transcriptions':
                    chunk_content = metadata.get('chunk_text') or metadata.get('text') or ''
                    if not chunk_content:
                        chunk_content = "[Content embedded in vector]"
                    contexts.append(f"Source: {metadata.get('source_file', 'Unknown')}\nContent: {chunk_content}")
                if len(contexts) >= 5:
                    break

        context = '\n-------\n'.join(contexts)
        return context if context else "No relevant transcript data found."

    except Exception as e:
        print(f"There was an error in the qualitative_analysis tool: {e}")
        return "There was an error performing qualitative analysis."