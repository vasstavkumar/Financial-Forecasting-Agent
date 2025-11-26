from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.utils.utils import ProcessRequest
from src.data_layer.sql_operations import log_request_response
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = str(uuid.uuid4())
    query = request.query

    try:
        # Log incoming request
        await log_request_response(
            request_id=session_id,
            request_data={"query": query},
            response_data={}
        )

        # Process the request
        process_request = ProcessRequest()
        response = await process_request.process_request(query, session_id)

        # Log response
        await log_request_response(
            request_id=session_id,
            request_data={"query": query},
            response_data=response
        )

        return response

    except Exception as e:
        error_response = {
            "status_code": 500,
            "status_messages": f"An error occurred: {str(e)}"
        }

        # Log error
        await log_request_response(
            request_id=session_id,
            request_data={"query": query},
            response_data=error_response
        )

        return error_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)