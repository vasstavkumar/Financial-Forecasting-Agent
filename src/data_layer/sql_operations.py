import json
import os
from datetime import datetime, timezone
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_connection_pool():
    """Create a PostgreSQL connection pool."""
    try:
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB")
        
        if not password:
            print("Warning: POSTGRES_PASSWORD not set in environment variables")
            return None
        
        pool = SimpleConnectionPool(
            1, 10,
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )
        return pool
    except Exception as e:
        print(f"Error creating PostgreSQL connection pool: {e}")
        return None

connection_pool = create_connection_pool()

async def log_request_response(request_id: str, request_data: dict, response_data: dict):
    if connection_pool is None:
        print("No database connection pool available.")
        return False

    conn = None
    cursor = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_logs (
                id SERIAL PRIMARY KEY,
                request_id VARCHAR(255) NOT NULL,
                request_data TEXT NOT NULL,
                response_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        cursor.execute("""
            INSERT INTO forecast_logs (request_id, request_data, response_data, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (
            request_id,
            json.dumps(request_data),
            json.dumps(response_data),
            datetime.now(timezone.utc)
        ))
        conn.commit()

        cursor.execute("SELECT * FROM forecast_logs WHERE request_id = %s", (request_id,))
        result = cursor.fetchall()
        print(f"Logged record with request_id {request_id}: {result}")
        return True

    except Exception as e:
        print(f"Database error while logging request/response: {e}")
        return False

    finally:
        if cursor:
            cursor.close()
        if conn:
            connection_pool.putconn(conn)

async def fetch_recent_logs(limit=20):
    if connection_pool is None:
        print("No database connection pool available.")
        return []
    
    conn = None
    cursor = None
    try:
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, request_data, response_data FROM forecast_logs ORDER BY timestamp DESC LIMIT %s", (limit,)
        )
        logs = cursor.fetchall()
        # Convert to list of dicts for consistency
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in logs]
        return result
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            connection_pool.putconn(conn)