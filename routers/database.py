from fastapi import APIRouter, Depends, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
from starlette import status
from env_settings import EnvSettings

env_settings = EnvSettings()

router = APIRouter()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            url=env_settings.POSTGRES_URI
        )
        return conn
    except psycopg2.OperationalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {e}"
        )

@router.get("", tags=["database"])
async def get_all_postgres_data(table_name: str, conn: psycopg2.extensions.connection = Depends(get_db_connection)):
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if not table_name.isalnum() and "_" not in table_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid table name."
                )

            cursor.execute(f"SELECT * FROM langchain_pg_embedding")
            data = cursor.fetchall()
            return data
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {e}"
        )
    finally:
        if conn:
            conn.close()
