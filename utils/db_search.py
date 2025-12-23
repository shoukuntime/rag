import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from env_settings import EnvSettings
from typing import List, Dict, Any
import json

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    env_settings = EnvSettings()
    try:
        conn = psycopg2.connect(env_settings.POSTGRES_URI)
        return conn
    except psycopg2.OperationalError:
        # In a real application, you'd want to log this error.
        return None

def search_articles(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Searches for relevant articles in the database based on the question keywords.
    """
    conn = get_db_connection()
    if conn is None:
        return []

    results = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # A simple keyword search using ILIKE.
            # This can be improved with more advanced full-text search capabilities.
            search_query = f"%{question}%"
            query = sql.SQL("""
                SELECT document, cmetadata FROM {} 
                WHERE document ILIKE %s
                LIMIT %s
            """).format(sql.Identifier('langchain_pg_embedding'))
            
            cursor.execute(query, (search_query, top_k))
            fetched_results = cursor.fetchall()
            
            # The 'cmetadata' likely contains the law name and other details.
            # We'll format it for consistency.
            for row in fetched_results:
                results.append({
                    "page_content": row['document'],
                    "metadata": row['cmetadata']
                })

    except psycopg2.Error:
        # Log the error
        pass
    finally:
        if conn:
            conn.close()
            
    return results

def search_articles_by_numbers(article_numbers: List[str]) -> List[Dict[str, Any]]:
    """
    Searches for articles in the database where metadata references match the given article numbers.
    """
    if not article_numbers:
        return []

    conn = get_db_connection()
    if conn is None:
        return []

    results = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Construct a query to find documents where cmetadata->'references' contains the article number.
            # We assume 'references' is a JSON array of strings in the metadata.
            
            conditions = []
            params = []
            for num in article_numbers:
                # Check if the 'references' array contains the number string
                # Using JSONB containment operator @>
                # We cast cmetadata to jsonb just in case it's json type
                conditions.append("cmetadata::jsonb -> 'references' @> %s::jsonb")
                # The parameter must be a JSON string representing an array containing the element
                params.append(json.dumps([num]))

            where_clause = sql.SQL(" OR ").join([sql.SQL(c) for c in conditions])
            
            query = sql.SQL("""
                SELECT document, cmetadata FROM {} 
                WHERE {}
            """).format(sql.Identifier('langchain_pg_embedding'), where_clause)
            
            cursor.execute(query, tuple(params))
            fetched_results = cursor.fetchall()
            
            for row in fetched_results:
                results.append({
                    "page_content": row['document'],
                    "metadata": row['cmetadata']
                })

    except psycopg2.Error as e:
        print(f"DB Error in search_articles_by_numbers: {e}")
    finally:
        if conn:
            conn.close()
            
    return results
