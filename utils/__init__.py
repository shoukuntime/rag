from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from pydantic import SecretStr
from env_settings import EnvSettings

env_settings = EnvSettings()

llm = ChatGoogleGenerativeAI(
    model=env_settings.MODEL_NAME,
    google_api_key=SecretStr(env_settings.GOOGLE_API_KEY)
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=env_settings.EMBEDDING_MODEL,
    google_api_key=SecretStr(env_settings.GOOGLE_API_KEY)
)

vector_store = PGVector(
    collection_name=env_settings.COLLECTION_NAME,
    connection=env_settings.POSTGRES_URI,
    embeddings=embeddings,
)
