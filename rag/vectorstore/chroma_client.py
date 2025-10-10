"""
ChromaDB 클라이언트 초기화 모듈
"""
from langchain_chroma import Chroma
from rag.vectorstore.embeddings import get_embeddings

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "merchant_docs"

def get_vectorstore(task_type: str = "retrieval_document"):
    """
    ChromaDB 벡터스토어 인스턴스 반환

    Args:
        task_type: "retrieval_document" (적재) 또는 "retrieval_query" (조회)
    """
    embeddings = get_embeddings(task_type=task_type)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    return vectorstore
