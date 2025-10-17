"""
Gemini 임베딩 초기화 모듈
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings(task_type: str = "retrieval_document"):
    """
    GoogleGenerativeAIEmbeddings 인스턴스 반환

    Args:
        task_type: "retrieval_document" (적재) 또는 "retrieval_query" (조회)

    Returns:
        GoogleGenerativeAIEmbeddings 인스턴스
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type=task_type
    )
