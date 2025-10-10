"""
벡터DB 검색 서비스
"""
from rag.vectorstore.faiss_client import get_vectorstore
from typing import List, Tuple
from langchain.schema import Document

def search_context(query: str, k: int = 5) -> Tuple[str, List[Document]]:
    """질의에 대한 유사 문서를 검색합니다."""
    vectorstore = get_vectorstore()

    if vectorstore is None:
        # DB가 없으면 빈 결과 반환
        return "", []

    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    return context, docs
