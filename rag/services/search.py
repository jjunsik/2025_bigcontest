"""
벡터DB 검색 서비스
"""

from rag.vectorstore.faiss_client import get_vectorstore
from typing import List, Tuple

from langchain.schema import Document


def search_context(query: str, k: int = 5) -> Tuple[str, List[Document]]:
    """
    질의에 대한 유사 문서를 검색합니다.

    Args:
        query: 검색 쿼리
        k: 반환할 문서 개수

    Returns:
        Tuple[str, List[Document]]: (컨텍스트 텍스트, Document 객체 리스트)
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "", []

    # 유사도 검색
    docs = vectorstore.similarity_search(query, k=k)

    # 컨텍스트 텍스트 생성
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Document 객체 그대로 반환
    return context, docs
