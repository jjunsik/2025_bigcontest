"""
벡터DB 검색 서비스
LLM 전략과 RAG content 간 유사도 검색
"""

from rag.vectorstore.faiss_client import get_vectorstore
from typing import List, Tuple
from langchain.schema import Document

def search_context(
    query: str,
    similarity_threshold: float = 0.7,
    fetch_k: int = 10
) -> Tuple[str, List[Document]]:
    """
    LLM 전략과 유사한 RAG content 검색

    검색 방식:
    1. query(LLM 전략)를 임베딩
    2. RAG content와 코사인 유사도 계산
    3. similarity_threshold 이상인 문서만 반환

    Args:
        query: LLM이 수립한 전략
        similarity_threshold: 유사도 임계값 (0~1)
        fetch_k: 최대 검색 개수

    Returns:
        Tuple[str, List[Document]]:
            - str: 컨텍스트 텍스트 (표시용)
            - List[Document]: Document 객체 리스트
              - doc.page_content: content (순수 내용)
              - doc.metadata: {"channel": ..., "title": ..., "video_link": ...}

    예시:
        query = "재방문 쿠폰 제공 전략"
        context, docs = search_context(query)

        for doc in docs:
            print(doc.page_content)  # "[마케팅 전략]\n쿠폰..."
            print(doc.metadata)      # {"channel": "잘 파는 청년", ...}
    """
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return "", []

    # FAISS 검색: query와 content 간 유사도 계산
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=fetch_k,
        score_threshold=similarity_threshold
    )

    # Document 추출
    filtered_docs = [doc for doc, score in docs_with_scores]

    # 컨텍스트 생성 (표시용)
    context_parts = []
    for doc in filtered_docs:
        # metadata + content 조합
        context_parts.append(
            f"채널: {doc.metadata.get('channel', '알 수 없음')}\n"
            f"제목: {doc.metadata.get('title', '제목 없음')}\n"
            f"영상 링크: {doc.metadata.get('video_link', '')}\n\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context, filtered_docs
