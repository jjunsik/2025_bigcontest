"""
FAISS 클라이언트 초기화 모듈
"""
from langchain_community.vectorstores import FAISS
from rag.vectorstore.embeddings import get_embeddings
import os

FAISS_PATH = "./faiss_db"

def get_vectorstore():
    """FAISS 벡터스토어 인스턴스 반환"""
    embeddings = get_embeddings()

    if os.path.exists(f"{FAISS_PATH}/index.faiss"):
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    else:
        return None

def get_document_count() -> int:
    """벡터DB에 저장된 문서 개수 반환"""
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return 0

    # FAISS 인덱스의 벡터 개수
    return vectorstore.index.ntotal
