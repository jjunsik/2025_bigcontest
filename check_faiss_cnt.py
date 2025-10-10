"""벡터DB 상태 확인"""
import os
from rag.vectorstore.faiss_client import get_document_count

if os.path.exists("./faiss_db/index.faiss"):
    count = get_document_count()
    print(f"✅ 벡터DB 존재: {count}개 문서")
else:
    print("❌ 벡터DB 없음")
