"""
문서 청킹 및 벡터DB 적재 서비스 (RPM 100 제한 대응 - 안전 버전)
"""

import os
import shutil
import time
from typing import List, Dict
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rag.vectorstore.embeddings import get_embeddings

FAISS_PATH = "./faiss_db"

# RPM 100 대응 설정 (더 안전하게)
BATCH_SIZE = 7
DELAY_SECONDS = 10

def ingest_youtube_tips_csv(csv_path: str) -> int:
    """
    유튜브 팁 CSV를 벡터DB에 적재 (RPM 100 제한 대응)

    Args:
        csv_path: CSV 파일 경로

    Returns:
        적재된 문서 개수
    """
    df = pd.read_csv(csv_path)

    texts = []
    metadatas = []

    # CSV 파싱
    for _, row in df.iterrows():
        if row['category'] == 'marketing_advice':
            text = f"""
                    채널: {row['channel']}
                    제목: {row['title']}
                    카테고리: 마케팅 조언
                    
                    마케팅 전략:
                    {row['content_marketing']}
                    """
        elif row['category'] == 'issue_solution':
            text = f"""
                    채널: {row['channel']}
                    제목: {row['title']}
                    카테고리: 문제 해결
                    
                    문제 상황:
                    {row['content_issue']}
                    
                    해결 방법:
                    {row['content_solution']}
                    """
        else:
            continue

        metadata = {
            "channel": str(row['channel']),
            "title": str(row['title']),
            "video_link": str(row['video_link']),
            "category": str(row['category'])
        }

        texts.append(text.strip())
        metadatas.append(metadata)

    # 배치 처리
    total_docs = len(texts)
    print(f"총 {total_docs}개 문서 적재 시작... (RPM 100 제한 대응 - 안전 모드)")
    print(f"예상 소요 시간: 약 {(total_docs // BATCH_SIZE) * DELAY_SECONDS // 60 + 1}분")

    embeddings = get_embeddings(task_type="retrieval_document")
    vectorstore = None

    for i in range(0, total_docs, BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]

        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"배치 {batch_num}/{total_batches}: {len(batch_texts)}개 문서 처리 중...")

        # Document 생성
        batch_documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(batch_texts, batch_metadatas)
        ]

        try:
            # FAISS 적재
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch_documents, embeddings)
            else:
                vectorstore.add_documents(batch_documents)

            print(f"  ✅ 배치 {batch_num} 완료")
        except Exception as e:
            print(f"  ❌ 배치 {batch_num} 실패: {e}")
            # 에러 발생 시 더 긴 대기 (20초)
            print(f"  ⏳ 에러 발생! 20초 대기 후 계속...")
            time.sleep(20)
            continue

        # Rate Limiting
        if i + BATCH_SIZE < total_docs:
            print(f"  ⏳ Rate Limiting: {DELAY_SECONDS}초 대기...")
            time.sleep(DELAY_SECONDS)

    # 저장
    if vectorstore:
        vectorstore.save_local(FAISS_PATH)
        print(f"✅ 총 {total_docs}개 문서 적재 완료!")
        return total_docs
    else:
        print(f"❌ 적재 실패")
        return 0

def clear_vectorstore():
    """벡터DB 초기화"""
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
        return True
    return False
