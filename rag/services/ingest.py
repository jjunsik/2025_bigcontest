"""
문서 청킹 및 벡터DB 적재 서비스
- 각 row를 개별 문서로 저장
- content만 임베딩 (순수 내용)
- metadata는 Document에 저장 (검색 결과와 함께 반환)
"""

import os
import shutil
import time

import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from rag.vectorstore.embeddings import get_embeddings

FAISS_PATH = "./faiss_db"
BATCH_SIZE = 7
DELAY_SECONDS = 10


def ingest_youtube_tips_csv(csv_path: str, append_mode: bool = True) -> int:
    """
    유튜브 팁 CSV를 벡터DB에 적재

    저장 방식:
    - page_content: content만 (임베딩 대상)
    - metadata: channel, title, video_link (검색 결과와 함께 반환)

    Args:
        csv_path: CSV 파일 경로
        append_mode: True면 기존 데이터에 추가, False면 완전 교체

    Returns:
        적재된 문서 개수
    """
    df = pd.read_csv(csv_path)
    print(f"📊 CSV 파일 로드 완료: {len(df)}개 행")

    # ============================================
    # 1. video_link별 metadata 수집
    # ============================================
    video_metadata = {}

    for idx, row in df.iterrows():
        if pd.isna(row['video_link']) or str(row['video_link']).strip() == '':
            continue

        video_link = str(row['video_link']).strip()

        if video_link not in video_metadata:
            video_metadata[video_link] = {
                "channel": None,
                "title": None
            }

        if video_metadata[video_link]["channel"] is None and pd.notna(row.get('channel')):
            video_metadata[video_link]["channel"] = str(row['channel'])

        if video_metadata[video_link]["title"] is None and pd.notna(row.get('title')):
            video_metadata[video_link]["title"] = str(row['title'])

    # metadata 기본값 설정
    for video_link, metadata in video_metadata.items():
        if metadata["channel"] is None:
            metadata["channel"] = "알 수 없음"
        if metadata["title"] is None:
            metadata["title"] = "제목 없음"

    # ============================================
    # 2. 각 row를 개별 문서로 생성
    # ============================================
    texts = []
    metadatas = []
    skipped_rows = 0

    for idx, row in df.iterrows():
        if pd.isna(row['video_link']) or str(row['video_link']).strip() == '':
            skipped_rows += 1
            continue

        video_link = str(row['video_link']).strip()
        channel = video_metadata[video_link]["channel"]
        title = video_metadata[video_link]["title"]

        # content 수집
        content_parts = []

        if pd.notna(row.get('content_marketing')):
            content_parts.append(f"[마케팅 전략]\n{row['content_marketing']}")

        if pd.notna(row.get('content_issue')):
            content_parts.append(f"[문제 상황]\n{row['content_issue']}")

        if pd.notna(row.get('content_solution')):
            content_parts.append(f"[해결 방법]\n{row['content_solution']}")

        if not content_parts:
            skipped_rows += 1
            continue

        # ============================================
        # 핵심: content만 page_content로 사용 (임베딩 대상)
        # ============================================
        combined_content = "\n\n".join(content_parts)

        # page_content: content만 (metadata 제외)
        text = combined_content

        # metadata: Document에 저장 (검색 결과와 함께 반환)
        metadata = {
            "channel": channel,
            "title": title,
            "video_link": video_link
        }

        texts.append(text.strip())
        metadatas.append(metadata)

    if skipped_rows > 0:
        print(f"⚠️ 총 {skipped_rows}개 행 건너뜀")

    # ============================================
    # 3. 적재
    # ============================================
    total_docs = len(texts)

    if total_docs == 0:
        print("❌ 적재할 문서가 없습니다.")
        return 0

    print(f"📝 총 {total_docs}개 문서 생성 완료")
    print(f"   임베딩 대상: content만 (순수 내용)")
    print(f"   metadata 저장: channel, title, video_link")
    print(f"⏱️ 예상 소요 시간: 약 {(total_docs // BATCH_SIZE) * DELAY_SECONDS // 60 + 1}분")

    embeddings = get_embeddings(task_type="retrieval_document")

    # ============================================
    # 4. 기존 데이터 로드
    # ============================================
    vectorstore = None

    if append_mode and os.path.exists(FAISS_PATH):
        try:
            print(f"📂 기존 벡터스토어 로드 중...")
            vectorstore = FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ 기존 벡터스토어 로드 완료")
        except Exception as e:
            print(f"⚠️ 기존 벡터스토어 로드 실패: {e}")
            vectorstore = None

    # ============================================
    # 5. 배치 처리
    # ============================================
    for i in range(0, total_docs, BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]

        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n배치 {batch_num}/{total_batches}: {len(batch_texts)}개 문서")

        for j, metadata in enumerate(batch_metadatas):
            print(f"  - 문서 {i+j+1}: {metadata['channel']} | {metadata['title'][:30]}...")

        batch_documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(batch_texts, batch_metadatas)
        ]

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    batch_documents,
                    embeddings,
                    distance_strategy=DistanceStrategy.COSINE
                )
            else:
                vectorstore.add_documents(batch_documents)

            print(f"  ✅ 배치 {batch_num} 완료")
        except Exception as e:
            print(f"  ❌ 배치 {batch_num} 실패: {e}")
            print(f"  ⏳ 20초 대기...")
            time.sleep(20)
            continue

        if i + BATCH_SIZE < total_docs:
            print(f"  ⏳ {DELAY_SECONDS}초 대기...")
            time.sleep(DELAY_SECONDS)

    # ============================================
    # 6. 저장
    # ============================================
    if vectorstore:
        vectorstore.save_local(FAISS_PATH)
        print(f"\n✅ 총 {total_docs}개 문서 적재 완료!")

        try:
            final_vectorstore = FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            total_count = final_vectorstore.index.ntotal
            print(f"📊 벡터스토어 최종 문서 개수: {total_count}개")
        except:
            pass

        return total_docs
    else:
        print(f"❌ 적재 실패")
        return 0


def clear_vectorstore():
    """벡터DB 초기화"""
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
        print(f"🗑️ 벡터스토어 삭제 완료")
        return True
    else:
        print(f"⚠️ 삭제할 벡터스토어 없음")
        return False
