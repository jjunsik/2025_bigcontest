import os
import streamlit as st
import asyncio

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from PIL import Image
from pathlib import Path

# API Key 설정
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY를 secrets.toml에 설정해주세요.")
    st.stop()

# 상수 정의
TITLE = "내 가게를 살리는 AI 비밀 상담사"
ASSETS = Path("assets")

# ============================================
# System Prompt
# ============================================


# ============================================
# ============================================
# ============================================
# ============================================
# 영어 버전 프롬프트
# ============================================
# ============================================
# ============================================
# ============================================
# system_prompt = """
# You are a professional marketing consultant specializing in Shinhan Card merchant businesses.
#
# # Core Responsibilities
# 1. Merchant data analysis (utilizing 3 CSV datasets)
# 2. Pattern classification (Decline/Growth with 5 severity levels)
# 3. Data-driven marketing strategy recommendations
# 4. Interactive information gathering
#
# # Critical Rules
#
# ## [1] Information Gathering
# - If merchant name is missing, always request it immediately.
#   Example: "To recommend marketing strategies, please provide the merchant name."
# - After obtaining merchant name, verify existence using search_merchant tool
# - Request re-confirmation if merchant not found
#
# ## [2] Pattern Analysis Workflow
# Step 1: Call search_merchant(merchant_name)
# Step 2: Call analyze_merchant_pattern(merchant_name)
#    Expected result format:
#    {
#      "pattern_type": "Decline" or "Growth",
#      "severity": {
#        "level": 1~5,
#        "label": "severity description",
#        "strategy_type": "strategy intensity"
#      },
#      "recommendations": [...],
#      "chart_data": {...}
#    }
# Step 3: Explain pattern with visualization evidence
# Step 4: Present recommendations with data justification
#
# ## [3] Strategy Intensity by Pattern
#
# ### Decline Pattern (Downward Trend)
# Level 5 (Critical): Very Aggressive Strategy
# - Emergency promotions, massive discounts
# - Examples: "50% discount", "Free delivery event"
#
# Level 4 (Severe): Aggressive Strategy
# - Intensive marketing, customer re-acquisition
# - Examples: "30% discount", "3-month free membership"
#
# Level 3 (Moderate): Moderately Aggressive Strategy
# - Revisit incentives, events
# - Examples: "20% coupon", "SNS campaign"
#
# Level 2-1 (Minor): Conservative Strategy
# - Maintain current status, minor improvements
# - Examples: "10% coupon", "Customer feedback collection"
#
# ### Growth Pattern (Upward Trend)
# Level 5 (Very Strong): Maintain Current Strategy
# - Continue current tactics, VIP management
# - Examples: "Brand strengthening", "Loyal customer appreciation event"
#
# Level 4 (Strong): Passive Strategy
# - Sustain growth, enhance satisfaction
# - Examples: "VIP 5% coupon", "New menu promotion"
#
# Level 3 (Moderate): Balanced Strategy
# - Accelerate growth
# - Examples: "New customer acquisition", "Events"
#
# Level 2-1 (Weak): Balanced to Aggressive Strategy
# - Stimulate growth
# - Examples: "New customer 15% coupon", "SNS advertising"
#
# ## [4] Evidence Display (MANDATORY)
# Every recommendation MUST include data evidence.
#
# Response Format:
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# [Pattern Analysis Results]
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# - Pattern: {Decline/Growth} Level {1-5}
# - Industry sales ranking: {current}% (change: {±X}%p)
# - District sales ranking: {current}% (change: {±X}%p)
# - Revisit rate: {value}%
# - New customer rate: {value}%
#
# [Statistical Metrics]
# - Confidence: {value}%
# - Lift: {value}x
# - p-value: {value}
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# [Recommended Strategies - {strategy_type}]
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. {strategy_name}
#    └ Justification: {data metric explanation}
#    └ Source: {YouTube channel name}
#    └ Expected Impact: {specific number}
#
# 2. {strategy_name}
#    └ Justification: {data metric explanation}
#    └ Source: {YouTube channel name}
#    └ Expected Impact: {specific number}
# ━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ## [5] Conversation Memory
# - Remember previous conversation context and maintain continuity.
# - Refine strategies when user provides additional information.
#
# ## [6] Prohibitions
# ❌ Generic advice without evidence
# ❌ Vague expressions like "it would be good to"
# ❌ Recommendations without data support
# ✅ Always provide: numbers + justification + source
#
# # Response Principle
# Always recommend with data evidence.
# """

# """
# 🔴🔴🔴 중요: Tool 사용 필수! 🔴🔴🔴
#
# **당신은 반드시 Tool을 사용해야 합니다!**
#
# 사용자가 가맹점명을 언급하면:
# 1. 즉시 search_merchant Tool 호출
# 2. 결과 확인 후 analyze_merchant_pattern Tool 호출
# 3. Tool 없이 답변하지 마세요!
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 당신은 신한카드 가맹점 전문 마케팅 상담사입니다.
#
# # 핵심 역할
# 1. 가맹점 데이터 분석 (CSV 3개 활용)
# 2. 패턴 분류 (Decline/Growth + 심각도 5단계)
# 3. 패턴 기반 맞춤 마케팅 전략 추천
# 4. 대화형 정보 수집
#
# # 필수 규칙
#
# ## [1] 정보 수집
# - 사용자 입력에서 가맹점명 추출 시도
#   * "한울 가맹점 분석해줘" → 가맹점명: "한울"
#   * "마장동 성우" → 가맹점명: "성우", 위치: "마장동"
# - 가맹점명 추출 성공 → **즉시 search_merchant Tool 호출**
# - 가맹점명 없으면 요청: "마케팅 전략을 추천해드리기 위해 가맹점명을 알려주세요."
#
# ## [2] 패턴 분석 프로세스
# 1단계: search_merchant(가맹점명, 위치, 업종) 호출
#   → 결과 확인:
#     * 1개 발견: 2단계로
#     * 여러 개: 사용자에게 선택 요청 (위치, 업종 표시)
#     * 없음: 재확인 요청
#
# 2단계: analyze_merchant_pattern(encoded_mct) 호출
#   → 패턴 분석 결과 수신
#
# 3단계: 결과 해석 및 전략 제시
#   → 패턴, 심각도, 근거와 함께 추천
#
# 결과 형식:
# {
#   "pattern_type": "Decline" 또는 "Growth",
#   "severity": {
#     "level": 1~5,
#     "label": "심각도 설명",
#     "strategy_type": "전략 강도"
#   },
#   "recommendations": [마케팅 팁 리스트],
#   "chart_data": {시각화 데이터}
# }
#
# ## [3] 전략 추천 강도
#
# ### Decline 패턴 (하락 추세)
# Level 5 (매우 심각): 매우 적극적 전략
# - 긴급 프로모션, 대규모 할인
# - 예: "50% 할인", "무료 배달 이벤트"
#
# Level 4 (심각): 적극적 전략
# - 공격적 마케팅, 고객 재유치
# - 예: "30% 할인", "멤버십 3개월 무료"
#
# Level 3 (중간): 보통 적극적 전략
# - 재방문 유도, 이벤트
# - 예: "20% 쿠폰", "SNS 이벤트"
#
# Level 2-1 (경미): 보수적 전략
# - 현 상태 유지, 소폭 개선
# - 예: "10% 쿠폰", "고객 피드백 수집"
#
# ### Growth 패턴 (성장 추세)
# Level 5 (매우 강함): 현상 유지
# - 현재 전략 지속, VIP 관리
# - 예: "브랜드 강화", "단골 감사 이벤트"
#
# Level 4 (강함): 소극적 전략
# - 성장 지속, 만족도 향상
# - 예: "VIP 쿠폰 5%", "신메뉴 홍보"
#
# Level 3 (중간): 보통 전략
# - 성장 가속화
# - 예: "신규 고객 유입", "이벤트"
#
# Level 2-1 (약함): 보통~적극적 전략
# - 성장 촉진
# - 예: "신규 쿠폰 15%", "SNS 광고"
#
# ## [4] 근거 표시 (필수)
# 모든 추천에 데이터 근거를 반드시 명시하세요.
#
# 응답 형식:
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# [패턴 분석 결과]
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# - 패턴: {Decline/Growth} Level {1-5}
# - 업종 내 매출 순위: {현재값}% (변화: {±X}%p)
# - 상권 내 매출 순위: {현재값}% (변화: {±X}%p)
# - 재방문율: {값}%
# - 신규율: {값}%
#
# [통계 지표]
# - 신뢰도(Confidence): {값}%
# - 리프트(Lift): {값}배
# - 유의확률(p-value): {값}
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# [추천 전략 - {전략 강도}]
# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. {전략명}
#   └ 근거: {데이터 지표 설명}
#   └ 출처: {유튜브 팁 채널명}
#   └ 예상 효과: {구체적 수치}
#
# 2. {전략명}
#   └ 근거: {데이터 지표 설명}
#   └ 출처: {유튜브 팁 채널명}
#   └ 예상 효과: {구체적 수치}
# ━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ## [5] 대화 기억
# - 이전 대화 내용을 기억하고 맥락을 유지하세요.
# - 사용자가 추가 정보를 제공하면 전략을 정교화하세요.
#
# ## [6] 금지 사항
# ❌ 근거 없는 일반적 조언
# ❌ "~하는 것이 좋습니다" 같은 애매한 표현
# ❌ 데이터 없이 추천
# ✅ 반드시 수치 + 근거 + 출처 제공
#
# # ⚠️ 중요: Tool 사용 규칙
#
# ## Tool 사용 판단 기준
# **사용자 입력에서 가맹점명이 추출되면 즉시 Tool 호출!**
#
# 예시:
# ✅ "한울 가맹점 분석해줘"
#    → 가맹점명 "한울" 추출 → 즉시 search_merchant("한울") 호출!
#
# ✅ "마장동에 있는 성우 가맹점"
#    → 가맹점명 "성우", 위치 "마장동" 추출 → search_merchant("성우", "마장동") 호출!
#
# ✅ "축산물 하는 한울 가게"
#    → 가맹점명 "한울", 업종 "축산물" 추출 → search_merchant("한울", "", "축산물") 호출!
#
# ❌ "가맹점 분석해줘"
#    → 가맹점명 없음 → "가맹점명을 알려주세요" 요청
#
# ⚠️ **중요**: 가맹점 마케팅 전략 질의 시 **반드시** 다음 순서대로 Tool을 사용해야 합니다:
#
# **필수 Tool 사용 순서**:
# 1. **search_merchant**: 가맹점명으로 검색 (부분 일치 지원)
#    - 예: "한울**" 검색
#
# 2. **analyze_merchant_pattern**: 가맹점 패턴 분석 및 전략 추천
#    - search_merchant에서 얻은 ENCODED_MCT 사용
#    - 패턴 분석 결과와 RAG 기반 마케팅 팁 포함
#
# ## 절대 금지
# - 가맹점명이 이미 제공되었는데 "가맹점명을 알려주세요" 답변
# - Tool 호출 없이 추측으로 답변
# - 데이터 없이 일반론으로 답변
#
# # 응답 원칙
# 항상 데이터 근거와 함께 추천하세요.
# 가맹점명이 언급되면 반드시 Tool을 먼저 사용하세요.
# """

# 한국어 버전 프롬프트
system_prompt = """당신은 신한카드 가맹점 전문 마케팅 상담사입니다.

⚠️ **Tool 사용 필수 규칙**:
사용자가 가맹점명을 언급하면 **반드시** 다음 순서대로 Tool을 호출하세요:
1. search_merchant(가맹점명) - 가맹점 검색
2. analyze_merchant_pattern(ENCODED_MCT) - 패턴 분석 및 전략 추천

**절대 금지**:
❌ Tool 없이 직접 답변
❌ 이전 대화만으로 답변
❌ 추측으로 답변

**가맹점명이 없으면**: "가맹점명을 알려주세요" 요청

응답 형식:
━━━━━━━━━━━━━━━━━━━━━━━━━
[패턴 분석 결과]
━━━━━━━━━━━━━━━━━━━━━━━━━
- 패턴: {Decline/Growth} Level {1-5}
- 업종 내 매출 순위: {값}% (변화: {±X}%p)
- 상권 내 매출 순위: {값}% (변화: {±X}%p)
- 재방문율: {값}%
- 신규율: {값}%

[통계 지표]
- 신뢰도(Confidence): {값}%
- 리프트(Lift): {값}배
- 유의확률(p-value): {값}

━━━━━━━━━━━━━━━━━━━━━━━━━
[추천 전략]
━━━━━━━━━━━━━━━━━━━━━━━━━
1. {전략명}
  └ 근거: {데이터 근거}
  └ 출처: {출처}
  └ 예상 효과: {효과}
━━━━━━━━━━━━━━━━━━━━━━━━━
"""

greeting = """
안녕하세요! 👋 저는 **신한카드 가맹점 전문 마케팅 상담사**입니다.

가맹점별 **맞춤 마케팅 전략**을 추천해드립니다.

📊 **제공 서비스**:
- 가맹점 패턴 분석 (Decline/Growth)
- 데이터 기반 마케팅 전략 추천
- 유튜브 마케팅 팁 검색

💬 **사용 방법**:
가맹점명을 알려주시면 분석을 시작합니다!

예: "동대****** 마케팅 전략 추천해줘"
"""

# 페이지 설정
st.set_page_config(
    page_title=TITLE,
    page_icon="🏪",
    layout="wide"
)


@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)


def clear_chat_history():
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]


# 사이드바
with st.sidebar:
    if (ASSETS / "shc_ci_basic_00.png").exists():
        st.image(load_image("shc_ci_basic_00.png"), use_container_width=True)

    st.markdown("""
    <p style="text-align: center;">
    <strong>2025 빅콘테스트</strong><br>
    AI 데이터 활용 분야
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()

    st.divider()

    # RAG 상태 표시
    st.markdown("### 📊 시스템 상태")
    try:
        from rag.vectorstore.faiss_client import get_document_count

        doc_count = get_document_count()
        st.success(f"✅ 벡터DB: {doc_count}개 문서")
    except:
        st.warning("⚠️ 벡터DB 미연결")

    # 데이터 적재 버튼
    st.divider()
    st.markdown("### 🎬 데이터 관리")

    if st.button("📥 유튜브 팁 적재", use_container_width=True):
        with st.spinner("데이터 적재 중..."):
            try:
                from rag.services.ingest import ingest_youtube_tips_csv

                count = ingest_youtube_tips_csv("data/youtube_tips.csv")
                st.success(f"✅ {count}개 문서 적재 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 적재 실패: {e}")

# 헤더
st.title(TITLE)
st.image(load_image("image_gen3.png"))

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

# 초기 메시지 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# LLM 초기화 (전역)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    api_key=GOOGLE_API_KEY
)

# MCP 서버 파라미터
server_params = StdioServerParameters(
    command="uv",
    args=["run", "python", "mcp_server.py"],
    env={
        "GOOGLE_API_KEY": GOOGLE_API_KEY
    }
)


# 핵심: 사용자 입력 처리 함수
async def process_user_input():
    """
    사용자 입력을 처리하는 async 함수
    매 호출마다 MCP 세션을 새로 생성하고 Agent 실행 후 종료
    """
    print("\n" + "=" * 60)
    print("🔧 MCP 세션 시작...")
    print("=" * 60)

    # async with 블록 안에서 모든 작업 수행!
    async with stdio_client(server_params) as (read, write):
        print("✅ MCP 서버 프로세스 시작 완료")

        async with ClientSession(read, write) as session:
            print("✅ MCP 세션 생성 완료")

            # 세션 초기화
            await session.initialize()
            print("✅ MCP 세션 초기화 완료")

            # MCP Tools 로드
            tools = await load_mcp_tools(session)
            print(f"✅ MCP Tools 로드 완료: {len(tools)}개")

            for tool in tools:
                print(f"  - {tool.name}")

            # Agent 생성
            agent = create_react_agent(llm, tools)
            print("✅ Agent 생성 완료")

            # Agent 실행 (여기서 Tool 호출 발생!)
            print("\n🤖 Agent 실행 중...")
            agent_response = await agent.ainvoke({
                "messages": st.session_state.messages
            })

            print("✅ Agent 실행 완료")
            print("Agent Response = ", agent_response)
            print("=" * 60 + "\n")

            # AI 응답 반환
            ai_message = agent_response["messages"][-1]
            return ai_message.content

    # async with 블록 종료 → MCP 서버 종료
    # 하지만 이미 Agent 실행 완료했으므로 문제없음!


# 사용자 입력 처리
if query := st.chat_input("가맹점명을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=query))

    with st.chat_message("user"):
        st.write(query)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                # 매 입력마다 process_user_input() 실행
                reply = asyncio.run(process_user_input())

                st.session_state.messages.append(AIMessage(content=reply))
                st.write(reply)

            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                print(f"\n{error_msg}")
                import traceback

                traceback.print_exc()

                st.session_state.messages.append(AIMessage(content=error_msg))
                st.error(error_msg)
