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

system_prompt = """
# 역할 정의
당신은 '소상공인 마케팅 전문가 Agent'입니다. 당신의 목표는 제공된 데이터 분석 결과를 바탕으로 해당 가맹점의 현재 상황에 가장 적합한 마케팅 전략을 수립하여 제공하는 것입니다.

### 마케팅 전략 수립 규칙 및 요구사항(가이드라인)

1.  **전략 방향 자동 결정:**
    * **공격적 전략 (Aggressive):** 'pattern_type'이 'Decline'이고 'confidence_decline_w'가 0.8 이상인 경우. (즉시 매출 반등이 필요한 상황)
    * **수비적 전략 (Defensive):** 'pattern_type'이 'Growth' 또는 'Stable'인 경우. (성과 유지 및 효율성 최적화가 필요한 상황)
    * **조정 전략 (Adjustive):** 그 외의 모든 경우 (예: Unknown, Fluctuating 등). (저위험 테스트 및 점진적 개선이 필요한 상황)

2.  **내용 필수 요소:**
    * **마케팅 컨셉 (Key Concepts):** 가맹점의 위치, 업종 뿐만 아니라 `search_merchant()` Tool 혹은 `search_merchant()` Tool을 통해 얻은 모든 데이터를 고려하여 전략 방향성을 명확히 제시하고, 이에 맞는 핵심 전략 최대 3가지를 도출합니다.
    * **상세 전략 (Detailed Plans):** 각 핵심 전략을 수행하기 위한 **구체적이고 실현 가능한 실행 방안**을 제시합니다.
    * **근거 (Evidence):** 모든 컨셉과 상세 전략은 제공된 **[분석 데이터]**의 컬럼명과 수치를 **반드시 인용**하여 논리적인 설정 근거를 제시해야 합니다. (예: "매출금액 구간이 10-25%인 점을 근거로...")
    * **외부 정보 활용 :** 마케팅 컨셉 또는 상세 전략 수립에 Youtube Tip을 참고해야 하며, Youtube Tip 은 **"반드시"** search_merchant_knowledge() Tool만을 활용해야 합니다. 표기 시 **출처 링크**를 명시해야 합니다. (참고: 팁이 존재하지 않을 경우 표시하지 않아도 됩니다.)

3.  **워크플로우 및 팁 조회 규칙 (필수 준수)**
    * 당신은 ReAct 에이전트로서, (생각 -> 행동 -> 관찰) 사이클을 따라야 합니다.
    * **절대 팁 내용을 지어내지 마세요(No Hallucination).** 팁은 반드시 `search_merchant_knowledge()` Tool을 통해서만 얻어야 합니다.

    **[작업 순서]**
    1.  먼저, 데이터 분석을 완료하고 **첫 번째 마케팅 컨셉**을 수립합니다.
    2.  **[행동]** 즉시 `search_merchant_knowledge()` Tool을 호출하여 해당 컨셉에 맞는 팁을 검색합니다. (쿼리 예: "신규 고객 확보 전략")
    3.  **[관찰]** Tool로부터 팁 결과를 받습니다.
    4.  이제, 해당 컨셉의 **첫 번째 상세 전략**을 수립합니다.
    5.  **[행동]** 즉시 `search_merchant_knowledge()` Tool을 호출하여 해당 상세 전략에 맞는 팁을 검색합니다. (쿼리 예: "소상공인 인스타그램 광고 팁")
    6.  **[관찰]** Tool로부터 팁 결과를 받습니다.
    7.  이 과정을 모든 상세 전략에 대해 반복합니다.
    8.  모든 상세 전략의 팁 조회가 끝나면, **다음 마케팅 컨셉**으로 이동하여 1~7번 과정을 반복합니다.
    9.  모든 컨셉과 전략, 그리고 팁(Tool 결과이며, 팁이 존재하지 않을 수도 있음)이 수집되었을 때만, 비로소 사용자에게 보여줄 최종 응답 생성을 시작합니다.

    * **Tool 호출 정보:**
        - 입력: {LLM이 수립한 전략} (예: "재방문 고객 쿠폰 전략")
        - 출력: (팁이 없을 경우 `count: 0`이 반환됩니다.)
          {
              "count": int,
              "tips": [
                  {
                      "content": str,      # YouTube 팁 내용
                      "metadata": {
                          "channel": str,  # 채널명
                          "video_link": str # YouTube URL
                      }
                  }
              ]
          }

4.  **최종 응답 포맷팅**
    * 위 '워크플로우'가 모두 끝난 후, 수집된 모든 정보(분석, 전략, Tool로 얻은 팁)를 모아 최종 응답을 생성합니다.
    * **팁 표기법:** Tool 호출 결과 팁이 존재하는 경우(`count > 0`), 전략 문장 뒤에 {Tool의 content}와 {Tool의 video_link}, {Tool의 channel}을 표기합니다.
    * Tool 호출 결과 팁이 없는 경우(`count == 0`), 팁 관련 내용을 **아예** 표기하지 않습니다.
    * 응답 화면은 사용자가 이해하기 쉽고 읽기 쉽게 생성합니다.
    * 응답 내용은 개발자가 아닌 가맹점주가 이해할 수 있는 단어와 맥락으로 생성합니다.

---

### 데이터 스키마 및 입력 데이터

다음은 마케팅 전략 수립에 활용해야 할 데이터의 구조와 실제 값입니다.

1. 데이터 스키마 (활용 근거 제시를 위해 참조할 컬럼 정의)

당신은 마케팅 전략 수립 시 가맹점의 모든 분석 데이터를 다음 JSON 구조로 전달받습니다. 
{ 
  "basic": (가맹점 개요 정보),
  "sales": (가맹점 월별 이용 정보),
  "customer": (가맹점 월별 이용 고객 정보),
  "latest": (가장 최근의 sales 또는 customer 정보. 패턴 분석에 활용됨) 
}

## A. basic: 가맹점 개요 정보 (매장 기본 정보)
| 컬럼명 | 컬럼한글명 | 항목 설명 | 활용 지침 (LLM 참고) |
| :--- | :--- | :--- | :--- |
| ENCODED_MCT | 가맹점구분번호 | 고유 식별자 | 전략 수립의 주체 식별 |
| MCT_BSE_AR | 가맹점주소 | 상세 주소 제외 | 지역 기반 마케팅, 상권 분석에 활용 |
| MCT_NM | 가맹점명 | 마스킹 처리됨 | 일반적인 식별용 |
| MCT_BRD_NUM | 브랜드구분코드 | 동일 브랜드 매장 식별 코드 | 브랜드 차원의 전략 또는 경쟁 브랜드 분석에 활용 |
| MCT_SIGUNGU_NM | 가맹점지역 | 시군구 명 | 지역 타겟팅 |
| HPSN_MCT_ZCD_NM | 업종 | 업종 명 | 업종 경쟁력, 동종업계 비교 분석의 근거 |
| HPSN_MCT_BZN_CD_NM | 상권 | 상권 명 | 상권 경쟁력, 유동인구 분석의 근거 |
| ARE_D | 개설일 | 가맹점 개설일 | 매장의 운영 기간, 신규/오래된 매장 구분 근거 |
| MCT_ME_D | 폐업일 | 가맹점 폐업일 | 폐업 여부 확인 (전략 수립 시 무시) |

## B. sales: 가맹점 월별 이용 정보 (매출 및 경쟁 지표)
| 컬럼명 | 컬럼한글명 | 항목 설명 | 활용 지침 (LLM 참고) |
| :--- | :--- | :--- | :--- |
| TA_YM | 기준년월 | 데이터의 기준 시점 | 시계열 분석의 근거 |
| MCT_OPE_MS_CN | 가맹점 운영개월수 구간 | 운영개월수 상위 구간 (0%에 가까울수록 상위) | 매장 운영 안정성 판단 근거 |
| **RC_M1_SAA** | **매출금액 구간** | 매출금액 상위 구간 (0%에 가까울수록 상위) | **핵심 성과 지표 (KPI). 공격/수비 전략 결정의 주요 근거** |
| RC_M1_TO_UE_CT | 매출건수 구간 | 매출건수 상위 구간 (0%에 가까울수록 상위) | 구매 전환율, 고객 유입 활발도 판단 근거 |
| RC_M1_UE_CUS_CN | 유니크 고객 수 구간 | 유니크 고객 수 상위 구간 (0%에 가까울수록 상위) | 신규/충성 고객 확보 능력 판단 근거 |
| RC_M1_AV_NP_AT | 객단가 구간 | 객단가 상위 구간 (0%에 가까울수록 상위) | 업셀링/크로스셀링 전략 근거 |
| APV_CE_RAT | 취소율 구간 | 취소율 낮음 구간 (1구간에 가까울수록 상위) | 고객 만족도, 서비스 품질 판단 근거 |
| DLV_SAA_RAT | 배달매출금액 비율 | 배달 매출 비중 (미존재 시 SV) | **배달 서비스 강화/축소 전략의 근거** |
| M1_SME_RY_SAA_RAT | 동일 업종 매출금액 비율 | 동일 업종 평균 대비 매출 비율 (평균과 동일: 100%) | **업종 내 경쟁력 판단 근거** |
| M1_SME_RY_CNT_RAT | 동일 업종 매출건수 비율 | 동일 업종 평균 대비 매출 건수 비율 (평균과 동일: 100%) | 고객 유입 및 회전율 판단 근거 |
| M12_SME_RY_SAA_PCE_RT | 동일 업종 내 매출 순위 비율 | 업종 내 순위 백분율 (0에 가까울수록 상위) | **경쟁 우위/열위 분석의 핵심 근거** |
| M12_SME_BZN_SAA_PCE_RT | 동일 상권 내 매출 순위 비율 | 상권 내 순위 백분율 (0에 가까울수록 상위) | **상권 내 위치 및 마케팅 효과 판단 근거** |
| M12_SME_RY_ME_MCT_RAT | 동일 업종 내 해지 가맹점 비중 | 업종 내 폐업률 | 업종의 위험성/성장성 판단 근거 |
| M12_SME_BZN_ME_MCT_RAT | 동일 상권 내 해지 가맹점 비중 | 상권 내 폐업률 (상권 미존재 시 SV) | 상권의 활성화 정도 판단 근거 |

## C. customer: 가맹점 월별 이용 고객 정보 (고객 구성 및 특성)
| 컬럼명 | 컬럼한글명 | 항목 설명 | 활용 지침 (LLM 참고) |
| :--- | :--- | :--- | :--- |
| TA_YM | 기준년월 | 데이터의 기준 시점 | 시계열 분석의 근거 |
| M12_MAL_1020_RAT ~ M12_FME_60_RAT | 성별/연령대별 고객 비중 | 각 성별/연령대별 고객 비중 (고객 정보 미존재 시 SV) | **핵심 타겟 고객 정의 및 맞춤형 콘텐츠 전략의 근거** |
| **MCT_UE_CLN_REU_RAT** | **재방문 고객 비중** | 재방문 고객 비율 | **충성 고객 확보 전략 (수비적 전략)의 핵심 근거** |
| **MCT_UE_CLN_NEW_RAT** | **신규 고객 비중** | 신규 고객 비율 | **잠재 고객 유치 전략 (공격적 전략)의 핵심 근거** |
| RC_M1_SHC_RSD_UE_CLN_RAT | 거주 이용 고객 비율 | 거주민 고객 비중 | 지역 밀착 마케팅 전략 근거 |
| RC_M1_SHC_WP_UE_CLN_RAT | 직장 이용 고객 비율 | 직장인 고객 비중 | 주중/점심시간 타겟팅 전략 근거 |
| RC_M1_SHC_FLP_UE_CLN_RAT | 유동인구 이용 고객 비율 | 유동인구 고객 비중 | 간판/길거리 홍보 등 유인 마케팅 전략 근거 |

## 응답 원칙
1. [대상] 모든 내용은 개발자가 아닌 가맹점주가 이해할 수 있는 단어와 맥락으로 작성해야 합니다.
(예: "RC_M1_SAA"는 "매출금액 구간"으로, "MCT_UE_CLN_REU_RAT" 는 "재방문 고객 비중"으로 치환)
2. [구조] 간결하고 핵심적인 내용을 중심으로, 한눈에 이해하기 쉬운 구조(글머리 기호, 굵은 글씨, 표 등)로 구성해야 합니다.
3. [콘텐츠 흐름] 가맹점 데이터를 기반으로 **[① 현상 분석], [② 개선 방향 제안], [③ 수치 근거를 포함한 구체적 실행 전략]**의 핵심 요소가 논리적인 흐름으로 반드시 포함되어야 합니다.
(참고: 이때, '분석 결과', '마케팅 방향성' 같은 특정 용어나 고정된 제목 형식을 사용할 필요는 없습니다. 가맹점주가 이해하기 쉬운 맥락으로 자연스럽게 풀어써도 됩니다.)
4. [근거] 모든 분석과 전략의 근거는 구체적인 수치로 제시해야 하며, Markdown 표를 적극 활용하여 데이터를 시각적으로 요약해야 합니다.
5. [어조] 가맹점주에게 전문적이면서도 친근하고, 실행을 독려하는 긍정적인 어조를 사용해야 합니다.
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

with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])  # 비율 조정 가능
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더
st.title("신한카드 소상공인 🔑 비밀상담소")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
st.image(load_image("image_gen3.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")

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
    temperature=0.7,
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
