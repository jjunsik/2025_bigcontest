"""
MCP Server for Merchant Marketing Analysis
- Tool 1: search_merchant - 가맹점 기본 정보 조회
- Tool 2: select_merchant - 여러 검색 결과 중 특정 가맹점 선택
- Tool 3: search_merchant_knowledge - RAG 기반 마케팅 근거 검색
- Tool 4: analyze_merchant_pattern - 패턴 분석 (전략 제공 안 함)
"""
import sys
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastmcp.server import FastMCP

# ============================================
# 전역 변수 및 경로 설정
# ============================================

DATA_DIR = Path("./data")
PATTERN_RULES_PATH = DATA_DIR / "pattern_rules_declclose_v6.json"

# CSV 파일 경로
SET1_PATH = DATA_DIR / "big_data_set1_f.csv"
SET2_PATH = DATA_DIR / "big_data_set2_f.csv"
SET3_PATH = DATA_DIR / "big_data_set3_f.csv"

# 전역 DataFrame
DF_SET1: Optional[pd.DataFrame] = None
DF_SET2: Optional[pd.DataFrame] = None
DF_SET3: Optional[pd.DataFrame] = None
PATTERN_RULES: Optional[List[Dict]] = None

# MCP 서버 초기화
mcp = FastMCP(
    "MerchantMarketingAnalysis",
    instructions="""
    신한카드 가맹점 마케팅 분석 MCP Server

    ## Available Tools

    ### 1. search_merchant
    가맹점명으로 가맹점 검색 (부분 일치 지원)
    - 1개 검색: 즉시 가맹점 정보 반환
    - 여러 개 검색: select_merchant로 선택 필요

    ### 2. select_merchant
    여러 검색 결과 중 특정 가맹점 선택
    - search_merchant에서 result_type="multiple"일 때만 사용
    - 사용자가 "2번 가맹점" 입력 시:
      → select_merchant(index=2, merchant_name="이전 검색어")

    ### 3. analyze_merchant_pattern
    가맹점 패턴 분석 및 상세 컨텍스트 제공
    - 패턴 데이터만 제공 (전략은 LLM이 직접 수립)

    ### 4. search_merchant_knowledge
    RAG 기반 마케팅 사례 검색 (유사도 0.7 이상)
    - LLM이 수립한 전략과 유사한 실제 사례 검색
    - 유사한 내용이 없으면 빈 결과 반환

    ## Tool 관계
    - analyze_merchant_pattern 호출 전 반드시 search_merchant 또는 select_merchant 실행 필요
    - encoded_mct는 search_merchant 결과에서 추출
    - select_merchant: index(번호)와 merchant_name(검색어) 필수

    ## 데이터 소스
    - SET1: 가맹점 기본 정보
    - SET2: 월별 매출/운영 지표
    - SET3: 월별 고객 특성
    - PATTERN_RULES: 패턴 매칭 규칙
    - RAG: 유튜브 마케팅 팁 (FAISS)
    """
)


def debug_log(msg):
    print(msg, file=sys.stderr, flush=True)

# ============================================
# 초기화 함수
# ============================================

def load_all_data() -> bool:
    """전역 DataFrame 로드"""
    global DF_SET1, DF_SET2, DF_SET3, PATTERN_RULES

    debug_log("\n=== 데이터 로딩 시작 ===")

    # SET1 로드
    try:
        if SET1_PATH.exists():
            DF_SET1 = pd.read_csv(SET1_PATH, encoding='cp949')
            debug_log(f"✅ SET1 로드 완료: {len(DF_SET1)} rows")
        else:
            debug_log(f"❌ SET1 파일 없음: {SET1_PATH}")
            DF_SET1 = None
    except Exception as e:
        debug_log(f"❌ SET1 로드 실패: {e}")
        DF_SET1 = None

    # SET2 로드
    try:
        if SET2_PATH.exists():
            DF_SET2 = pd.read_csv(SET2_PATH, encoding='cp949')
            debug_log(f"✅ SET2 로드 완료: {len(DF_SET2)} rows")
        else:
            debug_log(f"❌ SET2 파일 없음: {SET2_PATH}")
            DF_SET2 = None
    except Exception as e:
        debug_log(f"❌ SET2 로드 실패: {e}")
        DF_SET2 = None

    # SET3 로드
    try:
        if SET3_PATH.exists():
            DF_SET3 = pd.read_csv(SET3_PATH, encoding='utf-8')
            debug_log(f"✅ SET3 로드 완료: {len(DF_SET3)} rows")
        else:
            debug_log(f"❌ SET3 파일 없음: {SET3_PATH}")
            DF_SET3 = None
    except Exception as e:
        debug_log(f"❌ SET3 로드 실패: {e}")
        DF_SET3 = None

    # PATTERN_RULES 로드
    try:
        if PATTERN_RULES_PATH.exists():
            with open(PATTERN_RULES_PATH, 'r', encoding='utf-8') as f:
                PATTERN_RULES = json.load(f)
            debug_log(f"✅ PATTERN_RULES 로드 완료: {len(PATTERN_RULES)} rules")
        else:
            debug_log(f"❌ PATTERN_RULES 파일 없음: {PATTERN_RULES_PATH}")
            PATTERN_RULES = None
    except Exception as e:
        debug_log(f"❌ PATTERN_RULES 로드 실패: {e}")
        PATTERN_RULES = None

    debug_log("=== 데이터 로딩 완료 ===\n")

    # 최소 SET1만 있으면 OK
    return DF_SET1 is not None


# ============================================
# 헬퍼 함수
# ============================================

def search_merchants_by_name(partial_name: str, location: str = None, business_type: str = None) -> List[Dict[str, Any]]:
    """
    가맹점명 부분 검색 (위치, 업종 필터링 지원)
    """
    debug_log("search_merchants_by_name 함수 실행")

    if DF_SET1 is None:
        return []

    # 가맹점명 부분 일치
    matched = DF_SET1[DF_SET1['MCT_NM'].str.contains(partial_name, na=False, case=False, regex=False)]

    # 위치 필터
    if location:
        matched = matched[matched['MCT_BSE_AR'].str.contains(location, na=False, case=False, regex=False)]

    # 업종 필터
    if business_type:
        matched = matched[matched['HPSN_MCT_ZCD_NM'].str.contains(business_type, na=False, case=False, regex=False)]

    if matched.empty:
        return []

    # 결과 리스트 생성 (중복 제거)
    results = []
    seen_codes = set()

    for _, row in matched.iterrows():
        encoded_mct = row['ENCODED_MCT']

        if encoded_mct in seen_codes:
            continue

        seen_codes.add(encoded_mct)

        results.append({
            'encoded_mct': encoded_mct,
            'name': row['MCT_NM'],
            'location': row['MCT_BSE_AR'],
            'business_type': row['HPSN_MCT_ZCD_NM'],
            'index': len(results) + 1
        })

    debug_log(f"검색 결과: {len(results)}개")
    return results


def get_merchant_full_data(encoded_mct: str) -> Optional[Dict[str, Any]]:
    """
    ENCODED_MCT로 SET1, SET2, SET3 데이터 통합 조회
    """
    debug_log("get_merchant_full_data 함수 실행")

    if DF_SET1 is None:
        return None

    # SET1: 가맹점 기본 정보
    basic = DF_SET1[DF_SET1["ENCODED_MCT"] == encoded_mct]

    if basic.empty:
        return None

    basic_dict = basic.iloc[0].to_dict()

    # SET2: 매출/운영 지표 (월별)
    sales = []
    if DF_SET2 is not None:
        sales_data = DF_SET2[DF_SET2["ENCODED_MCT"] == encoded_mct]
        if not sales_data.empty:
            sales = sales_data.sort_values("TA_YM").to_dict('records')

    # SET3: 고객 특성 (월별)
    customer = []
    if DF_SET3 is not None:
        customer_data = DF_SET3[DF_SET3["ENCODED_MCT"] == encoded_mct]
        if not customer_data.empty:
            customer = customer_data.sort_values("TA_YM").to_dict('records')

    # 최신 월 데이터 통합
    latest = {}
    if sales:
        latest_sales = sales[-1]
        latest.update(latest_sales)

    if customer:
        latest_customer = customer[-1]
        latest.update(latest_customer)

    return {
        "basic": basic_dict,
        "sales": sales,
        "customer": customer,
        "latest": latest
    }


def calculate_monthly_diff(sales_data: List[Dict]) -> Dict[str, float]:
    """
    월별 데이터에서 최근 2개월 차분 계산
    신규 가맹점(1개월 데이터): 첫 달 값을 diff로 사용
    """
    # 차분 계산할 변수들
    diff_vars = [
        "M12_SME_RY_SAA_PCE_RT",
        "M12_SME_BZN_SAA_PCE_RT",
        "M1_SME_RY_SAA_RAT",
        "M12_SME_RY_ME_MCT_RAT",
        "M12_SME_BZN_ME_MCT_RAT",
        "DLV_SAA_RAT",
        "APV_CE_RAT"
    ]

    if len(sales_data) == 0:
        return {}

    # 신규 가맹점 (1개월 데이터만)
    if len(sales_data) == 1:
        first_month = sales_data[0]
        result = {}

        for var in diff_vars:
            first_val = first_month.get(var, 0)

            try:
                # 기존 로직과 동일: float 캐스팅 + NaN 체크
                first_val = float(first_val) if pd.notna(first_val) else 0.0
                result[f"{var}_diff"] = first_val  # 첫 달 값 = diff
            except:
                result[f"{var}_diff"] = 0.0

        return result

    # 기존 로직 (2개월 이상): 변경 없음
    latest = sales_data[-1]
    prev = sales_data[-2]

    result = {}
    for var in diff_vars:
        latest_val = latest.get(var, 0)
        prev_val = prev.get(var, 0)

        try:
            latest_val = float(latest_val) if pd.notna(latest_val) else 0.0
            prev_val = float(prev_val) if pd.notna(prev_val) else 0.0
            result[f"{var}_diff"] = latest_val - prev_val
        except:
            result[f"{var}_diff"] = 0.0

    return result


def calculate_severity(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    패턴의 심각도 계산 (5단계)
    """
    debug_log("calculate_severity 함수 실행")

    pattern_type = pattern.get("pattern_type")
    metrics = pattern.get("metrics", {})

    lift = metrics.get("lift_vs_baseline_decline_w", 1.0)
    confidence = metrics.get("confidence_decline_w", 0.5)

    if pattern_type == "Decline":
        if lift > 1.5 and confidence > 0.9:
            return {
                "level": 5,
                "label": "매우 심각한 하락",
                "strategy_type": "매우 적극적"
            }
        elif lift > 1.3 and confidence > 0.8:
            return {
                "level": 4,
                "label": "심각한 하락",
                "strategy_type": "적극적"
            }
        elif lift > 1.15 and confidence > 0.7:
            return {
                "level": 3,
                "label": "중간 수준 하락",
                "strategy_type": "보통 적극적"
            }
        elif lift > 1.05 and confidence > 0.6:
            return {
                "level": 2,
                "label": "경미한 하락",
                "strategy_type": "보수적"
            }
        else:
            return {
                "level": 1,
                "label": "약한 하락 징후",
                "strategy_type": "보수적"
            }

    elif pattern_type == "Growth":
        if lift < 0.5 and confidence > 0.9:
            return {
                "level": 5,
                "label": "매우 강한 성장",
                "strategy_type": "현상 유지"
            }
        elif lift < 0.7 and confidence > 0.8:
            return {
                "level": 4,
                "label": "강한 성장",
                "strategy_type": "소극적"
            }
        elif lift < 0.85 and confidence > 0.7:
            return {
                "level": 3,
                "label": "중간 수준 성장",
                "strategy_type": "보통"
            }
        elif lift < 0.95 and confidence > 0.6:
            return {
                "level": 2,
                "label": "약한 성장",
                "strategy_type": "보통"
            }
        else:
            return {
                "level": 1,
                "label": "성장 가능성",
                "strategy_type": "보통~적극적"
            }

    return {"level": 0, "label": "판정 불가", "strategy_type": "보통"}


def match_pattern_rules(merchant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    가맹점 데이터와 패턴 규칙 매칭
    """
    debug_log("match_pattern_rules 함수 실행")

    if PATTERN_RULES is None:
        return []

    sales = merchant_data.get("sales", [])

    # 월별 변화량 계산
    diff_data = calculate_monthly_diff(sales)

    if not diff_data:
        return []

    matched = []

    for rule in PATTERN_RULES:
        condition = rule.get("condition", {})

        # 모든 조건 체크
        all_match = True
        for var_name, direction in condition.items():
            var_diff = diff_data.get(f"{var_name}_diff", 0)

            # 방향 체크
            if direction == "down" and var_diff >= 0:
                all_match = False
                break
            elif direction == "up" and var_diff <= 0:
                all_match = False
                break

        if all_match:
            matched.append(rule)

    # confidence 순 정렬
    matched.sort(
        key=lambda x: x.get("metrics", {}).get("confidence_decline_w", 0),
        reverse=True
    )

    debug_log(f"매칭된 패턴: {len(matched)}개")
    return matched


# ============================================
# RAG 검색 내부 함수
# ============================================
def _search_rag_internal(
        query: str,
        similarity_threshold: float = 0.7,
        fetch_k: int = 10
) -> Dict[str, Any]:
    """
    RAG 검색 로직 (내부 함수)

    검색 방식:
    - query(LLM 전략)를 임베딩
    - RAG의 content(순수 마케팅 내용)와 코사인 유사도 계산
    - similarity_threshold 이상인 문서만 반환

    Args:
        query: 검색 쿼리 (LLM이 수립한 전략)
        similarity_threshold: 유사도 임계값 (0~1)
        fetch_k: 최대 검색 개수

    Returns:
        검색 결과 (count, tips, context 포함)
    """
    try:
        from rag.services.search import search_context

        debug_log(f"  🔍 RAG 검색: '{query}'")
        debug_log(f"     threshold={similarity_threshold}, fetch_k={fetch_k}")

        # 명시적 파라미터 전달
        context, docs = search_context(
            query=query,
            similarity_threshold=similarity_threshold,
            fetch_k=fetch_k
        )

        tips = []
        for doc in docs:
            tips.append({
                "content": doc.page_content,  # content만
                "metadata": doc.metadata  # channel, title, video_link
            })

        debug_log(f"  ✅ {len(docs)}개 문서 검색 완료")

        return {
            "count": len(docs),
            "tips": tips,
            "context": context
        }

    except Exception as e:
        debug_log(f"  ❌ RAG 검색 실패: {e}")
        import traceback
        traceback.print_exc()

        return {
            "count": 0,
            "tips": [],
            "context": "",
            "error": str(e)
        }


# ============================================
# Tool 1: search_merchant
# ============================================
@mcp.tool()
def search_merchant(merchant_name: str, location: str = "", business_type: str = "") -> Dict[str, Any]:
    """
    가맹점명으로 가맹점 검색 (부분 일치)

    ## 목적
    사용자가 제공한 가맹점명으로 ENCODED_MCT 코드를 찾습니다.

    ## 데이터 소스
    - SET1 (big_data_set1_f.csv): 가맹점 기본 정보

    ## 검색 방식
    - 가맹점명: 부분 일치 검색
      - 입력 예시: "한울" 입력 시 "한울**", "한울****" 등 검색
      - 데이터는 마스킹 처리되어 있음 (예: "한울**", "은지*", "동대******")
    - 위치 필터 (선택): "서울 성동구", "마장동" 등
    - 업종 필터 (선택): "축산물", "한식" 등

    ## 반환 타입별 처리
    ### result_type="single" (1개 검색)
    - data: Dict 타입
    - 포함: encoded_mct, name, location, business_type, latest_data

    ### result_type="multiple" (여러 개 검색)
    - data: List 타입
    - 각 항목: encoded_mct, name, location, business_type
    - index는 1부터 시작 (예: 1, 2, 3, ...)

    ### result_type="not_found" (검색 결과 없음)

    Args:
        merchant_name (str): 가맹점명 또는 일부 (필수)
        location (str): 위치 필터 (선택)
        business_type (str): 업종 필터 (선택)

    Returns:
        Dict[str, Any]: {
            "found": bool,
            "result_type": "single" | "multiple" | "not_found",
            "data": Dict | List,
            "count": int,
            "message": str
        }

    Examples:
        search_merchant("한울")  # "한울**", "한울****" 등 검색
        search_merchant("은지")  # "은지*", "은지**" 등 검색
    """
    debug_log(f"\n🔍 search_merchant 호출: '{merchant_name}', 위치='{location}', 업종='{business_type}'")

    if DF_SET1 is None:
        return {
            "found": False,
            "result_type": "error",
            "message": "데이터가 로드되지 않았습니다."
        }

    # 가맹점 검색
    search_results = search_merchants_by_name(
        merchant_name,
        location if location else None,
        business_type if business_type else None
    )

    if len(search_results) == 0:
        return {
            "found": False,
            "result_type": "not_found",
            "merchant_name": merchant_name,
            "message": f"'{merchant_name}' 가맹점을 찾을 수 없습니다."
        }

    elif len(search_results) == 1:
        result = search_results[0]
        encoded_mct = result['encoded_mct']
        merchant_data = get_merchant_full_data(encoded_mct)

        if merchant_data is None:
            return {
                "found": False,
                "result_type": "error",
                "message": "가맹점 데이터 조회 중 오류가 발생했습니다."
            }

        basic = merchant_data["basic"]
        latest = merchant_data["latest"]

        return {
            "found": True,
            "result_type": "single",
            "data": {
                "encoded_mct": encoded_mct,
                "name": basic.get("MCT_NM"),
                "location": basic.get("MCT_BSE_AR"),
                "business_type": basic.get("HPSN_MCT_ZCD_NM"),
                "business_detail": basic.get("HPSN_MCT_BZN_CD_NM"),
                "open_date": basic.get("ARE_D"),
                "latest_data": latest
            },
            "message": f"'{basic.get('MCT_NM')}' 가맹점 정보를 조회했습니다."
        }

    else:
        return {
            "found": True,
            "result_type": "multiple",
            "merchant_name": merchant_name,
            "count": len(search_results),
            "data": search_results,
            "message": f"'{merchant_name}'으로 {len(search_results)}개의 가맹점이 검색되었습니다."
        }


# ============================================
# Tool 2: select_merchant
# ============================================
@mcp.tool()
def select_merchant(index: int, merchant_name: str) -> Dict[str, Any]:
    """
    여러 검색 결과 중 특정 가맹점 선택

    ## 사용 시점
    search_merchant에서 result_type="multiple"일 때만 호출

    ## 전제조건
    - search_merchant 실행 완료
    - 사용자가 번호로 가맹점 선택 (예: "2번 가맹점")

    ## 프로세스
    1. merchant_name으로 다시 검색
    2. index번째 가맹점의 encoded_mct 추출 (1부터 시작)
    3. 해당 가맹점의 상세 정보 반환

    ## 데이터 소스
    - SET1: 가맹점 기본 정보
    - SET2: 월별 매출/운영 지표 (latest_data)
    - SET3: 월별 고객 특성 (latest_data)

    Args:
        index (int): 검색 결과의 순번 (1부터 시작, 필수)
        merchant_name (str): 이전 검색 쿼리 (필수, 예: "마하")

    Returns:
        Dict[str, Any]: {
            "found": bool,
            "data": {
                "encoded_mct": str,
                "name": str,
                "location": str,
                "business_type": str,
                "business_detail": str,
                "open_date": str,
                "latest_data": Dict
            },
            "message": str
        }

    Example:
        사용자: "2번 가맹점"
        → select_merchant(index=2, merchant_name="마하")

    Note:
        이 Tool 실행 후 analyze_merchant_pattern(encoded_mct) 호출 가능
    """
    debug_log(f"select_merchant 호출: index={index}, merchant_name={merchant_name}")

    if DF_SET1 is None:
        return {
            "found": False,
            "message": "데이터가 로드되지 않았습니다."
        }

    # 입력 검증
    if index <= 0:
        return {
            "found": False,
            "message": f"index는 1 이상이어야 합니다. (입력값: {index})"
        }

    if not merchant_name:
        return {
            "found": False,
            "message": "merchant_name이 필요합니다."
        }

    # merchant_name으로 다시 검색
    results = search_merchants_by_name(merchant_name)

    if not results:
        return {
            "found": False,
            "message": f"'{merchant_name}' 가맹점을 찾을 수 없습니다."
        }

    if index > len(results):
        return {
            "found": False,
            "message": f"index={index}는 범위를 벗어났습니다. (검색 결과: {len(results)}개)"
        }

    # index-1 (1부터 시작 → 0부터 시작)
    encoded_mct = results[index - 1]['encoded_mct']
    debug_log(f"✅ index={index} → encoded_mct={encoded_mct}")

    # 가맹점 데이터 조회
    merchant_data = get_merchant_full_data(encoded_mct)

    if merchant_data is None:
        return {
            "found": False,
            "encoded_mct": encoded_mct,
            "message": f"가맹점 코드 '{encoded_mct}'를 찾을 수 없습니다."
        }

    basic = merchant_data["basic"]
    latest = merchant_data["latest"]

    return {
        "found": True,
        "data": {
            "encoded_mct": encoded_mct,
            "name": basic.get("MCT_NM"),
            "location": basic.get("MCT_BSE_AR"),
            "business_type": basic.get("HPSN_MCT_ZCD_NM"),
            "business_detail": basic.get("HPSN_MCT_BZN_CD_NM"),
            "open_date": basic.get("ARE_D"),
            "latest_data": latest
        },
        "message": f"'{basic.get('MCT_NM')}' 가맹점을 선택했습니다."
    }


# ============================================
# Tool 3: search_merchant_knowledge
# ============================================
@mcp.tool()
def search_merchant_knowledge(query: str) -> Dict[str, Any]:
    """
    RAG 기반 마케팅 사례 검색 (코사인 유사도 0.7 이상)

    ## 목적
    LLM이 수립한 마케팅 전략과 유사한 실제 사례를 RAG에서 검색합니다.
    유사한 내용이 없으면 빈 결과를 반환합니다.

    ## 사용 시점
    LLM이 마케팅 전략을 작성한 후

    ## 데이터 소스
    - RAG: 유튜브 마케팅 팁 (FAISS)
      - 임베딩 모델: Google Gemini embedding-001
      - 유사도 임계값: 0.7

    ## 검색 방식
    1. query를 임베딩
    2. FAISS와 코사인 유사도 계산
    3. similarity_threshold=0.7 이상만 반환
    4. 유사한 내용이 없으면 count=0 반환

    Args:
        query (str): 검색할 마케팅 전략 또는 키워드

    Returns:
        Dict[str, Any]: {
            "found": bool,
            "count": int,  # 0일 수 있음
            "tips": [
                {
                    "content": str,
                    "metadata": {
                        "channel": str,
                        "title": str,
                        "video_link": str
                    }
                }
            ],
            "context": str  # count=0이면 빈 문자열
        }

    Note:
        유사한 사례가 없어도 정상 동작입니다. (count=0)
    """

    debug_log(f"\n🔍 search_merchant_knowledge 호출: '{query}'")

    # 명시적 파라미터 전달
    result = _search_rag_internal(
        query=query,
        similarity_threshold=0.7,  # 유사도 임계값
        fetch_k=10  # 최대 후보군
    )

    debug_log(f"✅ RAG 검색 완료: {result.get('count', 0)}개\n")

    return result


# ============================================
# Tool 4: analyze_merchant_pattern
# ============================================
@mcp.tool()
def analyze_merchant_pattern(encoded_mct: str) -> Dict[str, Any]:
    """
    가맹점 패턴 분석 및 상세 컨텍스트 제공

    ## 목적
    가맹점의 Decline/Growth 패턴을 식별하고 merchant_context를 제공합니다.

    ## 전제조건
    encoded_mct 필요 → search_merchant 또는 select_merchant 먼저 실행

    ## 데이터 소스
    - SET1: 가맹점 기본 정보
    - SET2: 월별 매출/운영 지표
    - SET3: 월별 고객 특성
    - PATTERN_RULES: 패턴 매칭 규칙

    ## 분석 프로세스
    1. SET1/SET2/SET3 조회
    2. 최근 2개월 차분 계산
    3. PATTERN_RULES와 매칭
    4. confidence 순 정렬
    5. 심각도 계산 (level 1~5)
    6. merchant_context 생성

    ## 제공 정보
    ### pattern (매칭된 패턴)
    - pattern_id, pattern_type (Decline/Growth)
    - condition: 패턴 조건
    - metrics: confidence, lift, support, p_value

    ### severity (심각도)
    - level: 1~5
    - label: "매우 심각한 하락", "강한 성장" 등
    - strategy_type: "매우 적극적", "보통", "현상 유지"

    ### merchant_context (전략 수립용)
    - name, location, business_type, open_date
    - latest_metrics: revisit_rate, new_customer_rate, monthly_sales_change, delivery_sales_ratio

    Args:
        encoded_mct (str): 가맹점 코드

    Returns:
        Dict[str, Any]: {
            "found": bool,
            "encoded_mct": str,
            "pattern": Dict or None,
            "severity": Dict or None,
            "merchant_context": Dict,
            "all_matched_patterns": List[str],
            "message": str
        }
    """
    debug_log("analyze_merchant_pattern Tool 호출 (패턴 분석만)")

    # 가맹점 데이터 조회
    merchant_data = get_merchant_full_data(encoded_mct)
    if merchant_data is None:
        return {
            "found": False,
            "encoded_mct": encoded_mct,
            "message": f"가맹점 코드 '{encoded_mct}'를 찾을 수 없습니다."
        }

    # 패턴 매칭
    matched_patterns = match_pattern_rules(merchant_data)

    # 가맹점 컨텍스트 생성 (LLM이 전략 수립 시 참고)
    basic = merchant_data.get("basic", {})
    latest = merchant_data.get("latest", {})
    sales = merchant_data.get("sales", [])
    diff_data = calculate_monthly_diff(sales)

    merchant_context = {
        "name": basic.get("MCT_NM"),
        "location": basic.get("MCT_BSE_AR"),
        "business_type": basic.get("HPSN_MCT_ZCD_NM"),
        "business_detail": basic.get("HPSN_MCT_BZN_CD_NM"),
        "open_date": basic.get("ARE_D"),
        "latest_metrics": {
            "revisit_rate": latest.get("MCT_UE_CLN_REU_RAT", 0),
            "new_customer_rate": latest.get("MCT_UE_CLN_NEW_RAT", 0),
            "monthly_sales_change": diff_data.get("M12_SME_RY_SAA_PCE_RT_diff", 0),
            "delivery_sales_ratio": latest.get("DLV_SAA_RAT", 0),
            "approval_count_ratio": latest.get("APV_CE_RAT", 0)
        }
    }

    if not matched_patterns:
        return {
            "found": True,
            "encoded_mct": encoded_mct,
            "pattern": None,
            "severity": None,
            "merchant_context": merchant_context,
            "message": "매칭되는 패턴을 찾을 수 없습니다. 가맹점 데이터를 참고하여 전략을 수립하세요."
        }

    # 최우선 패턴 선택
    best_pattern = matched_patterns[0]

    # 심각도 계산
    severity = calculate_severity(best_pattern)

    debug_log("analyze_merchant_pattern Tool 호출 종료")

    return {
        "found": True,
        "encoded_mct": encoded_mct,
        "pattern": {
            "pattern_id": best_pattern.get("pattern_id"),
            "pattern_type": best_pattern.get("pattern_type"),
            "condition": best_pattern.get("condition"),
            "metrics": best_pattern.get("metrics")
        },
        "severity": severity,
        "merchant_context": merchant_context,
        "all_matched_patterns": [p.get("pattern_id") for p in matched_patterns[:3]],
        "message": f"{severity['label']} 패턴이 감지되었습니다. 이 데이터를 바탕으로 {severity['strategy_type']} 마케팅 전략을 수립하세요."
    }


# ============================================
# 서버 실행
# ============================================

# ============================================
# 서버 실행
# ============================================

if __name__ == "__main__":
    # 데이터 로드
    if not load_all_data():
        debug_log("\n" + "=" * 50)
        debug_log("❌ 데이터 로딩 실패! 서버를 시작할 수 없습니다.")
        debug_log("최소 SET1 파일이 필요합니다.")
        debug_log("=" * 50 + "\n")
        sys.exit(1)

    debug_log("\n" + "=" * 50)
    debug_log("🚀 MCP Server 시작")
    debug_log("=" * 50 + "\n")

    mcp.run()
