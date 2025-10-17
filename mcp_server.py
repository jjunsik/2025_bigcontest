"""
MCP Server for Merchant Marketing Analysis
- Tool 1: search_merchant - 가맹점 기본 정보 조회
- Tool 2: search_merchant_knowledge - RAG 기반 마케팅 팁 검색
- Tool 3: analyze_merchant_pattern - 패턴 분석 및 전략 추천
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
신한카드 가맹점 마케팅 분석 서버입니다.

제공 Tool:
1. search_merchant - 가맹점 검색 (가맹점명, 위치, 업종으로 검색)
2. select_merchant - 여러 검색 결과 중 특정 가맹점 선택
3. search_merchant_knowledge - RAG 기반 마케팅 팁 검색
4. analyze_merchant_pattern - 가맹점 패턴 분석 및 마케팅 전략 추천

사용 흐름:
1. 사용자가 가맹점명(필수) + 위치/업종(선택)을 입력
2. search_merchant 호출 → 여러 결과가 나오면 사용자에게 선택 요청
3. 사용자가 선택 → select_merchant 호출 또는 analyze_merchant_pattern 직접 호출
4. 분석 결과 제공

주의사항:
- 사용자는 ENCODED_MCT를 모릅니다. 절대 사용자에게 ENCODED_MCT 입력을 요구하지 마세요.
- 여러 가맹점이 검색되면, 위치와 업종을 포함한 목록을 보여주고 "1번" 또는 "서울 성동구에 있는 거" 같은 방식으로 선택받으세요.
- 선택 후에는 해당 가맹점의 ENCODED_MCT를 사용하여 분석을 진행하세요.
"""
)

def debug_log(msg):
    print(msg, file=sys.stderr, flush=True)

# ============================================
# 초기화 함수
# ============================================

def load_all_data():
    """CSV 파일 및 패턴 규칙 로드"""
    global DF_SET1, DF_SET2, DF_SET3, PATTERN_RULES

    debug_log("=" * 50)
    debug_log("데이터 로딩 시작...")
    debug_log("=" * 50)

    # CSV 로드 (인코딩 수정) ✅
    try:
        if SET1_PATH.exists():
            DF_SET1 = pd.read_csv(SET1_PATH, encoding='cp949')  # ✅ cp949로 변경
            debug_log(f"✅ SET1 로드 완료: {len(DF_SET1)} rows")
        else:
            debug_log(f"⚠️ SET1 파일 없음: {SET1_PATH}")
    except Exception as e:
        debug_log(f"❌ SET1 로드 실패: {e}")

    try:
        if SET2_PATH.exists():
            DF_SET2 = pd.read_csv(SET2_PATH, encoding='cp949')  # ✅ cp949로 변경
            debug_log(f"✅ SET2 로드 완료: {len(DF_SET2)} rows")
        else:
            debug_log(f"⚠️ SET2 파일 없음: {SET2_PATH}")
    except Exception as e:
        debug_log(f"❌ SET2 로드 실패: {e}")

    try:
        if SET3_PATH.exists():
            DF_SET3 = pd.read_csv(SET3_PATH, encoding='utf-8')  # ✅ utf-8 유지
            debug_log(f"✅ SET3 로드 완료: {len(DF_SET3)} rows")
        else:
            debug_log(f"⚠️ SET3 파일 없음: {SET3_PATH}")
    except Exception as e:
        debug_log(f"❌ SET3 로드 실패: {e}")

    # 패턴 규칙 로드 (수정 불필요)
    try:
        if PATTERN_RULES_PATH.exists():
            with open(PATTERN_RULES_PATH, 'r', encoding='utf-8') as f:
                PATTERN_RULES = json.load(f)
            debug_log(f"✅ 패턴 규칙 로드 완료: {len(PATTERN_RULES)} patterns")
        else:
            debug_log(f"⚠️ 패턴 규칙 파일 없음: {PATTERN_RULES_PATH}")
    except Exception as e:
        debug_log(f"❌ 패턴 규칙 로드 실패: {e}")

    debug_log("=" * 50)
    return DF_SET1 is not None


# ============================================
# 헬퍼 함수
# ============================================

def search_merchants_by_name(partial_name: str, location: str = None, business_type: str = None) -> List[
    Dict[str, Any]]:
    """
    가맹점명 부분 검색 (위치, 업종 필터링 지원)

    Args:
        partial_name: 가맹점명 일부
        location: 위치 필터 (선택사항)
        business_type: 업종 필터 (선택사항)

    Returns:
        매칭된 가맹점 리스트
    """
    debug_log("search_merchants_by_name 함수 실행하여 질의를 통해 가맹점명 찾기 시작")

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
            'index': len(results) + 1  # 선택용 번호
        })

    debug_log("search_merchants_by_name 함수 실행하여 질의를 통해 가맹점명 찾기 종료")

    return results


# ============================================
# 헬퍼 함수
# ============================================
def get_merchant_full_data(encoded_mct: str) -> Optional[Dict[str, Any]]:
    """
    ENCODED_MCT로 SET1, SET2, SET3 데이터 통합 조회

    Args:
        encoded_mct: 가맹점 고유 코드 (ENCODED_MCT)

    Returns:
        {
            "basic": {...},
            "sales": [{...}],
            "customer": [{...}],
            "latest": {...}
        }
    """
    debug_log("get_merchant_full_data 함수 실행하여 가맹점 정보 찾기 시작")

    if DF_SET1 is None:
        return None

    # SET1: 가맹점 기본 정보
    basic = DF_SET1[DF_SET1["ENCODED_MCT"] == encoded_mct]  # ✅ 수정

    if basic.empty:
        return None

    basic_dict = basic.iloc[0].to_dict()

    # SET2: 매출/운영 지표 (월별)
    sales = []
    if DF_SET2 is not None:
        sales_data = DF_SET2[DF_SET2["ENCODED_MCT"] == encoded_mct]  # ✅ 수정
        if not sales_data.empty:
            sales = sales_data.sort_values("TA_YM").to_dict('records')

    # SET3: 고객 특성 (월별)
    customer = []
    if DF_SET3 is not None:
        customer_data = DF_SET3[DF_SET3["ENCODED_MCT"] == encoded_mct]  # ✅ 수정
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

    debug_log("get_merchant_full_data 함수 실행하여 가맹점 정보 찾기 종료")

    return {
        "basic": basic_dict,
        "sales": sales,
        "customer": customer,
        "latest": latest
    }


def calculate_monthly_diff(sales_data: List[Dict]) -> Dict[str, float]:
    """
    월별 데이터에서 최근 2개월 차분 계산

    Returns:
        {
            "M12_SME_RY_SAA_PCE_RT_diff": float,
            "M12_SME_BZN_SAA_PCE_RT_diff": float,
            ...
        }
    """
    debug_log("calculate_monthly_diff 함수 실행하여 2개월 동안 매출 정보 찾기 시작")

    if len(sales_data) < 2:
        return {}

    # 최신 2개월
    latest = sales_data[-1]
    prev = sales_data[-2]

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

    debug_log("calculate_monthly_diff 함수 실행하여 2개월 동안 매출 정보 찾기 종료")

    return result


def calculate_severity(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    패턴의 심각도 계산 (5단계)

    Returns:
        {
            "level": 1~5,
            "label": "매우 심각한 하락" | "강한 성장" 등,
            "strategy_type": "매우 적극적" | "현상 유지" 등
        }
    """
    debug_log("calculate_severity 함수 실행하여 패턴의 심각도 계산 시작")

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

    debug_log("calculate_severity 함수 실행하여 패턴의 심각도 계산 완료")

    return {"level": 0, "label": "판정 불가", "strategy_type": "보통"}


def match_pattern_rules(merchant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    가맹점 데이터와 패턴 규칙 매칭

    Returns:
        매칭된 패턴 리스트 (confidence 순 정렬)
    """
    debug_log("match_pattern_rules 함수 실행하여 가맹점에 해당하는 패턴 찾기 시작")

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

    debug_log("match_pattern_rules 함수 실행하여 가맹점에 해당하는 패턴 찾기 종료")

    return matched


def get_strategy_keywords(severity: Dict[str, Any], merchant_data: Dict[str, Any]) -> List[str]:
    """가맹점 특성에 맞는 RAG 검색 키워드 생성"""
    debug_log("get_strategy_keywords 함수 실행하여 심각도에 따른 마케팅 전략 키워드 생성 시작")

    keywords = []

    # 1. 패턴 기반 키워드
    pattern_label = severity.get("label", "")
    level = severity.get("level", 0)

    if "하락" in pattern_label:
        keywords.append("매출 감소 대응")
        if level >= 4:
            keywords.append("위기 극복 전략")
            keywords.append("긴급 개선 방안")
        elif level >= 3:
            keywords.append("매출 회복 방법")
            keywords.append("고객 이탈 방지")
        else:
            keywords.append("안정화 전략")
    elif "상승" in pattern_label:
        keywords.append("성장 전략")
        if level >= 4:
            keywords.append("급성장 유지 방법")
            keywords.append("빠른 확장 전략")
        elif level >= 3:
            keywords.append("매출 증대 방법")
            keywords.append("성장 가속화")
        else:
            keywords.append("안정적 성장")

    # 2. 업종 기반 키워드
    basic = merchant_data.get("basic", {})
    business_type = basic.get("HPSNMCTBZNCDNM", "")
    if business_type:
        keywords.append(f"{business_type} 마케팅")

    # 3. 지표 기반 키워드
    latest = merchant_data.get("latest", {})
    revisit_rate = latest.get("MCTUECLNREURAT", 0)
    new_rate = latest.get("MCTUECLNNEWRAT", 0)

    try:
        revisit_rate = float(revisit_rate) if revisit_rate else 0.0
        new_rate = float(new_rate) if new_rate else 0.0

        if revisit_rate < 30:
            keywords.append("재방문율 높이는 방법")
        if new_rate > 60:
            keywords.append("신규 고객 단골 만들기")
    except:
        pass

    # 4. 매출 추이 기반 키워드
    monthly_diff = severity.get("monthly_diff", {})
    monthly_sales_change = monthly_diff.get("M12SMERYSAAPCERT_diff", 0)

    try:
        monthly_sales_change = float(monthly_sales_change) if monthly_sales_change else 0.0

        if monthly_sales_change < -10:
            keywords.append("급격한 매출 감소 대응")
        elif monthly_sales_change > 10:
            keywords.append("성장 모멘텀 유지")
    except:
        pass

    # 중복 제거
    keywords = list(dict.fromkeys(keywords))

    debug_log(f"생성된 키워드: {keywords}")
    debug_log("get_strategy_keywords 함수 실행하여 심각도에 따른 마케팅 전략 키워드 생성 종료")

    return keywords if keywords else ["마케팅 전략", "고객 관리"]


# ============================================
# Tool 1: search_merchant (완전 수정)
# ============================================

@mcp.tool()
def search_merchant(merchant_name: str, location: str = "", business_type: str = "") -> Dict[str, Any]:
    """
    가맹점 기본 정보 조회 - 가맹점명으로 검색

    ⚠️ 사용자가 가맹점명을 언급하면 즉시 이 Tool을 호출하세요!

    이 Tool은 가맹점명으로 부분 일치 검색하여 가맹점 정보를 조회합니다.
    여러 가맹점이 검색되면 사용자에게 선택을 요청할 수 있도록 목록을 반환합니다.

    Args:
        merchant_name (str): 가맹점명 (필수)
            - 부분 일치 검색 지원
            - 예: "한울", "성우", "대보"
        location (str): 위치 필터 (선택)
            - 예: "서울 성동구", "마장동"
        business_type (str): 업종 필터 (선택)
            - 예: "축산물", "한식", "카페"

    Returns:
        Dict[str, Any]: 검색 결과

    사용 예시:
        User: "한울 가맹점 분석해줘" → search_merchant("한울")
        User: "마장동 성우 가맹점" → search_merchant("성우", "마장동")
    """

    # 디버깅 로그
    debug_log(f"\n🔍 search_merchant 호출: '{merchant_name}', 위치='{location}', 업종='{business_type}'")

    if DF_SET1 is None:
        debug_log("❌ DF_SET1이 None입니다!")
        return {
            "found": False,
            "result_type": "error",
            "message": "데이터가 로드되지 않았습니다."
        }

    debug_log(f"✅ DF_SET1 로드됨: {len(DF_SET1)} rows")

    # 가맹점 검색 (위치, 업종 필터 적용)
    search_results = search_merchants_by_name(
        merchant_name,
        location if location else None,
        business_type if business_type else None
    )

    debug_log(f"🔍 검색 결과: {len(search_results)}개")

    if len(search_results) == 0:
        debug_log(f"❌ '{merchant_name}' 검색 결과 없음")
        return {
            "found": False,
            "result_type": "not_found",
            "merchant_name": merchant_name,
            "location": location,
            "business_type": business_type,
            "message": f"'{merchant_name}' 가맹점을 찾을 수 없습니다. 가맹점명을 다시 확인해주세요."
        }

    elif len(search_results) == 1:
        # 검색 결과 1개 → 바로 데이터 조회
        result = search_results[0]
        encoded_mct = result['encoded_mct']

        debug_log(f"✅ 1개 결과 찾음: {encoded_mct}")

        merchant_data = get_merchant_full_data(encoded_mct)

        if merchant_data is None:
            debug_log(f"❌ get_merchant_full_data 실패: {encoded_mct}")
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
        # 검색 결과 여러 개 → 사용자에게 선택 요청
        debug_log(f"✅ {len(search_results)}개 결과 찾음")

        return {
            "found": True,
            "result_type": "multiple",
            "merchant_name": merchant_name,
            "count": len(search_results),
            "data": search_results,
            "message": f"'{merchant_name}'으로 {len(search_results)}개의 가맹점이 검색되었습니다. 아래 목록에서 원하시는 가맹점을 선택해주세요."
        }


def _search_rag_internal(query: str, k: int = 3) -> Dict[str, Any]:
    """
    RAG 검색 로직 (내부 함수)
    Tool 내부에서도 호출 가능
    """
    try:
        from rag.services.search import search_context

        debug_log(f"  🔍 RAG 검색: '{query}', k={k}")
        context, docs = search_context(query, k=k)

        tips = []
        for doc in docs:
            tips.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        debug_log(f"  ✅ {len(docs)}개 문서 검색 완료")

        return {
            "found": len(docs) > 0,
            "count": len(docs),
            "tips": tips,
            "context": context
        }

    except Exception as e:
        debug_log(f"  ❌ RAG 검색 실패: {e}")
        import traceback
        traceback.debug_log_exc()

        return {
            "found": False,
            "count": 0,
            "tips": [],
            "context": "",
            "error": str(e)
        }

# ============================================
# Tool 1-2: select_merchant (신규)
# ============================================

@mcp.tool()
def select_merchant(encoded_mct: str) -> Dict[str, Any]:
    """
    여러 검색 결과 중 특정 가맹점 선택

    Args:
        encoded_mct: 선택한 가맹점의 ENCODED_MCT 코드

    Returns:
        선택한 가맹점의 상세 정보
    """
    debug_log("select_merchant 특정 가맹점 선택하는 Tool 호출됨")

    if DF_SET1 is None:
        return {
            "found": False,
            "message": "데이터가 로드되지 않았습니다."
        }

    merchant_data = get_merchant_full_data(encoded_mct)

    if merchant_data is None:
        return {
            "found": False,
            "encoded_mct": encoded_mct,
            "message": f"가맹점 코드 '{encoded_mct}'를 찾을 수 없습니다."
        }

    basic = merchant_data["basic"]
    latest = merchant_data["latest"]

    debug_log("select_merchant 특정 가맹점 선택하는 Tool 호출 종료")

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
# Tool 2: search_merchant_knowledge
# ============================================

@mcp.tool()
def search_merchant_knowledge(query: str, k: int = 3) -> Dict[str, Any]:
    """RAG 기반 마케팅 팁 검색"""
    debug_log(f"\n🔍 search_merchant_knowledge Tool 호출: '{query}', k={k}")

    # 내부 함수 호출
    result = _search_rag_internal(query, k)

    debug_log(f"✅ RAG 검색 완료: {result.get('count', 0)}개\n")

    return result


# ============================================
# Tool 3: analyze_merchant_pattern (수정)
# ============================================

@mcp.tool()
def analyze_merchant_pattern(encoded_mct: str) -> Dict[str, Any]:
    """
    가맹점 패턴 분석 및 맞춤 전략 추천

    Args:
        encoded_mct: 분석할 가맹점의 ENCODED_MCT 코드

    Returns:
        패턴 분석 결과 및 마케팅 전략 추천
    """
    debug_log("analyze_merchant_pattern 가맹점 패턴 분석 및 맞춤 전략 추천 Tool 호출됨")

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

    if not matched_patterns:
        return {
            "found": True,
            "encoded_mct": encoded_mct,
            "pattern": None,
            "message": "매칭되는 패턴을 찾을 수 없습니다."
        }

    # 최우선 패턴 선택
    best_pattern = matched_patterns[0]

    # 심각도 계산
    severity = calculate_severity(best_pattern)

    # 전략 키워드 생성
    strategy_keywords = get_strategy_keywords(severity, merchant_data)

    # RAG 검색으로 마케팅 팁 수집
    all_tips = []
    for keyword in strategy_keywords:
        # ✅ 수정: Tool 호출 → 내부 함수 호출
        tip_result = _search_rag_internal(keyword, k=2)

        if tip_result.get("found"):
            all_tips.extend(tip_result.get("tips", []))

    recommendations = all_tips[:5]

    # 시각화 데이터
    latest = merchant_data.get("latest", {})
    sales = merchant_data.get("sales", [])
    diff_data = calculate_monthly_diff(sales)

    chart_data = {
        "encoded_mct": encoded_mct,
        "pattern_type": best_pattern.get("pattern_type"),
        "severity_level": severity.get("level"),
        "metrics": {
            "M12_SME_RY_SAA_PCE_RT": latest.get("M12_SME_RY_SAA_PCE_RT", 0),
            "M12_SME_BZN_SAA_PCE_RT": latest.get("M12_SME_BZN_SAA_PCE_RT", 0),
            "MCT_UE_CLN_REU_RAT": latest.get("MCT_UE_CLN_REU_RAT", 0),
            "MCT_UE_CLN_NEW_RAT": latest.get("MCT_UE_CLN_NEW_RAT", 0)
        },
        "diff": diff_data
    }

    debug_log("analyze_merchant_pattern 가맹점 패턴 분석 및 맞춤 전략 추천 Tool 호출 종료")

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
        "recommendations": recommendations,
        "chart_data": chart_data,
        "all_matched_patterns": [p.get("pattern_id") for p in matched_patterns[:3]],
        "message": f"{severity['label']} 패턴이 감지되었습니다. {severity['strategy_type']} 마케팅 전략을 추천합니다."
    }


# ============================================
# 서버 실행
# ============================================

# 서버 시작 시 데이터 로드
load_all_data()

if __name__ == "__main__":
    debug_log("\n" + "=" * 50)
    debug_log("MCP Server 시작")
    debug_log("=" * 50 + "\n")
    mcp.run()
