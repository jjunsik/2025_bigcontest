# 🧠 Project Overview

이 프로젝트는 **Agent 기반 데이터 분석 및 시각화 시스템**으로 구성되어 있습니다.  
주요 구성 요소는 `App 코드`, `데이터 전처리(crawler)`, `분석 코드(analyzer)`로 이루어집니다.

---

## 🚀 App 코드

**Agent 실행 방법:**
```bash
uv run streamlit run streamlit_app.py
```

- Agent 실행 시 자동으로 **MCP 서버**와 **FAISS 벡터 DB**가 함께 구동됩니다.

---

## 🕸 데이터 전처리 디렉토리

### 📂 crawler
- **역할:** Gemini 2.5 Flash 기반 Agent를 통한 데이터 수집 및 전처리  
- **구성:**  
  - 실행 파일(`.py`)  
  - 실행 전 필요한 사전 데이터  
  - 실행 후 생성되는 사후 데이터  
- **실행 방법:**  
  1. `.env` 파일에 `GOOGLE_API_KEY` 입력  
  2. `main.py`의 `main()` 함수 실행  

---

### 📊 analyzer
- **역할:** 선형회귀 분석 수행  
- **구성:**  
  - Colab에서 실행 가능한 코드(`.ipynb`)  
  - 분석 결과 파일  
- **실행 방법:**  
  Colab 환경에서 해당 코드를 열어 실행  

