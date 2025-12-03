 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index d872c8df5cf1ff8c76df814dc39441bfba3fcf5f..8ea740d4b10a171c709bf7d2b998661ee2eee21c 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,49 @@
-# FirstFin
\ No newline at end of file
+# FirstFin
+
+사회초년생을 위한 금융 상품 추천 데모 서비스입니다. Streamlit으로 구동되며, 카드/예금 상품 정보와 고객 로그를 기반으로 **TOM(Time-Occasion-Method) 라이프스타일 지표**, **하이브리드 추천 엔진**, **대화형 에이전트**를 제공합니다.
+
+## 주요 기능 개요
+- **데이터 로드 & 전처리**: `Data/` 폴더의 상품/고객/로그 데이터를 읽어 `load_all_data()`에서 전처리합니다. 고객 특성을 TOM 지표로 정제하기 위해 `get_clean_tom_dataset_v2()`와 `create_lifestyle_tom_features()`를 사용합니다.
+- **추천 엔진**: 
+  - `FirstFinKNNRecommender`: 고객 벡터를 기반으로 유사 고객을 찾습니다.
+  - `LogRecommender`: 최근 행동 로그를 가중합으로 점수화해 인기 상품을 추천합니다.
+  - `SatisfactionRecommender`: 유사 고객의 만족도 평점을 집계해 신뢰도 높은 상품을 노출합니다.
+  - `ZeroShotRecommender`: 로그가 부족한 콜드 스타트 고객/비회원에게 페르소나 기반 제로샷 추천을 제공합니다.
+  - `HybridEngine`: 위 엔진들의 결과를 통합하여 최종 추천을 반환합니다.
+- **벡터 검색**: `build_faiss_index()`가 상품 설명 임베딩을 생성·캐싱해 FAISS 인덱스를 만들고, `search_info()`에서 유사 상품을 빠르게 검색합니다.
+- **대화형 에이전트**: `run_agent()`가 OpenAI Chat Completions와 도구 호출(`run_hybrid`, `run_rule`, `get_details`, `search_info`)을 조합해 회원/비회원 모드에 맞는 답변을 생성합니다. `save_memory()`와 `load_memory()`로 대화 맥락을 파일에 기록합니다.
+- **Streamlit UI**: 
+  - 로그인(회원) 또는 페르소나 선택(비회원) 화면 제공
+  - 추천 결과 및 시스템 상태 표시
+  - 회원 전용 대시보드에서 TOM 지표 바 차트와 소비 추세 라인 차트를 시각화합니다.
+
+## 파일 구조와 역할
+- `app.py`: 전체 앱 로직이 집중된 메인 파일입니다.
+  - **환경 설정**: 페이지 설정, 로거, 경로/모델 상수, OpenAI 클라이언트 캐싱.
+  - **메모리 관리**: 세션별 대화 기록을 파일로 저장/삭제 (`get_memory_path`, `save_memory`, `load_memory`, `clear_memory`).
+  - **피처 엔지니어링**: TOM 지표 계산(`get_clean_tom_dataset_v2`)과 라이프스타일 피처 생성(`create_lifestyle_tom_features`).
+  - **데이터 적재**: `load_all_data()`가 엑셀/CSV를 안전하게 읽고, 고객 데이터로 학습용 피처 프레임을 구성합니다.
+  - **상품 DB & 인덱싱**: `build_product_db()`로 상품 메타를 생성하고, `build_faiss_index()`로 임베딩 인덱스를 만듭니다.
+  - **추천 엔진 클래스**: KNN, 로그 기반, 만족도 기반, 제로샷 추천 클래스와 이를 합치는 `HybridEngine`.
+  - **툴 함수**: Streamlit/LLM이 호출하는 `run_hybrid`, `run_rule`, `get_details`, `search_info`.
+  - **에이전트 및 UI**: `run_agent()`로 LLM 메시지 구성을 수행하고, 하단의 Streamlit 섹션에서 채팅 UI와 대시보드를 렌더링합니다.
+- `prompt_summarizer.py`: 로그·만족도 데이터를 LLM 프롬프트용 텍스트로 가공하는 유틸리티 클래스(`PromptDataSummarizer`). 로그 요약, 만족도 요약, 콜드스타트 판별, 제로샷 컨텍스트 생성 기능을 제공합니다.
+- `Data/`: 예금/카드 상품, 고객, 로그, 만족도 데이터 파일을 보관하는 폴더(리포지토리에는 샘플/빈 폴더만 포함될 수 있음).
+- `Run_ngrok.py`: ngrok 터널을 열어 외부에서 Streamlit 앱을 접근할 때 사용합니다.
+- `참고용.ipynb`: 분석/실험용 노트북.
+
+## 실행 방법
+1. OpenAI API 키를 환경변수 `OPENAI_API_KEY`로 설정하고, 필요 시 `.env` 파일에 저장합니다.
+2. `Data/` 폴더에 필요한 엑셀/CSV 데이터를 배치합니다.
+3. 의존성 설치: `pip install -r requirements.txt` (필요 시 Streamlit, pandas, scikit-learn, faiss-cpu, openai 등 설치).
+4. 애플리케이션 실행:
+   ```bash
+   streamlit run app.py
+   ```
+5. 로컬 브라우저(또는 ngrok 주소)에서 앱을 열어 회원/비회원 모드로 추천을 테스트합니다.
+
+## 개발 메모
+- OpenAI 키가 없으면 임베딩/대화 에이전트 기능이 비활성화됩니다. 이 경우 UI는 로드되지만 검색·추천 호출 시 경고 메시지를 표시합니다.
+- `cache/` 폴더에 상품 임베딩이 저장되며, 상품 데이터가 변경되면 캐시가 자동으로 무효화됩니다.
+- 신규 고객의 경우 로그가 부족하면 `ZeroShotRecommender`가 페르소나 기반 추천을 우선 제공합니다.
+
 
EOF
)
