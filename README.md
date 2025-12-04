# FirstFin

**FirstFin**은 사회초년생을 위한 금융 상품 추천 데모입니다. Streamlit으로 구동되며 고객/로그/만족도 데이터를 활용해 라이프스타일 지표(TOM), 다중 추천 엔진, 소비 패턴 분석, 대화형 에이전트를 제공합니다.

## 주요 기능
- **데이터 적재 및 전처리**: `Data/` 폴더에 있는 예금/카드 상품, 고객, 행동 로그, 만족도 데이터를 읽어 `load_all_data()`에서 전처리합니다. 고객 데이터는 TOM 지표(`get_clean_tom_dataset_v2`)와 라이프스타일 피처(`create_lifestyle_tom_features`)로 확장됩니다.
- **라이프스타일 분석**: 카드 거래를 기반으로 주말/카페/편의점 비율, 소비 추세를 계산하고, 카테고리·시간대·요일 패턴을 분석합니다. 초과 지출 구간을 찾아 절약 가능 금액과 실천 팁을 제안합니다.
- **추천 엔진 스택**:
  - `FirstFinKNNRecommender`: TOM·라이프스타일 벡터로 유사 고객을 탐색해 상품을 제안.
  - `LogRecommender`: 최근 고객 행동 로그를 가중합 점수로 계산.
  - `SatisfactionRecommender`: 유사 고객의 만족도 평가를 집계.
  - `ZeroShotRecommender`: 페르소나 기반 제로샷 추천(비회원/콜드스타트 대응).
  - `HybridEngine`: 위 엔진과 룰 엔진(`run_rule_engine`)을 통합해 최종 추천을 생성하며, 소비 분석/저축 코치(`SpendingAnalyzer`, `HabitCoach`)도 포함합니다.
- **벡터 검색**: 상품 요약 텍스트를 OpenAI 임베딩으로 변환하여 FAISS 인덱스를 구축(`build_faiss_index`)하고, 유사 상품/정보 검색(`search_info`)에 사용합니다. 임베딩은 `cache/`에 캐시됩니다.
- **대화형 에이전트**: `run_agent()`가 OpenAI Chat Completions와 도구 호출(`run_hybrid`, `run_rule`, `get_details`, `search_info`)을 조합해 회원/비회원 모드에 맞는 답변을 만듭니다. 대화 기록은 세션별 파일로 저장/초기화할 수 있습니다.
- **Streamlit UI**:
  - 로그인(회원) 또는 페르소나 선택(비회원) 후 추천/검색/대화 기능 제공.
  - 회원 대시보드에서 라이프스타일 지표 바 차트, 소비 트렌드, 절약 계획을 시각화합니다.
- **프롬프트 요약 유틸리티**: `PromptDataSummarizer`가 로그·만족도 데이터를 LLM 프롬프트용 텍스트로 변환하고, 콜드스타트 여부 판단 및 제로샷 컨텍스트를 생성합니다.
- **원격 접속 지원**: `Run_ngrok.py`로 Streamlit 앱을 실행하고 ngrok 터널을 열어 외부에서 접근할 수 있습니다.

## 프로젝트 구조
- `app.py`: 데이터 로드, 피처 엔지니어링, 추천 엔진, 소비 분석/저축 코치, FAISS 인덱싱, LLM 에이전트, Streamlit UI를 모두 포함한 메인 앱.
- `prompt_summarizer.py`: 로그/만족도 데이터 요약 및 제로샷 컨텍스트 생성 유틸리티.
- `Data/`: 예금/카드 상품, 고객, 거래, 로그, 만족도 데이터를 보관하는 폴더. 리포지토리에는 샘플 CSV가 포함되어 있습니다.
- `Run_ngrok.py`: 로컬에서 Streamlit을 띄우고 ngrok 터널을 생성하는 스크립트.
- `참고용.ipynb`: 실험 및 분석용 노트북.

## 필요 데이터
`Data/` 폴더에 아래 파일 이름을 맞춰 배치하세요.
- `deposit_product_info_최신.xlsx`: 예금 상품 메타데이터
- `card_product_info_최신.xlsx`: 카드 상품 메타데이터
- `customers_with_id.csv`: 고객 기본 정보(TOM 계산에 사용)
- `card_transactions_updated.csv`: 카드 거래 내역(라이프스타일/소비 분석용)
- `customer_logs.csv`: 서비스 내 행동 로그(로그 기반 추천용)
- `customer_satisfaction.csv`: 상품별 만족도 평가(만족도 추천용)

## 환경 변수
- `OPENAI_API_KEY`: OpenAI 키. 설정하지 않으면 임베딩 생성, 벡터 검색, 대화형 에이전트가 비활성화됩니다.
- 필요 시 `.env` 파일을 사용하면 `python-dotenv`로 자동 로드됩니다.

## 실행 방법
1. (선택) 가상환경을 생성합니다.
2. 의존성을 설치합니다.
   ```bash
   pip install streamlit pandas numpy scikit-learn faiss-cpu openai python-dotenv pyngrok
   ```
3. OpenAI 키를 환경 변수로 설정합니다.
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
4. 데이터 파일을 `Data/`에 준비한 뒤 Streamlit을 실행합니다.
   ```bash
   streamlit run app.py
   ```
5. 브라우저에서 앱을 열어 회원/비회원 모드로 추천·검색·대화를 테스트합니다. 캐시가 필요하면 `cache/`가 자동으로 생성됩니다.
6. 외부 공유가 필요하면 `python Run_ngrok.py`를 실행하여 터널 URL을 확보합니다.

## 기타 메모
- OpenAI 키가 없을 때는 임베딩/에이전트 호출 대신 경고 메시지를 표시하며, UI만 표시됩니다.
- 상품 데이터가 바뀌면 임베딩 캐시 해시가 달라져 자동으로 새로 생성됩니다.
- 신규 고객의 경우, Persona를 기반으로 ZeroShotRecommender가 작동됩니다.
