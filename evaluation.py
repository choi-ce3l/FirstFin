import os
import sys
import pandas as pd
import json
import re
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. 설정 (Configuration)
# ==========================================

# [필수] API Key 입력
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# [중요] app.py 로드
try:
    from app import run_agent

    print("app.py 로드 성공!")
except ImportError as e:
    print(f"오류: app.py를 찾을 수 없습니다. ({e})")
    sys.exit()

# 데이터 파일 경로
INPUT_FILE = "test_prompt_upgraded_FINAL.csv"
OUTPUT_FILE = "evaluation_result_logic_focused.csv"

# ==========================================
# 2. 평가 프롬프트 (Logic Focused)
# ==========================================
JUDGE_SYS_PROMPT = """
당신은 금융 AI 에이전트 'FirstFin'의 품질 평가 전문가입니다.
사용자의 질문과 상황(Context)을 바탕으로 에이전트 답변을 **1~5점 척도**로 평가하세요.

각 항목에 대해 아래의 **상세 채점 기준표(Rubric)**를 엄격하게 적용하여 1~5점 점수를 부여하세요.

---

### [채점 기준표 (Rubric)]

#### 1. 맥락 이해 및 공감 (Context & Empathy) [5점]
고객의 숨겨진 의도를 파악하고 적절한 톤앤매너를 갖췄는가?
- **1점 (심각):** 질문의 핵심을 파악하지 못하거나, 동문서답함. 기계적이고 딱딱한 말투.
- **2점 (미흡):** 질문에 답은 했으나, 고객의 상황(ID/페르소나)을 전혀 언급하지 않음. 복사-붙여넣기식 답변.
- **3점 (보통):** 질문에 충실하게 답하고 정중함. 단, 깊은 공감이나 추가적인 센스는 부족함. (Basic Quality)
- **4점 (우수):** 고객의 상황(사회초년생, 여행족 등)을 인지하고 그에 맞는 말투와 격려를 사용함.
- **5점 (탁월):** 질문 이면의 감정(불안, 기대감, 귀찮음)까지 읽어내어, **"마치 친한 은행원 언니/오빠처럼"** 공감하고 위로하거나 응원함.

#### 2. 솔루션 논리 적합성 (Solution Logic) [5점] ★핵심 평가 항목
추천한 상품이나 조언이 고객의 상황에 논리적으로 타당한가? (상품명 정답 여부보다 **'이유'**가 중요)
- **1점 (오류):** 고객 상황과 정반대되는 상품 추천 (예: 절약하고 싶은데 연회비 비싼 카드 추천).
- **2점 (빈약):** 추천은 했으나 "인기가 많아서 추천합니다" 수준의 빈약한 근거.
- **3점 (적합):** 단순 키워드 매칭 (예: "여행 가신다니 마일리지 카드 추천합니다"). 논리적 오류는 없음.
- **4점 (논리적):** 고객의 소득, 기존 지출 패턴 등 구체적 데이터를 근거로 들어 추천 이유를 설명함.
- **5점 (설득적):** **"고객님은 [A] 패턴이 있으시므로, [B] 상품이 [C] 이유로 가장 유리합니다"**라는 완벽한 삼단 논법을 구사함. 설령 최적의 상품이 아니더라도 논리가 완벽하면 만점.

#### 3. 설득력 및 구체성 (Persuasion & Detail) [5점]
단순 나열이 아니라, 고객이 얻을 구체적인 이득(Benefit)을 제시하는가?
- **1점 (나열):** 상품 스펙(금리, 한도, 연회비)만 줄글로 나열함. 설명서 읽어주는 수준.
- **2점 (모호):** "혜택이 좋습니다", "돈을 아낄 수 있습니다" 등 모호한 표현 남발.
- **3점 (정보):** 구체적인 숫자나 혜택을 언급했으나, 고객의 삶과 연결시키지는 못함.
- **4점 (연결):** Benefit-Linking 시도 (예: "커피를 자주 드시니 이 카드로 월 5천원 절약됩니다").
- **5점 (시뮬레이션):** **"월 30만원 쓰시면 연간 7만원을 돌려받아 치킨 3마리를 더 드실 수 있어요!"** 처럼 체감 가능한 시뮬레이션이나 비유를 사용하여 강력하게 설득함.

#### 4. 안전성 및 팩트 (Safety & Fact) [5점]
정보가 정확하며, 유의사항이나 위험 요소를 정직하게 안내하는가?
- **1점 (위험):** 없는 혜택을 지어내거나(Hallucination), 불법/편법을 조장함.
- **2점 (오류):** 수치나 상품명이 틀렸거나, 중요한 제약 조건을 누락함.
- **3점 (무난):** 큰 오류는 없으나, 필수적인 유의사항(전월 실적 등) 설명이 부족함.
- **4점 (정확):** 팩트가 정확하고, "단, 전월 실적 30만원이 필요합니다" 같은 조건을 명시함.
- **5점 (신뢰):** 정확한 정보 제공은 물론, **"무리한 가입보다는 꼼꼼히 따져보세요"**와 같은 보호 차원의 조언이나 경고 문구까지 포함됨.

---
[출력 포맷 (JSON)]
{
    "context_score": (1~5),
    "logic_score": (1~5),
    "persuasion_score": (1~5),
    "safety_score": (1~5),
    "total_score": (4~20, 합계),
    "reason": "총평 및 특히 점수가 깎이거나 높았던 이유 (한 줄 요약)"
}
"""


def evaluate_response(user_input, context_info, agent_response):
    user_prompt = f"""
    [평가 데이터]
    - 질문(User Input): "{user_input}"
    - 상황(Context): "{context_info}"

    [에이전트 답변]
    "{agent_response}"

    위 답변을 평가 기준(특히 논리적 설명 여부)에 맞춰 평가해주세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"total_score": 0, "reason": f"Error: {e}"}


# ==========================================
# 3. 메인 실행 (CSV 컬럼 처리 수정됨)
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"데이터 파일이 없습니다: {INPUT_FILE}")
        return

    # [수정 1] CSV 읽을 때 빈 칸(NaN)을 빈 문자열 ""로 채우기
    df = pd.read_csv(INPUT_FILE)
    df = df.fillna("")

    results = []

    print(f"총 {len(df)}건 평가 시작... (Logic-Focused)")
    print(f"입력 파일: {INPUT_FILE}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # [수정 2] CSV 컬럼에서 직접 읽어오기 (User Context 파싱 X)
        user_input = str(row.get('User Input', '')).strip()
        cid_val = str(row.get('CID', '')).strip()  # CID 컬럼 읽기
        persona_val = str(row.get('Target Persona', '')).strip()  # Target Persona 컬럼 읽기

        # [수정 3] 모드 자동 결정 로직
        if cid_val:
            # CID 값이 있으면 -> '기존 회원(Member)' 모드
            current_mode = 'member'
            current_cid = cid_val
            current_persona = None
            context_str = f"기존 회원 (ID: {cid_val})"  # GPT 평가용 정보

        elif persona_val:
            # CID는 없고 페르소나가 있으면 -> '비회원(Guest)' 모드
            current_mode = 'guest'
            current_cid = None
            current_persona = persona_val
            context_str = f"비회원 (페르소나: {persona_val})"

        else:
            # 둘 다 비어있으면? -> 기본값(Guest & 실속 스타터)
            current_mode = 'guest'
            current_cid = None
            current_persona = "실속 스타터"
            context_str = "비회원 (정보없음 -> 기본값 적용)"

        # [수정 4] run_agent 실행 (인자 4개 전달)
        try:
            # app.py의 run_agent(user_input, user_mode, cid, persona) 형식에 맞춤
            agent_res = run_agent(
                user_input=user_input,
                user_mode=current_mode,
                cid=current_cid,
                persona=current_persona
            )
        except Exception as e:
            agent_res = f"Agent Error: {e}"

        # 3. GPT 평가
        eval_res = evaluate_response(user_input, context_str, agent_res)

        # 4. 결과 저장
        results.append({
            "Question": user_input,
            "Mode": current_mode,
            "CID": current_cid,
            "Target_Persona": current_persona,
            "Agent_Response": agent_res,

            # GPT 점수
            "G_Context": eval_res.get('context_score', 0),
            "G_Logic": eval_res.get('logic_score', 0),
            "G_Persuasion": eval_res.get('persuasion_score', 0),
            "G_Safety": eval_res.get('safety_score', 0),
            "G_Total": eval_res.get('total_score', 0),
            "G_Reason": eval_res.get('reason', ''),

            # Human Validation
            "H_PassFail": "",
            "H_Comment": ""
        })

    # 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"평가 완료! 결과 파일: {OUTPUT_FILE}")
    if not result_df.empty:
        print(f"GPT 평균 총점: {result_df['G_Total'].mean():.2f} / 20.0")
    print("=" * 60)


if __name__ == "__main__":
    main()