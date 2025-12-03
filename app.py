import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import faiss
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import logging
import uuid

# dotenv 안전 import
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# -----------------------------------------
# 1. 설정 및 초기화
# -----------------------------------------
st.set_page_config(page_title="FirstFin - 사회초년생을 위한 맥락인지형 Agent", layout="wide")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = './Data/'
CACHE_PATH = './cache/'
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_VERSION = "v1"

os.makedirs(CACHE_PATH, exist_ok=True)

# -----------------------------------------
# [수정] 유효한 페르소나 목록을 상수로 정의
# -----------------------------------------
VALID_PERSONAS = ['밸런스 메인스트림', '스마트 플렉서', '알뜰 지킴이', '실속 스타터', '디지털 힙스터']
DEFAULT_PERSONA = None  # 기본값을 None으로 설정하여 명시적 선택 강제


# -----------------------------------------
# 2. OpenAI 클라이언트 (지연 초기화)
# -----------------------------------------
@st.cache_resource
def get_openai_client():
    """
    OpenAI API 클라이언트를 초기화하고 캐싱한다.
    환경 변수에서 API 키를 읽어 클라이언트를 생성한다.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        return None


# -----------------------------------------
# 3. 메모리 함수 (대화 기록 저장/로드)
# -----------------------------------------
def get_memory_path():
    """세션별 고유한 메모리 파일 경로를 반환한다."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return f"./FirstFin_memory_{st.session_state.session_id}.txt"


def save_memory(user_msg, assistant_msg):
    """사용자와 어시스턴트의 대화를 파일에 저장한다."""
    try:
        with open(get_memory_path(), "a", encoding="utf-8") as f:
            f.write(f"User: {user_msg}\nAgent: {assistant_msg}\n")
    except Exception as e:
        logger.warning(f"메모리 저장 실패: {e}")


def load_memory():
    """저장된 대화 기록을 파일에서 읽어온다."""
    try:
        path = get_memory_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"메모리 로드 실패: {e}")
    return ""


def clear_memory():
    """저장된 대화 기록 파일을 삭제한다."""
    try:
        path = get_memory_path()
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"메모리 삭제 실패: {e}")


# -----------------------------------------
# 4. 피처 엔지니어링 (TOM 지표)
# -----------------------------------------
# TOM 스키마: 고객 데이터에서 사용할 컬럼 정의
TOM_SCHEMA = {
    'required': ['customer_id'],
    'numeric_optional': [
        'SHC_GOLF_GD', 'SHC_VIP_CARD_TF', 'SHC_BUY_LUX_TF',
        'FIN_STOCK_24_4', 'FIN_COIN_24_4',
        'SHC_TRAVEL_AMT_24_4', 'SHC_ENT_AMT_24_4', 'SHC_STARBUCKS_AMT_24_4',
        'SHC_HOTEL_AMT_24_4', 'SHC_M_DF_AMT_24_4', 'SHC_1YEAR_MEAN_AMT',
        'ENT_SVOD_24_4', 'ENT_WEBTOON_24_4', 'COMM_SNS_24_4', 'SHOP_SOCIAL_24_4',
        'SHC_E_COMM_AMT_24_4', 'SHC_DLV_AMT_24_4',
        'SHC_ACCO_AMT_24_4', 'SHC_DEP_AMT_24_4', 'SHC_CLOTHES_AMT_24_4',
        'SHC_FOOD_AMT_24_4', 'SHC_CUL_AMT_24_4', 'NET_ASST_24'
    ],
    'categorical_optional': ['AGE', 'SEX', 'JB_TP']
}


def safe_get_column(df, col, default=0):
    """
    DataFrame에서 컬럼을 안전하게 가져온다.
    컬럼이 없으면 기본값으로 채워진 Series를 반환한다.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def get_clean_tom_dataset_v2(df):
    """
    원시 고객 데이터에서 TOM 지표를 계산한다.
    - TOM_Invest: 투자 성향 (주식 + 코인*2)
    - TOM_YOLO: 여가/사치 소비 비율
    - TOM_Digital: 디지털 친화도
    - TOM_Main_Interest: 주요 관심 카테고리
    - TOM_Asset: 순자산
    """
    temp_df = df.copy()
    if 'customer_id' not in temp_df.columns:
        logger.error("customer_id 컬럼이 없습니다.")
        return pd.DataFrame()

    # 수치형 컬럼 안전하게 변환
    for col in TOM_SCHEMA['numeric_optional']:
        temp_df[col] = safe_get_column(temp_df, col, 0)

    # TOM_Invest: 투자 성향 지표 (주식 거래 + 코인 거래*2)
    temp_df['TOM_Invest'] = (
            safe_get_column(temp_df, 'FIN_STOCK_24_4', 0) +
            safe_get_column(temp_df, 'FIN_COIN_24_4', 0) * 2.0
    )

    # TOM_YOLO: 여가/사치 소비 비율 (여행, 엔터테인먼트, 스타벅스, 호텔, 면세점)
    yolo_cols = ['SHC_TRAVEL_AMT_24_4', 'SHC_ENT_AMT_24_4', 'SHC_STARBUCKS_AMT_24_4',
                 'SHC_HOTEL_AMT_24_4', 'SHC_M_DF_AMT_24_4']
    total_spend = safe_get_column(temp_df, 'SHC_1YEAR_MEAN_AMT', 1).replace(0, 1)
    yolo_sum = sum(safe_get_column(temp_df, c, 0) for c in yolo_cols)
    temp_df['TOM_YOLO'] = yolo_sum / total_spend

    # TOM_Digital: 디지털 친화도 (OTT, 웹툰, SNS, 소셜커머스 이용 + 이커머스/배달 소비)
    digital_interest = ['ENT_SVOD_24_4', 'ENT_WEBTOON_24_4', 'COMM_SNS_24_4', 'SHOP_SOCIAL_24_4']
    digital_action = ['SHC_E_COMM_AMT_24_4', 'SHC_DLV_AMT_24_4']
    interest_mean = pd.concat([safe_get_column(temp_df, c, 0) for c in digital_interest], axis=1).mean(axis=1)
    action_sum = sum(safe_get_column(temp_df, c, 0) for c in digital_action)
    temp_df['TOM_Digital'] = interest_mean + (action_sum / total_spend * 5.0)

    # TOM_Main_Interest: 주요 관심 카테고리 (여행, 쇼핑, 음식, 문화 중 최대)
    categories = {
        'Travel': ['SHC_TRAVEL_AMT_24_4', 'SHC_ACCO_AMT_24_4'],
        'Shopping': ['SHC_DEP_AMT_24_4', 'SHC_CLOTHES_AMT_24_4'],
        'Food': ['SHC_FOOD_AMT_24_4', 'SHC_STARBUCKS_AMT_24_4'],
        'Culture': ['SHC_ENT_AMT_24_4', 'SHC_CUL_AMT_24_4']
    }
    cat_scores = pd.DataFrame(index=temp_df.index)
    for cat, cols in categories.items():
        cat_scores[cat] = sum(safe_get_column(temp_df, c, 0) for c in cols)
    temp_df['TOM_Main_Interest'] = cat_scores.idxmax(axis=1)

    # TOM_Asset: 순자산
    temp_df['TOM_Asset'] = safe_get_column(temp_df, 'NET_ASST_24', 0)

    # 필요한 컬럼만 추출
    keep_cols = ['customer_id', 'AGE', 'SEX', 'JB_TP', 'TOM_Invest', 'TOM_YOLO',
                 'TOM_Digital', 'TOM_Asset', 'TOM_Main_Interest']
    return temp_df[[c for c in keep_cols if c in temp_df.columns]].copy()


def create_lifestyle_tom_features(trans_df, profile_df):
    """
    거래 데이터를 분석하여 라이프스타일 TOM 피처를 생성한다.
    - TOM_Weekend: 주말 소비 비율
    - TOM_Cafe: 카페/식당 이용 비율
    - TOM_Conv: 편의점 이용 비율
    - TOM_Trend: 소비 증감 추세 (선형회귀 기울기)
    """
    if trans_df.empty or profile_df.empty:
        return profile_df

    local_trans = trans_df.copy()
    local_profile = profile_df.copy()

    # 거래 날짜 파싱
    try:
        local_trans['transaction_date'] = pd.to_datetime(
            local_trans['transaction_date'], format='mixed', errors='coerce'
        )
        local_trans = local_trans.dropna(subset=['transaction_date'])
    except Exception as e:
        logger.warning(f"날짜 변환 실패: {e}")
        return profile_df

    if local_trans.empty:
        return profile_df

    # 요일 및 월 인덱스 계산
    local_trans['day_of_week'] = local_trans['transaction_date'].dt.dayofweek
    local_trans['month_idx'] = (
            local_trans['transaction_date'].dt.year * 12 +
            local_trans['transaction_date'].dt.month
    )

    # TOM_Weekend: 주말(토, 일) 소비 비율
    grouped = local_trans.groupby('customer_id')
    weekend_mask = local_trans['day_of_week'] >= 5
    weekend_spend = local_trans[weekend_mask].groupby('customer_id')['amount'].sum()
    total_spend = grouped['amount'].sum().replace(0, 1)
    tom_weekend = (weekend_spend / total_spend).reindex(local_profile['customer_id']).fillna(0)

    # TOM_Cafe, TOM_Conv: 카테고리별 이용 비율
    tom_cafe = pd.Series(0.0, index=local_profile['customer_id'])
    tom_conv = pd.Series(0.0, index=local_profile['customer_id'])

    if 'merchant_category' in local_trans.columns:
        cat_counts = local_trans.pivot_table(
            index='customer_id', columns='merchant_category',
            values='transaction_id', aggfunc='count', fill_value=0
        )
        total_counts = grouped['transaction_id'].count().replace(0, 1)
        if '식당/카페' in cat_counts.columns:
            tom_cafe = (cat_counts['식당/카페'] / total_counts).reindex(local_profile['customer_id']).fillna(0)
        if '편의점' in cat_counts.columns:
            tom_conv = (cat_counts['편의점'] / total_counts).reindex(local_profile['customer_id']).fillna(0)

    # TOM_Trend: 월별 소비 추세 (선형회귀 기울기 / 평균)
    slopes = {}
    monthly_spend = local_trans.groupby(['customer_id', 'month_idx'])['amount'].sum().reset_index()
    for cust_id, group in monthly_spend.groupby('customer_id'):
        if len(group) > 1:
            X = group['month_idx'].values.reshape(-1, 1)
            y = group['amount'].values
            mean_y = np.mean(y) if np.mean(y) != 0 else 1
            model = LinearRegression().fit(X, y)
            slopes[cust_id] = model.coef_[0] / mean_y
        else:
            slopes[cust_id] = 0
    tom_trend_raw = pd.Series(slopes, name='TOM_Trend_Raw').reindex(local_profile['customer_id']).fillna(0)

    # 라이프스타일 피처 DataFrame 생성
    lifestyle_df = pd.DataFrame({
        'customer_id': local_profile['customer_id'].values,
        'TOM_Weekend': tom_weekend.values,
        'TOM_Cafe': tom_cafe.values,
        'TOM_Conv': tom_conv.values,
        'TOM_Trend_Raw': tom_trend_raw.values
    })
    final_df = pd.merge(local_profile, lifestyle_df, on='customer_id', how='left').fillna(0)

    # 수치형 피처 정규화 (0~1 스케일링)
    scaler = MinMaxScaler()
    num_cols = ['AGE', 'TOM_Invest', 'TOM_YOLO', 'TOM_Digital', 'TOM_Asset',
                'TOM_Weekend', 'TOM_Cafe', 'TOM_Conv']
    valid_nums = [c for c in num_cols if c in final_df.columns]
    if valid_nums:
        final_df[valid_nums] = scaler.fit_transform(final_df[valid_nums])

    # TOM_Trend: -1 ~ 1 범위로 클리핑
    final_df['TOM_Trend'] = final_df['TOM_Trend_Raw'].clip(-1, 1)

    # 범주형 피처 원핫 인코딩
    cat_cols = ['SEX', 'JB_TP', 'TOM_Main_Interest']
    valid_cats = [c for c in cat_cols if c in final_df.columns]
    final_df = pd.get_dummies(final_df, columns=valid_cats, prefix=['Sex', 'Job', 'Interest'])

    return final_df


# -----------------------------------------
# 5. 데이터 로드
# -----------------------------------------
def get_embedding_cache_path(product_db):
    """상품 DB 내용 기반으로 임베딩 캐시 파일 경로를 생성한다."""
    content_hash = hashlib.md5(
        product_db['summary_text'].to_json().encode()
    ).hexdigest()[:8]
    return os.path.join(CACHE_PATH, f"embeddings_{EMBEDDING_MODEL}_{EMBEDDING_VERSION}_{content_hash}.npy")


@st.cache_data(show_spinner="데이터 로드 중...")
def load_all_data():
    """
    모든 데이터 파일을 로드한다.
    - deposit: 예금 상품 정보
    - card: 카드 상품 정보
    - customers: 고객 기본 정보
    - customers_train: TOM 피처가 추가된 고객 데이터
    - logs: 고객 행동 로그
    - satisfaction: 고객 만족도 데이터
    """
    data = {
        'deposit': pd.DataFrame(),
        'card': pd.DataFrame(),
        'customers': pd.DataFrame(),
        'customers_train': pd.DataFrame(),
        'logs': pd.DataFrame(),
        'satisfaction': pd.DataFrame()
    }

    if not os.path.exists(DATA_PATH):
        logger.warning(f"데이터 경로가 존재하지 않습니다: {DATA_PATH}")
        return data

    # 예금 상품 로드
    try:
        deposit_path = DATA_PATH + "deposit_product_info_최신.xlsx"
        if os.path.exists(deposit_path):
            data['deposit'] = pd.read_excel(deposit_path)
            logger.info(f"예금 상품 로드: {len(data['deposit'])}개")
    except Exception as e:
        logger.warning(f"예금 상품 로드 실패: {e}")

    # 카드 상품 로드
    try:
        card_path = DATA_PATH + "card_product_info_최신.xlsx"
        if os.path.exists(card_path):
            data['card'] = pd.read_excel(card_path)
            logger.info(f"카드 상품 로드: {len(data['card'])}개")
    except Exception as e:
        logger.warning(f"카드 상품 로드 실패: {e}")

    # 고객 데이터 로드 및 TOM 피처 생성
    try:
        cust_path = DATA_PATH + "customers_with_id.csv"
        if os.path.exists(cust_path):
            raw_cust = pd.read_csv(cust_path)
            data['customers'] = raw_cust
            logger.info(f"고객 데이터 로드: {len(raw_cust)}명")

            if not raw_cust.empty:
                basic = get_clean_tom_dataset_v2(raw_cust)
                trans_path = DATA_PATH + "card_transactions_updated.csv"
                if os.path.exists(trans_path):
                    raw_trans = pd.read_csv(trans_path)
                    data['customers_train'] = create_lifestyle_tom_features(raw_trans,
                                                                            basic) if not raw_trans.empty else basic
                else:
                    data['customers_train'] = basic
    except Exception as e:
        logger.warning(f"고객 데이터 로드 실패: {e}")

    # 고객 행동 로그 로드
    try:
        logs_path = DATA_PATH + "customer_logs.csv"
        if os.path.exists(logs_path):
            data['logs'] = pd.read_csv(logs_path)
            logger.info(f"로그 데이터 로드: {len(data['logs'])}건")
    except Exception as e:
        logger.warning(f"로그 데이터 로드 실패: {e}")

    # 고객 만족도 데이터 로드
    try:
        sat_path = DATA_PATH + "customer_satisfaction.csv"
        if os.path.exists(sat_path):
            data['satisfaction'] = pd.read_csv(sat_path)
            logger.info(f"만족도 데이터 로드: {len(data['satisfaction'])}건")
    except Exception as e:
        logger.warning(f"만족도 데이터 로드 실패: {e}")

    return data


def build_product_db(data):
    """
    예금 및 카드 상품 정보를 통합하여 상품 DB를 구축한다.
    각 상품에 대해 요약 텍스트를 생성한다.
    """
    rows = []
    for _, r in data['deposit'].iterrows():
        rows.append({
            "product_id": r.get('product_id', ''),
            "product_name": r.get('product_name', ''),
            "product_type": "deposit",
            "summary_text": f"[{r.get('product_id', '')}] {r.get('product_name', '')} (예금): 금리 {r.get('max_rate', '')}%"
        })
    for _, r in data['card'].iterrows():
        rows.append({
            "product_id": r.get('product_id', ''),
            "product_name": r.get('product_name', ''),
            "product_type": "card",
            "category": r.get('card_category', ''),
            "summary_text": f"[{r.get('product_id', '')}] {r.get('product_name', '')} (카드): 혜택 {str(r.get('benefits', ''))[:100]}"
        })
    return pd.DataFrame(rows)


def build_faiss_index(product_db, client):
    """
    상품 요약 텍스트의 임베딩을 생성하고 FAISS 인덱스를 구축한다.
    캐시가 있으면 재사용하고, 없으면 새로 생성한다.
    """
    if len(product_db) == 0 or client is None:
        return None

    cache_path = get_embedding_cache_path(product_db)
    embeddings = None

    # 캐시된 임베딩 로드 시도
    if os.path.exists(cache_path):
        try:
            embeddings = np.load(cache_path)
            if embeddings.shape[0] != len(product_db) or embeddings.shape[1] != 1536:
                logger.warning("캐시된 임베딩 차원 불일치, 재생성합니다.")
                embeddings = None
                os.remove(cache_path)
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            embeddings = None

    # 새 임베딩 생성
    if embeddings is None:
        logger.info("새로운 임베딩 생성 중...")
        try:
            texts = product_db["summary_text"].tolist()
            batch_size = 50
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)

            embeddings = np.array(all_embeddings, dtype="float32")
            np.save(cache_path, embeddings)
            logger.info(f"임베딩 캐시 저장: {cache_path}")
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None

    # FAISS 인덱스 생성 (L2 거리 기반)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# -----------------------------------------
# 6. 추천 엔진
# -----------------------------------------

# [수정] 페르소나 정규화 함수 - 폴백 없이 None 반환
def normalize_persona_name(persona_with_suffix, fallback=None):
    """
    페르소나 이름에서 기본 페르소나를 추출한다.

    [수정사항]
    - 기존: 매칭 실패 시 무조건 '실속 스타터' 반환
    - 변경: 매칭 실패 시 fallback 파라미터 사용, fallback도 None이면 None 반환

    Args:
        persona_with_suffix: 접미사가 붙은 페르소나 이름 (예: "스마트 플렉서 (최근 소비 급증)")
        fallback: 매칭 실패 시 반환할 기본값 (None이면 None 반환)

    Returns:
        정규화된 페르소나 이름 또는 fallback 값
    """
    if persona_with_suffix is None:
        return fallback

    for base in VALID_PERSONAS:
        if base in persona_with_suffix:
            return base
    return fallback


def validate_persona(persona_name):
    """
    페르소나 이름이 유효한지 검증한다.

    Args:
        persona_name: 검증할 페르소나 이름

    Returns:
        bool: 유효하면 True, 아니면 False
    """
    if persona_name is None:
        return False
    normalized = normalize_persona_name(persona_name)
    return normalized in VALID_PERSONAS


class FirstFinKNNRecommender:
    """
    K-최근접 이웃(KNN) 기반 유사 고객 추천기.
    코사인 유사도를 사용하여 특성이 비슷한 고객을 찾는다.
    """

    def __init__(self, df):
        self.model = None
        self.df = None
        self.features = None
        if len(df) == 0:
            return
        self.df = df.set_index('customer_id') if 'customer_id' in df.columns else df
        self.features = self.df.select_dtypes(include=[np.number]).fillna(0)
        if len(self.features.columns) > 0:
            self.model = NearestNeighbors(n_neighbors=min(6, len(self.features)), metric='cosine')
            self.model.fit(self.features)

    def get_similar(self, cid, n=5):
        """주어진 고객 ID와 유사한 고객 목록을 반환한다."""
        if self.model is None or self.df is None or cid not in self.df.index:
            return []
        try:
            _, idx = self.model.kneighbors(self.features.loc[[cid]], n_neighbors=min(n + 1, len(self.features)))
            return self.df.iloc[idx[0][1:]].index.tolist()
        except Exception as e:
            logger.warning(f"KNN 추천 실패: {e}")
            return []


class LogRecommender:
    """
    고객 행동 로그 기반 추천기.
    클릭, 조회, 비교, 신청 등의 행동에 가중치를 부여하여 추천한다.
    """

    def __init__(self, df, card_df, dep_df):
        self.df = df.copy() if len(df) > 0 else pd.DataFrame()
        if len(self.df) > 0:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        # 행동 유형별 가중치
        self.weights = {'apply': 5, 'compare': 3, 'view': 2, 'click': 1}

    def recommend(self, cid, days=30, k=5):
        """최근 N일간의 행동 로그를 분석하여 상품을 추천한다."""
        if len(self.df) == 0:
            return []
        try:
            cutoff = self.df['timestamp'].max() - timedelta(days=days)
            logs = self.df[(self.df['customer_id'] == cid) & (self.df['timestamp'] >= cutoff)]
            scores = {}
            for _, r in logs.iterrows():
                pid = r['product_id']
                if pid not in scores:
                    scores[pid] = {'score': 0, 'name': r.get('product_name', ''), 'cat': r.get('product_category', '')}
                scores[pid]['score'] += self.weights.get(r.get('action_type', ''), 1)
            return [
                {'product_id': p, 'product_name': d['name'], 'category': d['cat'], 'score': round(d['score'], 2),
                 'source': 'log'}
                for p, d in sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:k]
            ]
        except Exception as e:
            logger.warning(f"로그 추천 실패: {e}")
            return []

    def summary(self, cid, days=7):
        """고객의 최근 활동을 요약한다."""
        if len(self.df) == 0:
            return "로그 데이터 없음"
        try:
            cutoff = self.df['timestamp'].max() - timedelta(days=days)
            logs = self.df[(self.df['customer_id'] == cid) & (self.df['timestamp'] >= cutoff)]
            if len(logs) == 0:
                return "최근 활동 없음"
            acts = logs['action_type'].value_counts().to_dict()
            prods = logs.groupby('product_name').size().nlargest(3).index.tolist()
            return (f"최근 {days}일간 총 {len(logs)}건 활동 감지 "
                    f"(클릭 {acts.get('click', 0)}회, 조회 {acts.get('view', 0)}회, "
                    f"비교 {acts.get('compare', 0)}회, 신청 {acts.get('apply', 0)}회). "
                    f"특히 '{', '.join(prods)}' 상품에 높은 관심을 보임.")
        except Exception as e:
            logger.warning(f"로그 요약 실패: {e}")
            return "로그 분석 중 오류 발생"


class SatisfactionRecommender:
    """
    고객 만족도 기반 추천기.
    유사 고객들이 높게 평가한 상품을 추천한다.
    """

    def __init__(self, df, card_df, dep_df):
        self.df = df
        self.card_df = card_df
        self.dep_df = dep_df

    def recommend(self, cid, similar_ids, k=5):
        """유사 고객들의 만족도 데이터를 기반으로 상품을 추천한다."""
        if len(self.df) == 0 or not similar_ids:
            return []
        try:
            sim = self.df[self.df['customer_id'].isin(similar_ids)]
            stats = sim[sim['rating'] >= 4.0].groupby(
                ['product_id', 'product_name', 'product_type']
            ).agg({'rating': 'mean', 'customer_id': 'count'}).reset_index()
            # 점수 = 평균 평점 * log(평가 수 + 1)
            stats['score'] = stats['rating'] * np.log1p(stats['customer_id'])
            return [
                {'product_id': r['product_id'], 'product_name': r['product_name'], 'score': round(r['score'], 2),
                 'source': 'satisfaction'}
                for _, r in stats.nlargest(k, 'score').iterrows()
            ]
        except Exception as e:
            logger.warning(f"만족도 추천 실패: {e}")
            return []


class ZeroShotRecommender:
    """
    콜드 스타트/비회원용 제로샷 추천기.
    사전 정의된 페르소나 프로필을 기반으로 키워드 매칭하여 추천한다.
    """
    # 페르소나별 프로필 정의
    PROFILES = {
        0: {'name': '밸런스 메인스트림', 'keywords': ['일상', '생활', '직장인'], 'card_keywords': ['일상', '생활', '직장인'],
            'dep_keywords': ['예금', '입출금', '자유']},
        1: {'name': '스마트 플렉서', 'keywords': ['여행', '쇼핑', '프리미엄'], 'card_keywords': ['여행', '쇼핑', '프리미엄', '항공'],
            'dep_keywords': ['예금', '투자', '고금리']},
        2: {'name': '알뜰 지킴이', 'keywords': ['생활', '마트', '공과금'], 'card_keywords': ['생활', '마트', '할인', '캐시백'],
            'dep_keywords': ['적금', '예금', '안전']},
        3: {'name': '실속 스타터', 'keywords': ['청년', '교통', '통신'], 'card_keywords': ['청년', '교통', '통신', '학생'],
            'dep_keywords': ['적금', '청년', '목돈']},
        4: {'name': '디지털 힙스터', 'keywords': ['쇼핑', '디지털', '문화'], 'card_keywords': ['쇼핑', '디지털', '온라인', '구독'],
            'dep_keywords': ['입출금', '적금', '모바일']}
    }

    def __init__(self, cust_df, card_df, dep_df, log_df, sat_df):
        self.cust_df = cust_df
        self.card_df = card_df
        self.dep_df = dep_df
        self.log_df = log_df
        self.sat_df = sat_df

    def is_cold(self, cid):
        """고객이 콜드 스타트 상태인지 확인한다. (로그 5건 미만)"""
        if len(self.log_df) == 0:
            return True
        return len(self.log_df[self.log_df['customer_id'] == cid]) < 5

    def _fuzzy_match(self, text, keywords, threshold=0.4):
        """텍스트와 키워드 간의 퍼지 매칭 점수를 계산한다."""
        if pd.isna(text):
            return 0
        text = str(text).lower()
        max_score = 0
        for kw in keywords:
            if kw.lower() in text:
                max_score = max(max_score, 1.0)
            else:
                ratio = SequenceMatcher(None, kw.lower(), text).ratio()
                max_score = max(max_score, ratio)
        return max_score if max_score >= threshold else 0

    def _get_profile_by_name(self, persona_name):
        """
        [수정] 페르소나 이름으로 프로필을 찾는다.
        매칭 실패 시 None을 반환한다 (기존: 실속 스타터 반환).
        """
        if persona_name is None:
            return None
        normalized = normalize_persona_name(persona_name)
        if normalized is None:
            return None
        for pid, p in self.PROFILES.items():
            if p['name'] == normalized:
                return p
        return None

    def recommend(self, cid, k=5):
        """
        [수정] 고객 ID 기반으로 페르소나를 찾아 추천한다.
        고객 데이터가 없으면 빈 리스트를 반환한다 (기존: 실속 스타터로 폴백).
        """
        if len(self.cust_df) == 0:
            return []
        cust_row = self.cust_df[self.cust_df['customer_id'] == cid]

        # [수정] 고객이 없으면 빈 리스트 반환 (폴백 없음)
        if cust_row.empty:
            logger.warning(f"고객 ID {cid}를 찾을 수 없습니다. 추천을 생성할 수 없습니다.")
            return []

        persona_id = int(cust_row.iloc[0].get('Persona_Cluster', -1))

        # [수정] 유효하지 않은 페르소나 클러스터면 빈 리스트 반환
        if persona_id not in self.PROFILES:
            logger.warning(f"고객 {cid}의 페르소나 클러스터 {persona_id}가 유효하지 않습니다.")
            return []

        profile = self.PROFILES[persona_id]
        return self._recommend_by_profile(profile, k)

    def recommend_by_persona_name(self, persona_name, k=5):
        """
        [수정] 페르소나 이름으로 직접 추천한다 (비회원용).
        유효하지 않은 페르소나면 빈 리스트를 반환한다.
        """
        profile = self._get_profile_by_name(persona_name)

        # [수정] 프로필을 찾지 못하면 빈 리스트 반환 (폴백 없음)
        if profile is None:
            logger.warning(f"페르소나 '{persona_name}'을 찾을 수 없습니다. 유효한 페르소나: {VALID_PERSONAS}")
            return []

        return self._recommend_by_profile(profile, k)

    def _recommend_by_profile(self, profile, k=5):
        """프로필 기반으로 상품을 추천한다."""
        results = []

        # 카드 추천
        if len(self.card_df) > 0 and 'card_category' in self.card_df.columns:
            card_scores = self.card_df.copy()
            card_scores['match_score'] = card_scores['card_category'].apply(
                lambda x: self._fuzzy_match(x, profile['card_keywords']))
            if 'product_name' in card_scores.columns:
                card_scores['name_score'] = card_scores['product_name'].apply(
                    lambda x: self._fuzzy_match(x, profile['keywords']))
                card_scores['total_score'] = card_scores['match_score'] * 0.7 + card_scores['name_score'] * 0.3
            else:
                card_scores['total_score'] = card_scores['match_score']
            for _, card in card_scores[card_scores['total_score'] > 0].nlargest(3, 'total_score').iterrows():
                results.append({'product_id': card['product_id'], 'product_name': card['product_name'],
                                'score': round(card['total_score'] * 5, 2), 'reason': f"{profile['name']} 맞춤 추천",
                                'source': 'zeroshot'})

        # 예금 추천
        if len(self.dep_df) > 0 and 'product_name' in self.dep_df.columns:
            dep_scores = self.dep_df.copy()
            dep_scores['match_score'] = dep_scores['product_name'].apply(
                lambda x: self._fuzzy_match(x, profile['dep_keywords']))
            for _, dep in dep_scores[dep_scores['match_score'] > 0].nlargest(2, 'match_score').iterrows():
                results.append({'product_id': dep['product_id'], 'product_name': dep['product_name'],
                                'score': round(dep['match_score'] * 4, 2), 'reason': f"{profile['name']} 맞춤 추천",
                                'source': 'zeroshot'})

        return results[:k]


def run_rule_engine(profile_name, intent, card_df, dep_df):
    """
    [수정] 룰 기반 추천 엔진.
    페르소나와 의도에 맞는 상품을 키워드 매칭으로 추천한다.
    유효하지 않은 페르소나면 빈 리스트를 반환한다.

    [문제2 해결] 구체적인 추천 이유를 함께 제공한다.
    """
    normalized_profile = normalize_persona_name(profile_name)

    if normalized_profile is None:
        logger.warning(f"룰 엔진: 페르소나 '{profile_name}'을 찾을 수 없습니다.")
        return []

    # 페르소나별, 의도별 추천 규칙 + 구체적 추천 이유
    RULES = {
        '실속 스타터': {
            'default': {
                'card': ['청년', '교통', '통신'],
                'deposit': ['적금', '청년'],
                'card_reasons': {
                    '청년': '사회초년생 전용 혜택으로 연회비 부담 없이 시작할 수 있어요',
                    '교통': '출퇴근 교통비 10% 할인, 월 10만원 사용 시 연 12만원 절약 가능',
                    '통신': '통신비 자동이체 할인으로 월 2-3천원 추가 절약'
                },
                'deposit_reasons': {
                    '적금': '매월 꾸준히 저축하는 습관 형성 + 목돈 마련의 첫걸음',
                    '청년': '청년 우대금리 적용으로 일반 상품 대비 최대 1%p 높은 금리'
                }
            },
            '여행': {
                'card': ['항공', '여행'],
                'deposit': ['여행', '적금'],
                'card_reasons': {
                    '항공': '마일리지 적립으로 연 1-2회 여행 시 항공권 할인 혜택',
                    '여행': '해외 결제 수수료 면제 + 여행자 보험 무료 제공'
                },
                'deposit_reasons': {
                    '여행': '목표 금액 설정으로 여행 자금 체계적 마련',
                    '적금': '여행 목표와 저축 습관을 동시에 잡는 일석이조'
                }
            },
            '저축': {
                'card': ['캐시백'],
                'deposit': ['적금', '정기'],
                'card_reasons': {
                    '캐시백': '소비할 때마다 캐시백 적립, 이를 적금으로 자동 연결 가능'
                },
                'deposit_reasons': {
                    '적금': '월 50만원씩 1년이면 600만원 + 이자, 첫 목돈의 시작',
                    '정기': '확정 금리로 안정적인 수익 보장'
                }
            }
        },
        '스마트 플렉서': {
            'default': {
                'card': ['프리미엄', '여행', '쇼핑'],
                'deposit': ['고금리', '예금'],
                'card_reasons': {
                    '프리미엄': '라운지 이용, 발렛파킹 등 프리미엄 서비스로 스마트한 소비',
                    '여행': '해외 결제 1.5% 적립, 연 2회 여행 시 10만원 이상 혜택',
                    '쇼핑': '백화점/명품 5% 할인으로 100만원 쇼핑 시 5만원 절약'
                },
                'deposit_reasons': {
                    '고금리': '높은 금리로 여유 자금 불리기, 다음 여행 자금 마련',
                    '예금': '목돈을 안전하게 굴리면서 여행 계획 세우기'
                }
            },
            '여행': {
                'card': ['항공', 'VIP'],
                'deposit': ['외화'],
                'card_reasons': {
                    '항공': '마일리지 2배 적립으로 연 1회 무료 항공권 가능',
                    'VIP': '공항 라운지 무료 이용으로 여행의 품격 UP'
                },
                'deposit_reasons': {
                    '외화': '환율 유리할 때 미리 환전, 여행 경비 절약'
                }
            },
            '쇼핑': {
                'card': ['쇼핑', '백화점'],
                'deposit': ['자유'],
                'card_reasons': {
                    '쇼핑': '온라인 쇼핑 5% 할인 + 무이자 할부 혜택',
                    '백화점': '백화점 VIP 등급 연계 추가 할인'
                },
                'deposit_reasons': {
                    '자유': '입출금 자유로워서 쇼핑 자금 유동적 관리'
                }
            }
        },
        '알뜰 지킴이': {
            'default': {
                'card': ['캐시백', '마트', '생활'],
                'deposit': ['적금', '예금'],
                'card_reasons': {
                    '캐시백': '모든 소비에서 캐시백, 티끌 모아 태산 전략',
                    '마트': '대형마트 5% 할인, 월 20만원 장보기 시 연 12만원 절약',
                    '생활': '공과금/통신비 할인으로 고정비 절감'
                },
                'deposit_reasons': {
                    '적금': '절약한 금액을 적금으로 자동 이체, 저축 습관화',
                    '예금': '비상금 통장으로 안전하게 보관'
                }
            },
            '저축': {
                'card': ['적립'],
                'deposit': ['정기적금'],
                'card_reasons': {
                    '적립': '포인트 적립 후 현금 전환, 추가 저축 재원 마련'
                },
                'deposit_reasons': {
                    '정기적금': '매월 정해진 금액 자동이체로 강제 저축 효과'
                }
            }
        },
        '디지털 힙스터': {
            'default': {
                'card': ['온라인', '쇼핑', '구독'],
                'deposit': ['모바일', '입출금'],
                'card_reasons': {
                    '온라인': '온라인 쇼핑 7% 할인, 월 15만원 사용 시 연 12.6만원 절약',
                    '쇼핑': '간편결제 추가 적립으로 포인트 이중 혜택',
                    '구독': '넷플릭스/유튜브 등 구독료 10% 할인'
                },
                'deposit_reasons': {
                    '모바일': '앱으로 간편하게 관리, 실시간 알림으로 소비 패턴 확인',
                    '입출금': '수시 입출금으로 유연한 자금 관리'
                }
            },
            '구독': {
                'card': ['스트리밍'],
                'deposit': ['자유'],
                'card_reasons': {
                    '스트리밍': 'OTT 구독료 15% 할인, 월 3만원 구독 시 연 5.4만원 절약'
                },
                'deposit_reasons': {
                    '자유': '구독료 자동이체 + 나머지 금액 저축 자동화'
                }
            }
        },
        '밸런스 메인스트림': {
            'default': {
                'card': ['일상', '생활'],
                'deposit': ['예금', '자유'],
                'card_reasons': {
                    '일상': '전 가맹점 1% 적립, 복잡한 조건 없이 심플하게',
                    '생활': '점심값+커피값 월 30만원 사용 시 연 3.6만원 적립'
                },
                'deposit_reasons': {
                    '예금': '연회비 없이 안정적인 이자 수익',
                    '자유': '필요할 때 언제든 출금 가능한 유연성'
                }
            }
        }
    }

    # 의도 감지
    intent_map = {
        '여행': ['여행', '해외', '항공'],
        '저축': ['저축', '적금', '목돈'],
        '쇼핑': ['쇼핑', '백화점'],
        '구독': ['구독', '넷플릭스']
    }
    detected_intent = 'default'
    intent_lower = intent.lower() if intent else ''
    for key, keywords in intent_map.items():
        if any(kw in intent_lower for kw in keywords):
            detected_intent = key
            break

    profile_rules = RULES.get(normalized_profile)
    if profile_rules is None:
        logger.warning(f"룰 엔진: '{normalized_profile}' 페르소나에 대한 규칙이 없습니다.")
        return []

    rule = profile_rules.get(detected_intent, profile_rules.get('default', {}))

    # 추천 이유 딕셔너리 가져오기
    card_reasons = rule.get('card_reasons', {})
    deposit_reasons = rule.get('deposit_reasons', {})

    results = []

    # 카드 추천 + 구체적 이유
    if len(card_df) > 0 and 'card_category' in card_df.columns:
        for kw in rule.get('card', []):
            matches = card_df[card_df['card_category'].str.contains(kw, case=False, na=False)]
            for _, r in matches.head(1).iterrows():
                specific_reason = card_reasons.get(kw, f"'{kw}' 관련 혜택 제공")
                results.append({
                    'product_id': r['product_id'],
                    'product_name': r['product_name'],
                    'reason': specific_reason,
                    'keyword': kw,
                    'persona_fit': f"{normalized_profile} 성향에 적합",
                    'source': 'rule'
                })

    # 예금 추천 + 구체적 이유
    if len(dep_df) > 0 and 'product_name' in dep_df.columns:
        for kw in rule.get('deposit', []):
            matches = dep_df[dep_df['product_name'].str.contains(kw, case=False, na=False)]
            for _, r in matches.head(1).iterrows():
                specific_reason = deposit_reasons.get(kw, f"'{kw}' 관련 상품")
                results.append({
                    'product_id': r['product_id'],
                    'product_name': r['product_name'],
                    'reason': specific_reason,
                    'keyword': kw,
                    'persona_fit': f"{normalized_profile} 성향에 적합",
                    'source': 'rule'
                })

    # 중복 제거
    seen = set()
    return [r for r in results if not (r['product_id'] in seen or seen.add(r['product_id']))][:5]


# -----------------------------------------
# 소비 습관 교정 엔진 (문제 3 해결)
# -----------------------------------------

class SpendingAnalyzer:
    """
    거래 데이터를 분석하여 소비 패턴을 진단하는 클래스.
    - 카테고리별 지출 분석
    - 과소비 카테고리 식별
    - 시간대별/요일별 소비 패턴 분석
    - 월급날 전후 소비 패턴 분석
    """

    # 카테고리별 권장 지출 비율 (월 소득 대비)
    RECOMMENDED_RATIOS = {
        '식비': 0.15,  # 15%
        '교통': 0.10,  # 10%
        '통신': 0.05,  # 5%
        '쇼핑': 0.10,  # 10%
        '여가/문화': 0.10,  # 10%
        '배달': 0.05,  # 5%
        '카페/음료': 0.03,  # 3%
        '편의점': 0.03,  # 3%
        '구독서비스': 0.03,  # 3%
        '기타': 0.10  # 10%
    }

    # 카테고리 매핑 (원본 데이터 -> 통합 카테고리)
    CATEGORY_MAPPING = {
        '식당/카페': '식비',
        '카페': '카페/음료',
        '스타벅스': '카페/음료',
        '편의점': '편의점',
        '마트': '식비',
        '배달': '배달',
        '배달앱': '배달',
        '교통': '교통',
        '대중교통': '교통',
        '택시': '교통',
        '주유': '교통',
        '쇼핑': '쇼핑',
        '온라인쇼핑': '쇼핑',
        '의류': '쇼핑',
        '통신': '통신',
        '구독': '구독서비스',
        'OTT': '구독서비스',
        '여가': '여가/문화',
        '문화': '여가/문화',
        '영화': '여가/문화',
        '게임': '여가/문화'
    }

    def __init__(self, transactions_df, customers_df):
        self.trans_df = transactions_df.copy() if len(transactions_df) > 0 else pd.DataFrame()
        self.cust_df = customers_df

        if len(self.trans_df) > 0:
            self.trans_df['transaction_date'] = pd.to_datetime(
                self.trans_df['transaction_date'], errors='coerce'
            )
            self.trans_df = self.trans_df.dropna(subset=['transaction_date'])
            self.trans_df['day_of_week'] = self.trans_df['transaction_date'].dt.dayofweek
            self.trans_df['hour'] = self.trans_df['transaction_date'].dt.hour
            self.trans_df['day'] = self.trans_df['transaction_date'].dt.day

            # 카테고리 통합
            if 'merchant_category' in self.trans_df.columns:
                self.trans_df['unified_category'] = self.trans_df['merchant_category'].apply(
                    lambda x: self._map_category(x)
                )

    def _map_category(self, category):
        """원본 카테고리를 통합 카테고리로 매핑한다."""
        if pd.isna(category):
            return '기타'
        category = str(category)
        for key, value in self.CATEGORY_MAPPING.items():
            if key in category:
                return value
        return '기타'

    def analyze_customer(self, cid, estimated_income=2500000):
        """
        고객의 소비 패턴을 종합 분석한다.

        Args:
            cid: 고객 ID
            estimated_income: 추정 월 소득 (기본값 250만원)

        Returns:
            분석 결과 딕셔너리
        """
        if len(self.trans_df) == 0:
            return {"status": "error", "message": "거래 데이터가 없습니다."}

        cust_trans = self.trans_df[self.trans_df['customer_id'] == cid]

        if len(cust_trans) == 0:
            return {"status": "error", "message": f"고객 {cid}의 거래 내역이 없습니다."}

        # 최근 3개월 데이터만 사용
        latest_date = cust_trans['transaction_date'].max()
        three_months_ago = latest_date - timedelta(days=90)
        recent_trans = cust_trans[cust_trans['transaction_date'] >= three_months_ago]

        if len(recent_trans) == 0:
            return {"status": "error", "message": "최근 3개월간 거래 내역이 없습니다."}

        # 월평균 지출 계산
        months_count = max(1, (latest_date - recent_trans['transaction_date'].min()).days / 30)
        total_spending = recent_trans['amount'].sum()
        monthly_avg = total_spending / months_count

        # 카테고리별 분석
        category_analysis = self._analyze_by_category(recent_trans, monthly_avg, estimated_income)

        # 시간대별 분석
        time_analysis = self._analyze_by_time(recent_trans)

        # 요일별 분석
        weekday_analysis = self._analyze_by_weekday(recent_trans)

        # 월급날 분석 (25일 기준)
        payday_analysis = self._analyze_payday_pattern(recent_trans)

        # 충동소비 패턴 감지
        impulse_analysis = self._detect_impulse_spending(recent_trans)

        return {
            "status": "success",
            "customer_id": cid,
            "analysis_period": f"{three_months_ago.strftime('%Y-%m-%d')} ~ {latest_date.strftime('%Y-%m-%d')}",
            "monthly_average_spending": round(monthly_avg),
            "estimated_income": estimated_income,
            "spending_ratio": round(monthly_avg / estimated_income * 100, 1),
            "category_analysis": category_analysis,
            "time_analysis": time_analysis,
            "weekday_analysis": weekday_analysis,
            "payday_analysis": payday_analysis,
            "impulse_analysis": impulse_analysis,
            "transaction_count": len(recent_trans)
        }

    def _analyze_by_category(self, trans, monthly_avg, income):
        """카테고리별 지출 분석 및 과소비 감지."""
        if 'unified_category' not in trans.columns:
            return {"status": "카테고리 정보 없음"}

        # 월 기준으로 환산
        months = max(1, (trans['transaction_date'].max() - trans['transaction_date'].min()).days / 30)

        category_spending = trans.groupby('unified_category')['amount'].sum() / months
        category_counts = trans.groupby('unified_category').size() / months

        results = []
        overspending_categories = []

        for cat, amount in category_spending.items():
            recommended_ratio = self.RECOMMENDED_RATIOS.get(cat, 0.05)
            recommended_amount = income * recommended_ratio
            actual_ratio = amount / income

            status = "적정"
            if actual_ratio > recommended_ratio * 1.5:
                status = "과소비"
                overspending_categories.append({
                    "category": cat,
                    "monthly_amount": round(amount),
                    "recommended": round(recommended_amount),
                    "excess": round(amount - recommended_amount),
                    "ratio": round(actual_ratio * 100, 1)
                })
            elif actual_ratio > recommended_ratio * 1.2:
                status = "주의"

            results.append({
                "category": cat,
                "monthly_amount": round(amount),
                "monthly_count": round(category_counts.get(cat, 0), 1),
                "recommended": round(recommended_amount),
                "actual_ratio": round(actual_ratio * 100, 1),
                "recommended_ratio": round(recommended_ratio * 100, 1),
                "status": status
            })

        # 금액 순으로 정렬
        results.sort(key=lambda x: x['monthly_amount'], reverse=True)

        return {
            "details": results,
            "overspending": overspending_categories,
            "overspending_total": sum(item['excess'] for item in overspending_categories)
        }

    def _analyze_by_time(self, trans):
        """시간대별 소비 패턴 분석."""
        if 'hour' not in trans.columns:
            return {}

        hourly = trans.groupby('hour')['amount'].agg(['sum', 'count'])

        # 피크 시간대 찾기
        peak_hour = hourly['sum'].idxmax()

        # 심야 소비 (23시 ~ 4시)
        late_night_mask = trans['hour'].isin([23, 0, 1, 2, 3, 4])
        late_night_spending = trans[late_night_mask]['amount'].sum()
        late_night_ratio = late_night_spending / trans['amount'].sum() * 100 if trans['amount'].sum() > 0 else 0

        return {
            "peak_hour": int(peak_hour),
            "peak_hour_amount": round(hourly.loc[peak_hour, 'sum']),
            "late_night_spending": round(late_night_spending),
            "late_night_ratio": round(late_night_ratio, 1),
            "late_night_warning": late_night_ratio > 10
        }

    def _analyze_by_weekday(self, trans):
        """요일별 소비 패턴 분석."""
        if 'day_of_week' not in trans.columns:
            return {}

        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        daily = trans.groupby('day_of_week')['amount'].agg(['sum', 'count'])

        # 주중(월-금) vs 주말(토-일) 비교
        weekday_total = daily.loc[0:4, 'sum'].sum() if len(daily) > 0 else 0
        weekend_total = daily.loc[5:6, 'sum'].sum() if len(daily) > 0 else 0

        weekend_ratio = weekend_total / (weekday_total + weekend_total) * 100 if (
                                                                                             weekday_total + weekend_total) > 0 else 0

        # 가장 많이 쓰는 요일
        peak_day = daily['sum'].idxmax() if len(daily) > 0 else 0

        return {
            "peak_day": weekday_names[int(peak_day)],
            "peak_day_amount": round(daily.loc[peak_day, 'sum']) if len(daily) > 0 else 0,
            "weekday_total": round(weekday_total),
            "weekend_total": round(weekend_total),
            "weekend_ratio": round(weekend_ratio, 1),
            "weekend_heavy": weekend_ratio > 40
        }

    def _analyze_payday_pattern(self, trans, payday=25):
        """월급날(기본 25일) 전후 소비 패턴 분석."""
        if 'day' not in trans.columns:
            return {}

        # 월급 직후 (25일 ~ 말일, 1일 ~ 5일)
        post_payday_mask = (trans['day'] >= payday) | (trans['day'] <= 5)
        post_payday_spending = trans[post_payday_mask]['amount'].sum()

        # 월급 전 (20일 ~ 24일)
        pre_payday_mask = (trans['day'] >= 20) & (trans['day'] < payday)
        pre_payday_spending = trans[pre_payday_mask]['amount'].sum()

        total = trans['amount'].sum()
        post_ratio = post_payday_spending / total * 100 if total > 0 else 0

        return {
            "post_payday_spending": round(post_payday_spending),
            "pre_payday_spending": round(pre_payday_spending),
            "post_payday_ratio": round(post_ratio, 1),
            "payday_spike": post_ratio > 50,
            "message": "월급 직후 과소비 경향이 있습니다." if post_ratio > 50 else "소비가 비교적 균등하게 분포되어 있습니다."
        }

    def _detect_impulse_spending(self, trans, threshold_multiplier=3):
        """충동소비 패턴 감지 (평균의 3배 이상 거래)."""
        if len(trans) == 0:
            return {}

        avg_transaction = trans['amount'].mean()
        std_transaction = trans['amount'].std()

        # 이상치 거래 (평균 + 3*표준편차 이상)
        threshold = avg_transaction + threshold_multiplier * std_transaction
        impulse_trans = trans[trans['amount'] > threshold]

        impulse_details = []
        for _, row in impulse_trans.head(5).iterrows():
            impulse_details.append({
                "date": row['transaction_date'].strftime('%Y-%m-%d'),
                "amount": round(row['amount']),
                "category": row.get('unified_category', '기타')
            })

        return {
            "average_transaction": round(avg_transaction),
            "impulse_threshold": round(threshold),
            "impulse_count": len(impulse_trans),
            "impulse_total": round(impulse_trans['amount'].sum()),
            "impulse_examples": impulse_details,
            "has_impulse_pattern": len(impulse_trans) > 3
        }


class HabitCoach:
    """
    소비 습관 교정 조언을 생성하는 클래스.
    SpendingAnalyzer의 분석 결과를 바탕으로 구체적인 절약 방안과 목표를 제시한다.
    """

    # 카테고리별 절약 팁
    SAVING_TIPS = {
        '배달': {
            'tips': [
                "주 3회 배달을 2회로 줄이면 월 4-5만원 절약",
                "배달 대신 편의점 도시락 활용 (1회당 5천원 절약)",
                "배달앱 알림 끄기로 충동 주문 방지",
                "직접 픽업 시 할인 혜택 활용"
            ],
            'saving_potential': 0.3  # 30% 절약 가능
        },
        '카페/음료': {
            'tips': [
                "텀블러 지참 시 300-500원 할인",
                "사무실 커피머신 활용으로 1잔당 3천원 절약",
                "주 5회를 3회로 줄이면 월 2-3만원 절약",
                "구독권/스탬프 활용으로 무료 음료 적립"
            ],
            'saving_potential': 0.4
        },
        '편의점': {
            'tips': [
                "1+1, 2+1 행사 상품 위주로 구매",
                "도시락은 마감 할인 시간대 활용",
                "편의점 대신 대형마트 소포장 활용",
                "충동구매 방지: 필요한 것만 메모 후 방문"
            ],
            'saving_potential': 0.25
        },
        '구독서비스': {
            'tips': [
                "사용하지 않는 구독 서비스 정리 (월 1-2만원 절약)",
                "가족/친구와 공유 계정 활용",
                "연간 결제로 할인 혜택 받기",
                "무료 체험 기간 활용 후 결정"
            ],
            'saving_potential': 0.5
        },
        '쇼핑': {
            'tips': [
                "장바구니 담기 후 24시간 대기 규칙",
                "세일 기간 집중 구매로 20-30% 절약",
                "위시리스트 작성 후 우선순위 구매",
                "중고거래 플랫폼 활용"
            ],
            'saving_potential': 0.3
        },
        '교통': {
            'tips': [
                "정기권 구매로 월 10-15% 절약",
                "자전거/도보 출퇴근 가능 거리 활용",
                "카풀 서비스 활용",
                "택시 대신 대중교통 (1회당 1-2만원 절약)"
            ],
            'saving_potential': 0.15
        },
        '여가/문화': {
            'tips': [
                "조조/심야 할인 영화 관람",
                "문화누리카드/청년 할인 활용",
                "무료 문화행사/전시회 활용",
                "스트리밍 서비스로 대체"
            ],
            'saving_potential': 0.25
        }
    }

    # 목표별 필요 저축액 예시
    GOAL_EXAMPLES = {
        '해외여행': {'amount': 2000000, 'description': '동남아 1주일 여행'},
        '유럽여행': {'amount': 4000000, 'description': '유럽 2주 여행'},
        '자동차': {'amount': 5000000, 'description': '중고차 구매 자금'},
        '전세자금': {'amount': 10000000, 'description': '전세 보증금 일부'},
        '결혼자금': {'amount': 20000000, 'description': '결혼 준비 자금'},
        '비상금': {'amount': 3000000, 'description': '3개월치 생활비 비상금'}
    }

    def __init__(self, analyzer: SpendingAnalyzer):
        self.analyzer = analyzer

    def generate_saving_plan(self, cid, goal_name=None, goal_amount=None, target_months=12):
        """
        고객 맞춤 저축 계획을 생성한다.

        Args:
            cid: 고객 ID
            goal_name: 목표 이름 (예: '해외여행', '비상금')
            goal_amount: 목표 금액 (없으면 goal_name에서 자동 설정)
            target_months: 목표 달성 기간 (개월)

        Returns:
            저축 계획 딕셔너리
        """
        # 소비 분석 결과 가져오기
        analysis = self.analyzer.analyze_customer(cid)

        if analysis.get('status') == 'error':
            return analysis

        # 목표 설정
        if goal_amount is None and goal_name:
            goal_info = self.GOAL_EXAMPLES.get(goal_name, self.GOAL_EXAMPLES['비상금'])
            goal_amount = goal_info['amount']
            goal_description = goal_info['description']
        else:
            goal_amount = goal_amount or 3000000
            goal_description = goal_name or '목표 저축'

        monthly_target = goal_amount / target_months

        # 절약 가능 금액 계산
        saving_opportunities = self._calculate_saving_opportunities(analysis)

        # 구체적인 실천 방안 생성
        action_plan = self._generate_action_plan(analysis, saving_opportunities, monthly_target)

        # 진행 상황 시뮬레이션
        progress_simulation = self._simulate_progress(
            monthly_target,
            saving_opportunities['total_potential'],
            target_months
        )

        return {
            "status": "success",
            "customer_id": cid,
            "goal": {
                "name": goal_description,
                "amount": goal_amount,
                "target_months": target_months,
                "monthly_target": round(monthly_target)
            },
            "current_spending": {
                "monthly_average": analysis['monthly_average_spending'],
                "spending_ratio": analysis['spending_ratio']
            },
            "saving_opportunities": saving_opportunities,
            "action_plan": action_plan,
            "progress_simulation": progress_simulation,
            "motivation_message": self._generate_motivation_message(
                goal_description, goal_amount, saving_opportunities['total_potential'], target_months
            )
        }

    def _calculate_saving_opportunities(self, analysis):
        """카테고리별 절약 가능 금액을 계산한다."""
        opportunities = []
        total_potential = 0

        category_data = analysis.get('category_analysis', {})
        overspending = category_data.get('overspending', [])

        for item in overspending:
            cat = item['category']
            tips_info = self.SAVING_TIPS.get(cat, {'tips': [], 'saving_potential': 0.2})

            potential_saving = round(item['excess'] * tips_info['saving_potential'])
            total_potential += potential_saving

            opportunities.append({
                "category": cat,
                "current_monthly": item['monthly_amount'],
                "excess_amount": item['excess'],
                "potential_saving": potential_saving,
                "saving_tips": tips_info['tips'][:2],  # 상위 2개 팁만
                "difficulty": "쉬움" if tips_info['saving_potential'] >= 0.3 else "보통"
            })

        # 절약 효과가 큰 순으로 정렬
        opportunities.sort(key=lambda x: x['potential_saving'], reverse=True)

        return {
            "details": opportunities,
            "total_potential": total_potential,
            "top_3_categories": [o['category'] for o in opportunities[:3]]
        }

    def _generate_action_plan(self, analysis, saving_opportunities, monthly_target):
        """구체적인 주간/월간 실천 계획을 생성한다."""
        actions = []

        # 과소비 카테고리별 행동 계획
        for opp in saving_opportunities['details'][:3]:
            cat = opp['category']
            actions.append({
                "category": cat,
                "weekly_action": self._get_weekly_action(cat, opp['current_monthly']),
                "monthly_goal": f"{cat} 지출 {opp['potential_saving']:,}원 줄이기",
                "tracking_method": f"매주 {cat} 지출 내역 확인하기"
            })

        # 시간대별 조언
        time_analysis = analysis.get('time_analysis', {})
        if time_analysis.get('late_night_warning'):
            actions.append({
                "category": "심야 소비",
                "weekly_action": "밤 11시 이후 결제 앱 알림 끄기",
                "monthly_goal": f"심야 소비 {time_analysis['late_night_ratio']}% → 5% 이하로",
                "tracking_method": "심야 거래 내역 주간 점검"
            })

        # 월급날 패턴 조언
        payday = analysis.get('payday_analysis', {})
        if payday.get('payday_spike'):
            actions.append({
                "category": "월급날 관리",
                "weekly_action": "월급날 자동이체로 저축 먼저 빼기",
                "monthly_goal": "월급 직후 3일간 지출 30% 줄이기",
                "tracking_method": "월급 후 일주일 지출 별도 기록"
            })

        return {
            "priority_actions": actions,
            "automation_suggestion": "월급날 자동이체 설정: 저축 → 고정비 → 생활비 순서",
            "review_schedule": "매주 일요일 저녁, 주간 소비 리뷰 시간 갖기"
        }

    def _get_weekly_action(self, category, monthly_amount):
        """카테고리별 주간 실천 행동을 반환한다."""
        weekly_amount = monthly_amount / 4

        actions = {
            '배달': f"이번 주 배달 {int(weekly_amount / 15000)}회 → {max(1, int(weekly_amount / 15000) - 1)}회로 줄이기",
            '카페/음료': f"주 {int(weekly_amount / 5000)}잔 → {max(2, int(weekly_amount / 5000) - 2)}잔으로 줄이기",
            '편의점': f"편의점 방문 시 5천원 이하만 지출하기",
            '쇼핑': f"이번 주 쇼핑 예산 {int(weekly_amount * 0.7):,}원으로 제한",
            '구독서비스': "사용하지 않는 구독 1개 해지하기",
            '여가/문화': f"무료 문화행사 1개 참여하기"
        }

        return actions.get(category, f"주간 {category} 예산 {int(weekly_amount * 0.8):,}원으로 관리")

    def _simulate_progress(self, monthly_target, monthly_saving, months):
        """목표 달성 진행 상황을 시뮬레이션한다."""
        progress = []
        cumulative = 0

        for month in range(1, months + 1):
            cumulative += monthly_saving
            target_cumulative = monthly_target * month

            progress.append({
                "month": month,
                "saved": round(cumulative),
                "target": round(target_cumulative),
                "achievement_rate": round(cumulative / target_cumulative * 100, 1) if target_cumulative > 0 else 0
            })

        # 목표 달성 예상 시점
        if monthly_saving > 0:
            months_to_goal = (monthly_target * months) / monthly_saving
            achievable = months_to_goal <= months
        else:
            months_to_goal = float('inf')
            achievable = False

        return {
            "monthly_progress": progress[:6],  # 처음 6개월만
            "final_amount": round(monthly_saving * months),
            "target_amount": round(monthly_target * months),
            "achievable": achievable,
            "months_to_goal": round(months_to_goal, 1) if months_to_goal != float('inf') else "달성 어려움"
        }

    def _generate_motivation_message(self, goal_name, goal_amount, monthly_saving, months):
        """동기부여 메시지를 생성한다."""
        if monthly_saving <= 0:
            return "현재 소비 패턴에서 절약 가능한 부분을 찾지 못했어요. 소비 내역을 더 자세히 살펴볼까요?"

        total_savable = monthly_saving * months
        achievement_rate = total_savable / goal_amount * 100

        if achievement_rate >= 100:
            return f"지금 절약 습관만 들이면 {months}개월 후 '{goal_name}' 목표 달성 가능해요! 매월 {monthly_saving:,}원씩 모으면 목표 금액 {goal_amount:,}원을 넘길 수 있어요."
        elif achievement_rate >= 70:
            return f"절약 실천으로 목표의 {achievement_rate:.0f}%인 {total_savable:,}원을 모을 수 있어요. '{goal_name}'까지 거의 다 왔어요!"
        else:
            extra_needed = (goal_amount - total_savable) / months
            return f"현재 계획으로는 {total_savable:,}원을 모을 수 있어요. 목표 달성을 위해 월 {extra_needed:,.0f}원 추가 저축이 필요해요. 작은 습관부터 시작해봐요!"

    def get_quick_tips(self, cid):
        """빠른 절약 팁을 제공한다 (분석 없이 일반적인 조언)."""
        analysis = self.analyzer.analyze_customer(cid)

        if analysis.get('status') == 'error':
            return {
                "status": "error",
                "tips": [
                    "배달 횟수 주 1회 줄이기: 월 6만원 절약",
                    "커피 텀블러 지참하기: 월 1만원 절약",
                    "구독 서비스 정리하기: 월 1-2만원 절약"
                ]
            }

        tips = []
        overspending = analysis.get('category_analysis', {}).get('overspending', [])

        for item in overspending[:3]:
            cat = item['category']
            tips_info = self.SAVING_TIPS.get(cat, {'tips': ['지출 내역 점검하기']})
            tips.append({
                "category": cat,
                "current": f"월 {item['monthly_amount']:,}원",
                "tip": tips_info['tips'][0],
                "potential": f"월 {int(item['excess'] * 0.3):,}원 절약 가능"
            })

        return {
            "status": "success",
            "quick_tips": tips,
            "total_potential": sum(int(item['excess'] * 0.3) for item in overspending[:3])
        }


class HybridEngine:
    """
    하이브리드 추천 엔진.
    여러 추천 전략(KNN, 로그, 만족도, 제로샷)을 결합하여 최종 추천을 생성한다.
    [문제3] 소비 습관 분석 및 교정 기능 추가.
    """
    # 추천 소스별 가중치
    WEIGHTS = {'satisfaction': 0.4, 'knn': 0.35, 'log': 0.25}

    def __init__(self, data):
        self.knn = FirstFinKNNRecommender(data.get('customers_train', pd.DataFrame()))
        self.zero = ZeroShotRecommender(data['customers'], data['card'], data['deposit'], data['logs'],
                                        data['satisfaction'])
        self.log = LogRecommender(data['logs'], data['card'], data['deposit'])
        self.sat = SatisfactionRecommender(data['satisfaction'], data['card'], data['deposit'])
        self.raw_customers = data.get('customers', pd.DataFrame())
        self.tom_df = data.get('customers_train', pd.DataFrame())
        self.card_df = data['card']
        self.dep_df = data['deposit']

        # [문제3] 소비 분석 엔진 초기화
        trans_path = DATA_PATH + "card_transactions_updated.csv"
        if os.path.exists(trans_path):
            trans_df = pd.read_csv(trans_path)
        else:
            trans_df = pd.DataFrame()

        self.spending_analyzer = SpendingAnalyzer(trans_df, data['customers'])
        self.habit_coach = HabitCoach(self.spending_analyzer)

    def get_tom_profile(self, cid):
        """고객의 TOM 지표 프로필을 조회한다."""
        if self.tom_df.empty:
            return {"status": "데이터 없음"}
        try:
            if 'customer_id' not in self.tom_df.columns:
                return {"status": "데이터 구조 오류"}
            tom_indexed = self.tom_df.set_index('customer_id')
            if cid not in tom_indexed.index:
                return {"status": "고객 ID 없음"}
            row = tom_indexed.loc[cid]
            trend_raw = row.get('TOM_Trend_Raw', row.get('TOM_Trend', 0))
            return {
                "Trend": f"{trend_raw:.1%}",
                "YOLO": f"{row.get('TOM_YOLO', 0):.2f}",
                "Digital": f"{row.get('TOM_Digital', 0):.2f}",
                "Weekend": f"{row.get('TOM_Weekend', 0):.2f}"
            }
        except Exception as e:
            logger.warning(f"TOM 프로필 조회 실패: {e}")
            return {"status": "조회 실패"}

    def get_persona_name(self, cid):
        """
        [수정] 고객의 페르소나 이름을 조회한다.
        고객 데이터가 없으면 None을 반환한다 (기존: 실속 스타터 반환).
        """
        if self.raw_customers.empty:
            return None
        row = self.raw_customers[self.raw_customers['customer_id'] == cid]
        if row.empty:
            return None

        persona_cluster = row.iloc[0].get('Persona_Cluster', -1)
        base = self.zero.PROFILES.get(int(persona_cluster), {}).get('name')

        if base is None:
            return None

        # TOM 지표 기반 페르소나 수정
        if not self.tom_df.empty and 'customer_id' in self.tom_df.columns:
            tom_indexed = self.tom_df.set_index('customer_id')
            if cid in tom_indexed.index:
                t = tom_indexed.loc[cid]
                if t.get('TOM_YOLO', 0) > 0.7:
                    return f"스마트 플렉서 (최근 소비 급증)"
                trend_raw = t.get('TOM_Trend_Raw', t.get('TOM_Trend', 0))
                if trend_raw < -0.1:
                    return f"알뜰 지킴이 (절약 모드)"
        return base

    def recommend(self, cid, k=3):
        """
        [수정] 기존 회원용 하이브리드 추천.
        콜드 스타트 시 제로샷 추천, 아니면 로그+만족도 결합 추천.
        """
        if self.zero.is_cold(cid):
            recs = self.zero.recommend(cid, k)
            # [수정] 추천이 비어있으면 명확한 메시지 반환
            if not recs:
                return {
                    'recs': [],
                    'is_cold': True,
                    'ctx': {'log_sum': "고객 정보를 찾을 수 없습니다. 고객 ID를 확인해주세요."}
                }
            return {'recs': recs, 'is_cold': True, 'ctx': {'log_sum': "신규 고객 - 기본 페르소나 기반 추천"}}

        similar = self.knn.get_similar(cid)
        log_recs = self.log.recommend(cid, k=10)
        sat_recs = self.sat.recommend(cid, similar, k=10)

        # 점수 병합
        merged = {}
        for r in log_recs:
            merged[r['product_id']] = {'info': r, 'score': r['score'] * self.WEIGHTS['log']}
        for r in sat_recs:
            if r['product_id'] in merged:
                merged[r['product_id']]['score'] += r['score'] * self.WEIGHTS['satisfaction']
            else:
                merged[r['product_id']] = {'info': r, 'score': r['score'] * self.WEIGHTS['satisfaction']}

        final = sorted(merged.values(), key=lambda x: x['score'], reverse=True)[:k]
        return {'recs': [f['info'] for f in final], 'is_cold': False, 'ctx': {'log_sum': self.log.summary(cid)}}

    def recommend_guest(self, persona_name, k=3):
        """
        [수정] 비회원용 페르소나 기반 추천.
        유효하지 않은 페르소나면 에러 메시지를 반환한다.
        """
        # [수정] 페르소나 유효성 검증
        if not validate_persona(persona_name):
            return {
                'recs': [],
                'is_cold': True,
                'ctx': {'log_sum': f"'{persona_name}'은 유효하지 않은 페르소나입니다. 유효한 페르소나: {', '.join(VALID_PERSONAS)}"}
            }

        recs = self.zero.recommend_by_persona_name(persona_name, k)
        normalized = normalize_persona_name(persona_name)
        return {
            'recs': recs,
            'is_cold': True,
            'ctx': {'log_sum': f"비회원 - '{normalized}' 페르소나 기반 추천"}
        }


# -----------------------------------------
# 7. Streamlit UI
# -----------------------------------------
st.title("FirstFin - 사회초년생을 위한 은행 상품 추천 Agent")
st.markdown("**TOM(Time-Occasion-Method)** 및 **Lifestyle** 기반 하이브리드 추천 시스템")

# 데이터 로드
data = load_all_data()

# OpenAI 클라이언트 초기화
client = get_openai_client()

# 상품 DB 생성
product_db = build_product_db(data)

# FAISS 인덱스 생성
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
    if client is not None and len(product_db) > 0:
        with st.spinner("임베딩 인덱스 생성 중... (최초 1회)"):
            st.session_state.faiss_index = build_faiss_index(product_db, client)

index = st.session_state.faiss_index

# 엔진 초기화
engine = HybridEngine(data)


# -----------------------------------------
# 8. Tool 함수
# -----------------------------------------
def validate_tool_args(fn_name, args):
    """Tool 함수 인자의 유효성을 검증한다."""
    validators = {
        'run_hybrid': lambda a: 'cid' in a and isinstance(a.get('cid'), str),
        'run_rule': lambda a: 'profile' in a and 'intent' in a,
        'get_details': lambda a: 'pids' in a and isinstance(a.get('pids'), (list, str)),
        'search_info': lambda a: 'query' in a and isinstance(a.get('query'), str),
        # [문제3] 새로운 Tool 검증 추가
        'analyze_spending': lambda a: 'cid' in a and isinstance(a.get('cid'), str),
        'get_saving_plan': lambda a: 'cid' in a and isinstance(a.get('cid'), str),
        'get_quick_saving_tips': lambda a: 'cid' in a and isinstance(a.get('cid'), str)
    }
    validator = validators.get(fn_name)
    if validator is None:
        return False, f"알 수 없는 함수: {fn_name}"
    if not validator(args):
        return False, f"잘못된 인자: {fn_name}({args})"
    return True, None


def run_hybrid(cid, intent=""):
    """기존 회원용 하이브리드 추천을 실행한다."""
    try:
        r = engine.recommend(cid, 3)
        return json.dumps({"recommendations": r['recs'], "context": r['ctx'], "is_cold_start": r['is_cold']},
                          ensure_ascii=False)
    except Exception as e:
        logger.error(f"하이브리드 추천 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def run_rule(profile, intent):
    """
    [수정] 비회원용 룰 기반 추천을 실행한다.
    유효하지 않은 페르소나면 에러 메시지를 반환한다.
    """
    try:
        # [수정] 페르소나 유효성 검증 추가
        if not validate_persona(profile):
            return json.dumps({
                "error": f"'{profile}'은 유효하지 않은 페르소나입니다.",
                "valid_personas": VALID_PERSONAS,
                "recommendations": []
            }, ensure_ascii=False)

        results = run_rule_engine(profile, intent, data['card'], data['deposit'])
        return json.dumps({
            "recommendations": results,
            "profile": normalize_persona_name(profile),
            "detected_intent": intent
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"룰 기반 추천 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_details(pids):
    """상품 ID 목록으로 상세 정보를 조회한다."""
    try:
        if isinstance(pids, str):
            pids = json.loads(pids)
        details = []
        for p in pids:
            matches = product_db[product_db['product_id'] == p]
            if len(matches) > 0:
                details.append(matches.iloc[0]['summary_text'])
            else:
                details.append(f"[{p}] 상품 정보를 찾을 수 없습니다.")
        return "\n".join(details)
    except Exception as e:
        logger.error(f"상품 상세 조회 실패: {e}")
        return f"조회 실패: {e}"


def search_info(query):
    """키워드로 상품을 검색한다. (FAISS 벡터 검색)"""
    if index is None or client is None:
        return "검색 기능을 사용할 수 없습니다."
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        q_vec = np.array([response.data[0].embedding], dtype="float32")
        _, I = index.search(q_vec, 3)
        results = [product_db.iloc[i]["summary_text"] for i in I[0] if i < len(product_db)]
        return "\n".join(results) if results else "관련 상품을 찾을 수 없습니다."
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        return f"검색 실패: {e}"


def analyze_spending(cid, estimated_income=2500000):
    """
    [문제3] 고객의 소비 패턴을 분석한다.

    Args:
        cid: 고객 ID
        estimated_income: 추정 월 소득 (기본값 250만원)

    Returns:
        소비 분석 결과 JSON
    """
    try:
        result = engine.spending_analyzer.analyze_customer(cid, estimated_income)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"소비 분석 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_saving_plan(cid, goal_name="비상금", goal_amount=None, target_months=12):
    """
    [문제3] 고객 맞춤 저축 계획을 생성한다.

    Args:
        cid: 고객 ID
        goal_name: 목표 이름 (해외여행, 유럽여행, 비상금, 전세자금 등)
        goal_amount: 목표 금액 (없으면 goal_name 기준으로 자동 설정)
        target_months: 목표 기간 (개월)

    Returns:
        저축 계획 JSON
    """
    try:
        result = engine.habit_coach.generate_saving_plan(
            cid,
            goal_name=goal_name,
            goal_amount=goal_amount,
            target_months=target_months
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"저축 계획 생성 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_quick_saving_tips(cid):
    """
    [문제3] 빠른 절약 팁을 제공한다.

    Args:
        cid: 고객 ID

    Returns:
        절약 팁 JSON
    """
    try:
        result = engine.habit_coach.get_quick_tips(cid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"절약 팁 생성 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_persona_explanation_context(persona_name):
    """
    페르소나별 맞춤 설명 컨텍스트를 반환한다.
    LLM이 더 구체적이고 공감가는 추천 설명을 생성하도록 돕는다.
    """
    normalized = normalize_persona_name(persona_name)

    contexts = {
        '실속 스타터': """
- 핵심 니즈: 사회초년생으로서 첫 목돈 마련, 기본 생활비 절약
- 주요 지출: 교통비, 통신비, 점심값, 카페
- 추천 포인트 강조: 
  * "월급 200만원 중 50만원 저축하면 1년에 600만원 + 이자"
  * "교통비 10% 할인으로 월 1만원, 연 12만원 절약"
  * "청년 우대 금리 혜택으로 일반 예금 대비 0.5%p 추가 이자"
- 감정 포인트: 첫 통장에 1000만원 찍히는 뿌듯함, 부모님께 독립 증명
- 피해야 할 것: 연회비 높은 프리미엄 카드, 복잡한 조건의 상품""",

        '스마트 플렉서': """
- 핵심 니즈: 여행/쇼핑을 즐기면서도 스마트하게 혜택 챙기기
- 주요 지출: 해외여행, 호텔, 명품, 맛집
- 추천 포인트 강조:
  * "연 2회 해외여행 시 마일리지로 1회 항공권 무료"
  * "백화점 5% 할인으로 100만원 쇼핑 시 5만원 절약"
  * "여행 적금 + 카드 마일리지 이중 혜택 전략"
- 감정 포인트: 똑똑하게 즐기는 자신에 대한 만족, SNS 인증샷
- 주의: 과소비 조장이 아닌 '스마트한 소비' 프레이밍""",

        '알뜰 지킴이': """
- 핵심 니즈: 생활비 최대한 절약, 안정적 저축
- 주요 지출: 마트, 공과금, 생필품, 대중교통
- 추천 포인트 강조:
  * "마트 5% 청구할인으로 월 10만원 장보기 시 5천원 절약"
  * "공과금 자동이체 할인으로 연 2-3만원 추가 절약"
  * "캐시백을 적금으로 자동 연결하면 티끌 모아 목돈"
- 감정 포인트: 알뜰하게 살림하는 똑순이/똑돌이 자부심, 비상금 안전망
- 강조: 작은 절약이 쌓이는 복리 효과""",

        '디지털 힙스터': """
- 핵심 니즈: 온라인 쇼핑, 구독 서비스 혜택 극대화
- 주요 지출: 넷플릭스/유튜브 프리미엄, 배달앱, 온라인쇼핑, 게임
- 추천 포인트 강조:
  * "넷플릭스+유튜브+멜론 구독료 월 3만원 중 10% 할인 = 연 3.6만원"
  * "배달앱 상시 5% 할인, 월 10만원 주문 시 연 6만원 절약"
  * "간편결제 추가 적립으로 포인트 이중 적립"
- 감정 포인트: 트렌드를 놓치지 않으면서 현명하게 소비하는 나
- 주의: 구독 서비스 과다 가입 경고도 함께 제공""",

        '밸런스 메인스트림': """
- 핵심 니즈: 무난하고 안정적인 혜택, 복잡한 조건 싫어함
- 주요 지출: 점심값, 커피, 편의점, 일상 소비 전반
- 추천 포인트 강조:
  * "전 가맹점 1% 적립, 복잡한 조건 없이 심플하게"
  * "점심값 월 20만원 사용 시 연 2.4만원 적립"
  * "연회비 없는 카드로 부담 없이 시작"
- 감정 포인트: 복잡하게 따지지 않아도 손해 안 보는 안심감
- 강조: 단순하지만 확실한 혜택, 신경 쓸 것 없는 편리함"""
    }

    return contexts.get(normalized, contexts['실속 스타터'])


def run_agent(user_input, user_mode, cid=None, persona=None):
    """
    [수정] 에이전트 실행.
    - user_mode: 'member' (기존 회원) 또는 'guest' (비회원)
    - cid: 회원일 경우 고객 ID
    - persona: 비회원일 경우 선택한 페르소나

    [수정사항]
    - 비회원 모드에서 페르소나가 None이면 에러 메시지 반환
    - 회원 모드에서 고객 ID가 없으면 에러 메시지 반환
    """
    if client is None:
        return "OpenAI API가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 확인해주세요."

    # [수정] 입력 유효성 검증
    if user_mode == 'member':
        if not cid:
            return "고객 ID가 필요합니다. 사이드바에서 고객 ID를 입력해주세요."
        analyzed_persona = engine.get_persona_name(cid)
        if analyzed_persona is None:
            return f"고객 ID '{cid}'를 찾을 수 없습니다. 올바른 ID를 입력하거나 비회원 모드를 이용해주세요."
        tom_info = engine.get_tom_profile(cid)
        log_summary = engine.log.summary(cid)
        tom_insight = f"(TOM지표: {json.dumps(tom_info, ensure_ascii=False)})\n[최근 행동 로그]: {log_summary}"
        context_type = "기존 회원"
        tool_instruction = "반드시 `run_hybrid` 도구를 사용하여 개인화된 추천을 제공하세요."
    else:
        # [수정] 비회원 모드에서 페르소나 필수 검증
        if not persona:
            return "페르소나를 선택해주세요. 사이드바에서 본인의 소비 성향에 맞는 페르소나를 선택해주세요."
        if not validate_persona(persona):
            return f"'{persona}'은 유효하지 않은 페르소나입니다. 유효한 페르소나: {', '.join(VALID_PERSONAS)}"

        analyzed_persona = normalize_persona_name(persona)
        tom_insight = "(비회원 - 페르소나 기반 추천)"
        context_type = "비회원/신규"
        tool_instruction = "반드시 `run_rule` 도구를 사용하여 페르소나 기반 추천을 제공하세요."

    # 페르소나별 맞춤 컨텍스트 생성
    persona_context = get_persona_explanation_context(analyzed_persona)

    sys_msg = f"""
# Role: 금융 AI 파트너 'FirstFin'

## Context
- 사용자 유형: {context_type}
- 고객 ID: {cid or 'Guest'}
- 페르소나: "{analyzed_persona}"
- 데이터 분석: {tom_insight}

## 페르소나 특성
{persona_context}

## Guidelines

### 1. Tool 사용 규칙
- {tool_instruction}
- 추천 상품은 반드시 `get_details`로 혜택 확인 후 설명
- 허위 혜택 절대 금지, Tool 결과만 사용

### 2. 추천 근거 설명 (매우 중요)
반드시 "왜 이 상품이 당신에게 적합한지"를 다음 구조로 설명하세요:

**a) 현재 상황 연결**: 사용자의 페르소나/소비 패턴과 상품을 연결
   - 예시: "월급날 다음날 충동소비가 많으신 패턴을 보면..."
   - 예시: "여행을 좋아하시는 스마트 플렉서 성향에 맞게..."

**b) 단기 동기 + 장기 습관 연결**: 즉각적 혜택과 장기적 이점을 함께 제시
   - 예시: "이 카드로 여행 마일리지를 모으면서(단기), 동시에 자동 적금으로 다음 여행 자금도 마련할 수 있어요(장기)"
   - 예시: "매일 커피값 5천원 중 10%가 자동 적립되어(단기), 1년이면 18만원이 모여요(장기)"

**c) 구체적 숫자 제시**: 추상적 설명 대신 계산된 혜택 제시
   - 예시: "월 30만원 교통비 사용 시, 이 카드로 연 36,000원 절약"
   - 예시: "주 2회 배달 주문 시, 월 8,000원 캐시백"

**d) 감정적 보상 강조**: 금전적 혜택 + 심리적 만족감 연결
   - 예시: "아끼는 재미와 함께, 연말에 보너스처럼 쌓인 금액을 확인하는 뿌듯함까지!"
   - 예시: "여행 갈 때마다 '이건 내가 모은 마일리지!'라는 성취감을 느낄 수 있어요"

### 3. 응답 톤 & 형식
- 사회초년생 눈높이에 맞춰 쉽게 설명 (전문 용어 최소화)
- 친근하지만 신뢰감 있는 톤
- 상품당 2-3문장으로 핵심 포인트 전달
- 강요하지 않고 선택지 제공 ("~하시면 좋을 것 같아요", "~도 고려해보세요")

### 4. 금지 사항
- 근거 없는 수치 제시 금지
- "좋은 상품입니다" 같은 모호한 추천 금지
- 사용자 상황과 무관한 일반적 설명 금지

### 5. 소비 습관 교정 기능 (회원 전용)
사용자가 소비 습관, 절약, 저축 관련 질문을 하면:

**a) 소비 패턴 분석** (`analyze_spending` 사용)
- 카테고리별 지출 현황과 과소비 영역 파악
- 시간대별/요일별 소비 패턴 분석
- 월급날 전후 충동소비 패턴 감지

**b) 맞춤 저축 계획** (`get_saving_plan` 사용)
- 목표 설정: 여행, 비상금, 전세자금 등
- 구체적인 절약 금액과 방법 제시
- 예시: "배달비 월 15만원 중 30% 절약 시 월 4.5만원, 연 54만원 저축 가능"

**c) 동기부여 메시지**
- 단기 목표와 장기 목표 연결
- 예시: "매주 배달 1회만 줄이면 1년 후 유럽 여행 경비 70% 마련!"
- 작은 성공 경험 강조로 지속 가능한 습관 형성 유도

**d) 응답 예시**
```
[소비 분석 결과]
- 배달: 월 15만원 (권장 대비 2배 초과)
- 카페: 월 8만원 (권장 대비 1.5배)

[절약 제안]
1. 배달 주 4회 → 2회: 월 6만원 절약
2. 텀블러 지참: 월 1.5만원 절약
→ 월 7.5만원, 연 90만원 저축 가능!

[목표 연결]
이 금액이면 12개월 후 동남아 여행 자금 완성!
```
"""

    msgs = [{"role": "system", "content": sys_msg}]
    mem = load_memory()
    if mem:
        msgs.append({"role": "user", "content": f"[이전 대화 요약]\n{mem[-600:]}"})
    msgs.append({"role": "user", "content": user_input})

    # Tool 정의
    tools = [
        {"type": "function",
         "function": {"name": "run_hybrid", "description": "기존 회원용: 고객 ID 기반 개인화 추천 (로그, 만족도, 유사고객 분석)",
                      "parameters": {"type": "object", "properties": {"cid": {"type": "string", "description": "고객 ID"},
                                                                      "intent": {"type": "string",
                                                                                 "description": "사용자 의도"}},
                                     "required": ["cid"]}}},
        {"type": "function", "function": {"name": "run_rule", "description": "비회원용: 페르소나 기반 룰 추천",
                                          "parameters": {"type": "object", "properties": {"profile": {"type": "string",
                                                                                                      "description": f"페르소나 이름. 유효한 값: {', '.join(VALID_PERSONAS)}"},
                                                                                          "intent": {"type": "string",
                                                                                                     "description": "사용자 의도"}},
                                                         "required": ["profile", "intent"]}}},
        {"type": "function", "function": {"name": "get_details", "description": "상품 ID로 상세 혜택 조회",
                                          "parameters": {"type": "object", "properties": {
                                              "pids": {"type": "array", "items": {"type": "string"}}},
                                                         "required": ["pids"]}}},
        {"type": "function", "function": {"name": "search_info", "description": "키워드로 상품 검색",
                                          "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                         "required": ["query"]}}},
        # [문제3] 소비 습관 교정 Tool 추가
        {"type": "function",
         "function": {"name": "analyze_spending",
                      "description": "고객의 소비 패턴을 분석한다. 카테고리별 지출, 과소비 영역, 시간대별/요일별 패턴, 충동소비 감지 등을 제공한다. 회원 전용 기능.",
                      "parameters": {"type": "object",
                                     "properties": {
                                         "cid": {"type": "string", "description": "고객 ID"},
                                         "estimated_income": {"type": "number", "description": "추정 월 소득 (기본값 250만원)",
                                                              "default": 2500000}
                                     },
                                     "required": ["cid"]}}},
        {"type": "function",
         "function": {"name": "get_saving_plan",
                      "description": "고객 맞춤 저축 계획을 생성한다. 과소비 카테고리 분석 후 구체적인 절약 방법과 목표 달성 시뮬레이션을 제공한다. 회원 전용 기능.",
                      "parameters": {"type": "object",
                                     "properties": {
                                         "cid": {"type": "string", "description": "고객 ID"},
                                         "goal_name": {"type": "string",
                                                       "description": "목표 이름 (해외여행, 유럽여행, 비상금, 전세자금, 결혼자금 등)",
                                                       "default": "비상금"},
                                         "goal_amount": {"type": "number",
                                                         "description": "목표 금액 (없으면 goal_name 기준 자동 설정)"},
                                         "target_months": {"type": "number", "description": "목표 기간 (개월)", "default": 12}
                                     },
                                     "required": ["cid"]}}},
        {"type": "function",
         "function": {"name": "get_quick_saving_tips",
                      "description": "빠른 절약 팁을 제공한다. 간단한 분석 후 즉시 실천 가능한 절약 조언을 제공한다. 회원 전용 기능.",
                      "parameters": {"type": "object",
                                     "properties": {
                                         "cid": {"type": "string", "description": "고객 ID"}
                                     },
                                     "required": ["cid"]}}}
    ]

    try:
        res = client.chat.completions.create(model="gpt-4o", messages=msgs, tools=tools, tool_choice="auto",
                                             temperature=0.1)
        msg = res.choices[0].message
    except Exception as e:
        logger.error(f"OpenAI API 호출 실패: {e}")
        return f"AI 응답 생성 중 오류가 발생했습니다: {e}"

    if msg.tool_calls:
        msgs.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                msgs.append({"role": "tool", "tool_call_id": tc.id, "name": fn, "content": f"인자 파싱 오류: {e}"})
                continue

            is_valid, error_msg = validate_tool_args(fn, args)
            if not is_valid:
                msgs.append({"role": "tool", "tool_call_id": tc.id, "name": fn, "content": f"검증 실패: {error_msg}"})
                continue

            if fn == "run_hybrid":
                result = run_hybrid(args.get('cid') or cid, args.get('intent', ''))
            elif fn == "run_rule":
                # [수정] LLM이 전달한 profile이 유효하지 않으면 사용자 선택 페르소나 사용
                profile_arg = args.get('profile')
                if not validate_persona(profile_arg):
                    profile_arg = analyzed_persona
                result = run_rule(profile_arg, args.get('intent', ''))
            elif fn == "get_details":
                result = get_details(args.get('pids', []))
            elif fn == "search_info":
                result = search_info(args.get('query', ''))
            # [문제3] 소비 습관 분석 Tool 처리 추가
            elif fn == "analyze_spending":
                result = analyze_spending(
                    args.get('cid') or cid,
                    args.get('estimated_income', 2500000)
                )
            elif fn == "get_saving_plan":
                result = get_saving_plan(
                    args.get('cid') or cid,
                    args.get('goal_name', '비상금'),
                    args.get('goal_amount'),
                    args.get('target_months', 12)
                )
            elif fn == "get_quick_saving_tips":
                result = get_quick_saving_tips(args.get('cid') or cid)
            else:
                result = "알 수 없는 도구입니다."
            msgs.append({"role": "tool", "tool_call_id": tc.id, "name": fn, "content": str(result)})

        try:
            final = client.chat.completions.create(model="gpt-4o", messages=msgs, temperature=0.7)
            answer = final.choices[0].message.content
        except Exception as e:
            logger.error(f"최종 응답 생성 실패: {e}")
            answer = "응답 생성 중 오류가 발생했습니다."
    else:
        answer = msg.content

    save_memory(user_input, answer)
    return answer


# -----------------------------------------
# 9. 사이드바 - 모드 분리
# -----------------------------------------
with st.sidebar:
    st.header("사용자 설정")

    if client is None:
        st.error("API Key 미설정")
        st.caption("`.env` 파일에 `OPENAI_API_KEY=sk-...` 추가")
    else:
        st.success("API 연결됨")

    st.divider()

    # 모드 선택
    user_mode = st.radio(
        "사용자 유형 선택",
        options=["guest", "member"],
        format_func=lambda x: "비회원 / 신규 방문자" if x == "guest" else "기존 회원 (ID 로그인)",
        index=0,
        help="기존 회원은 축적된 데이터 기반 초개인화 추천, 비회원은 페르소나 기반 추천"
    )

    st.divider()

    # 모드에 따른 UI 분기
    cid_input = None
    selected_persona = None

    if user_mode == "member":
        # 기존 회원 모드
        st.subheader("기존 회원 로그인")
        cid_input = st.text_input(
            "고객 ID 입력",
            placeholder="예: C00001",
            help="고객 ID를 입력하면 거래 내역, 관심 상품 등을 분석하여 맞춤 추천을 제공합니다."
        )

        if cid_input:
            # ID 유효성 검증
            if cid_input in data['customers']['customer_id'].values:
                st.success(f"로그인 성공: {cid_input}")

                # 고객 정보 미리보기
                persona_name = engine.get_persona_name(cid_input)
                if persona_name:
                    st.info(f"분석된 페르소나: **{persona_name}**")
                else:
                    st.warning("페르소나 분석에 실패했습니다.")
            else:
                st.warning(f"'{cid_input}' ID를 찾을 수 없습니다. 비회원 모드로 전환하세요.")
                cid_input = None
        else:
            st.caption("고객 ID를 입력해주세요.")

    else:
        # 비회원 모드
        st.subheader("비회원 / 신규 방문자")
        st.caption("본인의 소비 성향과 가장 가까운 페르소나를 선택해주세요.")

        # [수정] 페르소나 맵을 VALID_PERSONAS 기반으로 구성
        persona_map = {
            "실속 스타터": "사회초년생 | 목돈 마련, 교통/통신비 할인 중시",
            "스마트 플렉서": "YOLO | 여행, 호캉스, 명품 소비 선호",
            "디지털 힙스터": "트렌드 | 넷플릭스, 간편결제 혜택 필수",
            "알뜰 지킴이": "절약 | 마트/공과금 할인 최우선",
            "밸런스 메인스트림": "직장인 | 점심/커피 등 무난한 혜택"
        }

        selected_persona = st.selectbox(
            "나의 소비 성향은?",
            options=VALID_PERSONAS,  # [수정] 상수 사용
            index=0
        )
        st.info(f"{persona_map.get(selected_persona, '')}")

    st.divider()

    # 시스템 상태
    with st.expander("시스템 상태"):
        st.write(f"- 고객: {len(data['customers']):,}명")
        st.write(f"- 로그: {len(data['logs']):,}건")
        st.write(f"- 상품: {len(product_db):,}개")
        st.write(f"- 인덱스: {'O' if index else 'X'}")
        st.write(f"- 현재 모드: **{'기존회원' if user_mode == 'member' else '비회원'}**")
        if user_mode == 'guest' and selected_persona:
            st.write(f"- 선택 페르소나: **{selected_persona}**")

    if st.button("대화 초기화"):
        clear_memory()
        st.session_state.session = []
        st.rerun()

# -----------------------------------------
# 10. 채팅 인터페이스
# -----------------------------------------
if "session" not in st.session_state:
    st.session_state.session = []

# 현재 모드 표시
if user_mode == "member" and cid_input:
    st.caption(f"**기존 회원 모드** | 고객 ID: `{cid_input}` | 개인화 추천 활성화")
else:
    st.caption(f"**비회원 모드** | 페르소나: `{selected_persona}` | 페르소나 기반 추천")

for role, msg in st.session_state.session:
    with st.chat_message(role):
        st.write(msg)

if user_msg := st.chat_input("금융 상품에 대해 물어보세요..."):
    st.session_state.session.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            reply = run_agent(
                user_input=user_msg,
                user_mode=user_mode,
                cid=cid_input if user_mode == "member" else None,
                persona=selected_persona if user_mode == "guest" else None
            )
            st.write(reply)
    st.session_state.session.append(("assistant", reply))

# -----------------------------------------
# 11. 대시보드 (기존 회원 전용)
# -----------------------------------------
if user_mode == "member" and cid_input and 'customers_train' in data and not data['customers_train'].empty:
    user_vec = data['customers_train'][data['customers_train']['customer_id'] == cid_input]

    if not user_vec.empty:
        st.divider()
        st.subheader(f"FirstFin Insight: {cid_input}")

        # 탭으로 대시보드 구분
        tab1, tab2 = st.tabs(["라이프스타일 분석", "소비 습관 교정"])

        with tab1:
            col1, col2 = st.columns(2)

            target_cols = ['TOM_Invest', 'TOM_YOLO', 'TOM_Weekend', 'TOM_Digital', 'TOM_Cafe']
            valid_cols = [c for c in target_cols if c in user_vec.columns]

            if valid_cols:
                tom_metrics = user_vec[valid_cols].T
                tom_metrics.columns = ['Score']
                name_map = {
                    'TOM_Invest': '투자성향',
                    'TOM_YOLO': 'YOLO지수',
                    'TOM_Weekend': '주말소비',
                    'TOM_Digital': '디지털친화',
                    'TOM_Cafe': '취향(카페)'
                }
                tom_metrics.index = [name_map.get(c, c) for c in valid_cols]
                with col1:
                    st.markdown("**TOM 라이프스타일 프로필**")
                    st.bar_chart(tom_metrics, color="#4A90E2")

            with col2:
                st.markdown("**소비 증감 추세**")
                trend_col = 'TOM_Trend_Raw' if 'TOM_Trend_Raw' in user_vec.columns else 'TOM_Trend'
                if trend_col in user_vec.columns:
                    trend_val = user_vec[trend_col].values[0]
                    base_point = 100
                    df_trend = pd.DataFrame(
                        [base_point * (1 - trend_val), base_point, base_point * (1 + trend_val)],
                        columns=['예상 소비 흐름'],
                        index=['지난달', '이번달', '다음달(예측)']
                    )
                    if trend_val > 0.05:
                        st.warning(f"소비 급증! (+{trend_val:.1%})")
                        st.line_chart(df_trend, color="#FF4B4B")
                    elif trend_val < -0.05:
                        st.success(f"절약 모드 ({trend_val:.1%})")
                        st.line_chart(df_trend, color="#2ECC71")
                    else:
                        st.info(f"안정적 ({trend_val:.1%})")
                        st.line_chart(df_trend, color="#808495")

        # [문제3] 소비 습관 교정 탭
        with tab2:
            st.markdown("### 소비 패턴 분석 및 교정")

            # 소비 분석 실행
            spending_analysis = engine.spending_analyzer.analyze_customer(cid_input)

            if spending_analysis.get('status') == 'success':
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**월평균 지출 현황**")
                    monthly_spending = spending_analysis['monthly_average_spending']
                    spending_ratio = spending_analysis['spending_ratio']

                    st.metric(
                        label="월평균 지출",
                        value=f"{monthly_spending:,.0f}원",
                        delta=f"소득 대비 {spending_ratio:.1f}%"
                    )

                    # 과소비 카테고리 표시
                    category_data = spending_analysis.get('category_analysis', {})
                    overspending = category_data.get('overspending', [])

                    if overspending:
                        st.markdown("**과소비 주의 카테고리**")
                        for item in overspending[:3]:
                            excess_pct = (item['monthly_amount'] / item['recommended'] - 1) * 100
                            st.warning(f"{item['category']}: 월 {item['monthly_amount']:,}원 (권장 대비 +{excess_pct:.0f}%)")
                    else:
                        st.success("과소비 카테고리가 없습니다!")

                with col2:
                    st.markdown("**소비 패턴 진단**")

                    # 시간대 분석
                    time_analysis = spending_analysis.get('time_analysis', {})
                    if time_analysis.get('late_night_warning'):
                        st.error(f"심야 소비 비율: {time_analysis['late_night_ratio']:.1f}% (주의 필요)")

                    # 요일 분석
                    weekday_analysis = spending_analysis.get('weekday_analysis', {})
                    if weekday_analysis.get('weekend_heavy'):
                        st.warning(f"주말 소비 집중: {weekday_analysis['weekend_ratio']:.1f}%")
                    else:
                        st.info(f"주말 소비 비율: {weekday_analysis.get('weekend_ratio', 0):.1f}% (균형적)")

                    # 월급날 패턴
                    payday = spending_analysis.get('payday_analysis', {})
                    if payday.get('payday_spike'):
                        st.warning(f"월급 직후 과소비: {payday['post_payday_ratio']:.1f}%")

                    # 충동소비 패턴
                    impulse = spending_analysis.get('impulse_analysis', {})
                    if impulse.get('has_impulse_pattern'):
                        st.error(f"충동소비 감지: {impulse['impulse_count']}건 (총 {impulse['impulse_total']:,}원)")

                # 절약 계획 섹션
                st.divider()
                st.markdown("### 맞춤 저축 계획")

                # 목표 설정
                goal_col1, goal_col2, goal_col3 = st.columns(3)
                with goal_col1:
                    goal_name = st.selectbox(
                        "저축 목표",
                        options=['비상금', '해외여행', '유럽여행', '전세자금', '결혼자금'],
                        index=0
                    )
                with goal_col2:
                    target_months = st.slider("목표 기간 (개월)", 6, 24, 12)
                with goal_col3:
                    if st.button("저축 계획 생성"):
                        st.session_state.show_saving_plan = True

                # 저축 계획 표시
                if st.session_state.get('show_saving_plan', False):
                    saving_plan = engine.habit_coach.generate_saving_plan(
                        cid_input,
                        goal_name=goal_name,
                        target_months=target_months
                    )

                    if saving_plan.get('status') == 'success':
                        # 목표 정보
                        goal_info = saving_plan['goal']
                        st.info(
                            f"**목표**: {goal_info['name']} - {goal_info['amount']:,}원 / {goal_info['target_months']}개월")

                        # 절약 기회
                        opportunities = saving_plan.get('saving_opportunities', {})
                        total_potential = opportunities.get('total_potential', 0)

                        st.metric(
                            label="월간 절약 가능 금액",
                            value=f"{total_potential:,}원",
                            delta=f"연간 {total_potential * 12:,}원"
                        )

                        # 카테고리별 절약 방법
                        if opportunities.get('details'):
                            st.markdown("**카테고리별 절약 방법**")
                            for opp in opportunities['details'][:3]:
                                with st.expander(f"{opp['category']} - 월 {opp['potential_saving']:,}원 절약 가능"):
                                    st.write(f"현재 지출: 월 {opp['current_monthly']:,}원")
                                    st.write(f"초과 금액: {opp['excess_amount']:,}원")
                                    st.write("**절약 팁:**")
                                    for tip in opp.get('saving_tips', []):
                                        st.write(f"- {tip}")

                        # 동기부여 메시지
                        st.success(saving_plan.get('motivation_message', ''))

                        # 실천 계획
                        action_plan = saving_plan.get('action_plan', {})
                        if action_plan.get('priority_actions'):
                            st.markdown("**주간 실천 계획**")
                            for action in action_plan['priority_actions'][:3]:
                                st.write(f"- **{action['category']}**: {action['weekly_action']}")
                    else:
                        st.error(saving_plan.get('message', '저축 계획 생성에 실패했습니다.'))
            else:
                st.warning(spending_analysis.get('message', '소비 분석 데이터가 없습니다.'))