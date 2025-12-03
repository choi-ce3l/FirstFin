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
# 🔧 1. Settings & Initialization
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
# 🔑 2. OpenAI Client (Lazy Init)
# -----------------------------------------
@st.cache_resource
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        return None


# -----------------------------------------
# 💾 3. Memory Functions
# -----------------------------------------
def get_memory_path():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return f"./FirstFin_memory_{st.session_state.session_id}.txt"


def save_memory(user_msg, assistant_msg):
    try:
        with open(get_memory_path(), "a", encoding="utf-8") as f:
            f.write(f"User: {user_msg}\nAgent: {assistant_msg}\n")
    except Exception as e:
        logger.warning(f"메모리 저장 실패: {e}")


def load_memory():
    try:
        path = get_memory_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.warning(f"메모리 로드 실패: {e}")
    return ""


def clear_memory():
    try:
        path = get_memory_path()
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"메모리 삭제 실패: {e}")


# -----------------------------------------
# 🛠️ 4. Feature Engineering (TOM)
# -----------------------------------------
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
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def get_clean_tom_dataset_v2(df):
    temp_df = df.copy()
    if 'customer_id' not in temp_df.columns:
        logger.error("customer_id 컬럼이 없습니다.")
        return pd.DataFrame()

    for col in TOM_SCHEMA['numeric_optional']:
        temp_df[col] = safe_get_column(temp_df, col, 0)

    temp_df['TOM_Invest'] = (
            safe_get_column(temp_df, 'FIN_STOCK_24_4', 0) +
            safe_get_column(temp_df, 'FIN_COIN_24_4', 0) * 2.0
    )

    yolo_cols = ['SHC_TRAVEL_AMT_24_4', 'SHC_ENT_AMT_24_4', 'SHC_STARBUCKS_AMT_24_4',
                 'SHC_HOTEL_AMT_24_4', 'SHC_M_DF_AMT_24_4']
    total_spend = safe_get_column(temp_df, 'SHC_1YEAR_MEAN_AMT', 1).replace(0, 1)
    yolo_sum = sum(safe_get_column(temp_df, c, 0) for c in yolo_cols)
    temp_df['TOM_YOLO'] = yolo_sum / total_spend

    digital_interest = ['ENT_SVOD_24_4', 'ENT_WEBTOON_24_4', 'COMM_SNS_24_4', 'SHOP_SOCIAL_24_4']
    digital_action = ['SHC_E_COMM_AMT_24_4', 'SHC_DLV_AMT_24_4']
    interest_mean = pd.concat([safe_get_column(temp_df, c, 0) for c in digital_interest], axis=1).mean(axis=1)
    action_sum = sum(safe_get_column(temp_df, c, 0) for c in digital_action)
    temp_df['TOM_Digital'] = interest_mean + (action_sum / total_spend * 5.0)

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
    temp_df['TOM_Asset'] = safe_get_column(temp_df, 'NET_ASST_24', 0)

    keep_cols = ['customer_id', 'AGE', 'SEX', 'JB_TP', 'TOM_Invest', 'TOM_YOLO',
                 'TOM_Digital', 'TOM_Asset', 'TOM_Main_Interest']
    return temp_df[[c for c in keep_cols if c in temp_df.columns]].copy()


def create_lifestyle_tom_features(trans_df, profile_df):
    if trans_df.empty or profile_df.empty:
        return profile_df

    local_trans = trans_df.copy()
    local_profile = profile_df.copy()

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

    local_trans['day_of_week'] = local_trans['transaction_date'].dt.dayofweek
    local_trans['month_idx'] = (
            local_trans['transaction_date'].dt.year * 12 +
            local_trans['transaction_date'].dt.month
    )

    grouped = local_trans.groupby('customer_id')
    weekend_mask = local_trans['day_of_week'] >= 5
    weekend_spend = local_trans[weekend_mask].groupby('customer_id')['amount'].sum()
    total_spend = grouped['amount'].sum().replace(0, 1)
    tom_weekend = (weekend_spend / total_spend).reindex(local_profile['customer_id']).fillna(0)

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

    lifestyle_df = pd.DataFrame({
        'customer_id': local_profile['customer_id'].values,
        'TOM_Weekend': tom_weekend.values,
        'TOM_Cafe': tom_cafe.values,
        'TOM_Conv': tom_conv.values,
        'TOM_Trend_Raw': tom_trend_raw.values
    })
    final_df = pd.merge(local_profile, lifestyle_df, on='customer_id', how='left').fillna(0)

    scaler = MinMaxScaler()
    num_cols = ['AGE', 'TOM_Invest', 'TOM_YOLO', 'TOM_Digital', 'TOM_Asset',
                'TOM_Weekend', 'TOM_Cafe', 'TOM_Conv']
    valid_nums = [c for c in num_cols if c in final_df.columns]
    if valid_nums:
        final_df[valid_nums] = scaler.fit_transform(final_df[valid_nums])

    final_df['TOM_Trend'] = final_df['TOM_Trend_Raw'].clip(-1, 1)

    cat_cols = ['SEX', 'JB_TP', 'TOM_Main_Interest']
    valid_cats = [c for c in cat_cols if c in final_df.columns]
    final_df = pd.get_dummies(final_df, columns=valid_cats, prefix=['Sex', 'Job', 'Interest'])

    return final_df


# -----------------------------------------
# 📦 5. Data Load
# -----------------------------------------
def get_embedding_cache_path(product_db):
    content_hash = hashlib.md5(
        product_db['summary_text'].to_json().encode()
    ).hexdigest()[:8]
    return os.path.join(CACHE_PATH, f"embeddings_{EMBEDDING_MODEL}_{EMBEDDING_VERSION}_{content_hash}.npy")


@st.cache_data(show_spinner="📂 데이터 로드 중...")
def load_all_data():
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

    try:
        deposit_path = DATA_PATH + "deposit_product_info_최신.xlsx"
        if os.path.exists(deposit_path):
            data['deposit'] = pd.read_excel(deposit_path)
            logger.info(f"예금 상품 로드: {len(data['deposit'])}개")
    except Exception as e:
        logger.warning(f"예금 상품 로드 실패: {e}")

    try:
        card_path = DATA_PATH + "card_product_info_최신.xlsx"
        if os.path.exists(card_path):
            data['card'] = pd.read_excel(card_path)
            logger.info(f"카드 상품 로드: {len(data['card'])}개")
    except Exception as e:
        logger.warning(f"카드 상품 로드 실패: {e}")

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

    try:
        logs_path = DATA_PATH + "customer_logs.csv"
        if os.path.exists(logs_path):
            data['logs'] = pd.read_csv(logs_path)
            logger.info(f"로그 데이터 로드: {len(data['logs'])}건")
    except Exception as e:
        logger.warning(f"로그 데이터 로드 실패: {e}")

    try:
        sat_path = DATA_PATH + "customer_satisfaction.csv"
        if os.path.exists(sat_path):
            data['satisfaction'] = pd.read_csv(sat_path)
            logger.info(f"만족도 데이터 로드: {len(data['satisfaction'])}건")
    except Exception as e:
        logger.warning(f"만족도 데이터 로드 실패: {e}")

    return data


def build_product_db(data):
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
    if len(product_db) == 0 or client is None:
        return None

    cache_path = get_embedding_cache_path(product_db)
    embeddings = None

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

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# -----------------------------------------
# 🤖 6. Recommendation Engines
# -----------------------------------------
def normalize_persona_name(persona_with_suffix):
    base_personas = ['밸런스 메인스트림', '스마트 플렉서', '알뜰 지킴이', '실속 스타터', '디지털 힙스터']
    for base in base_personas:
        if base in persona_with_suffix:
            return base
    return '실속 스타터'


class FirstFinKNNRecommender:
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
        if self.model is None or self.df is None or cid not in self.df.index:
            return []
        try:
            _, idx = self.model.kneighbors(self.features.loc[[cid]], n_neighbors=min(n + 1, len(self.features)))
            return self.df.iloc[idx[0][1:]].index.tolist()
        except Exception as e:
            logger.warning(f"KNN 추천 실패: {e}")
            return []


class LogRecommender:
    def __init__(self, df, card_df, dep_df):
        self.df = df.copy() if len(df) > 0 else pd.DataFrame()
        if len(self.df) > 0:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.weights = {'apply': 5, 'compare': 3, 'view': 2, 'click': 1}

    def recommend(self, cid, days=30, k=5):
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
    def __init__(self, df, card_df, dep_df):
        self.df = df
        self.card_df = card_df
        self.dep_df = dep_df

    def recommend(self, cid, similar_ids, k=5):
        if len(self.df) == 0 or not similar_ids:
            return []
        try:
            sim = self.df[self.df['customer_id'].isin(similar_ids)]
            stats = sim[sim['rating'] >= 4.0].groupby(
                ['product_id', 'product_name', 'product_type']
            ).agg({'rating': 'mean', 'customer_id': 'count'}).reset_index()
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
        if len(self.log_df) == 0:
            return True
        return len(self.log_df[self.log_df['customer_id'] == cid]) < 5

    def _fuzzy_match(self, text, keywords, threshold=0.4):
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

    def recommend(self, cid, k=5):
        if len(self.cust_df) == 0:
            return []
        cust_row = self.cust_df[self.cust_df['customer_id'] == cid]
        persona_id = 3 if cust_row.empty else int(cust_row.iloc[0].get('Persona_Cluster', 3))
        profile = self.PROFILES.get(persona_id, self.PROFILES[3])
        results = []

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

        if len(self.dep_df) > 0 and 'product_name' in self.dep_df.columns:
            dep_scores = self.dep_df.copy()
            dep_scores['match_score'] = dep_scores['product_name'].apply(
                lambda x: self._fuzzy_match(x, profile['dep_keywords']))
            for _, dep in dep_scores[dep_scores['match_score'] > 0].nlargest(2, 'match_score').iterrows():
                results.append({'product_id': dep['product_id'], 'product_name': dep['product_name'],
                                'score': round(dep['match_score'] * 4, 2), 'reason': f"{profile['name']} 맞춤 추천",
                                'source': 'zeroshot'})

        return results[:k]

    def recommend_by_persona_name(self, persona_name, k=5):
        """페르소나 이름으로 직접 추천 (비회원용)"""
        normalized = normalize_persona_name(persona_name)
        profile = None
        for pid, p in self.PROFILES.items():
            if p['name'] == normalized:
                profile = p
                break
        if profile is None:
            profile = self.PROFILES[3]  # 기본값: 실속 스타터

        results = []

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
    normalized_profile = normalize_persona_name(profile_name)
    RULES = {
        '실속 스타터': {'default': {'card': ['청년', '교통', '통신'], 'deposit': ['적금', '청년']},
                   '여행': {'card': ['항공', '여행'], 'deposit': ['여행', '적금']},
                   '저축': {'card': ['캐시백'], 'deposit': ['적금', '정기']}},
        '스마트 플렉서': {'default': {'card': ['프리미엄', '여행', '쇼핑'], 'deposit': ['고금리', '예금']},
                    '여행': {'card': ['항공', 'VIP'], 'deposit': ['외화']}, '쇼핑': {'card': ['쇼핑', '백화점'], 'deposit': ['자유']}},
        '알뜰 지킴이': {'default': {'card': ['캐시백', '마트', '생활'], 'deposit': ['적금', '예금']},
                   '저축': {'card': ['적립'], 'deposit': ['정기적금']}},
        '디지털 힙스터': {'default': {'card': ['온라인', '쇼핑', '구독'], 'deposit': ['모바일', '입출금']},
                    '구독': {'card': ['스트리밍'], 'deposit': ['자유']}},
        '밸런스 메인스트림': {'default': {'card': ['일상', '생활'], 'deposit': ['예금', '자유']}}
    }
    intent_map = {'여행': ['여행', '해외', '항공'], '저축': ['저축', '적금', '목돈'], '쇼핑': ['쇼핑', '백화점'], '구독': ['구독', '넷플릭스']}
    detected_intent = 'default'
    intent_lower = intent.lower() if intent else ''
    for key, keywords in intent_map.items():
        if any(kw in intent_lower for kw in keywords):
            detected_intent = key
            break
    profile_rules = RULES.get(normalized_profile, RULES['실속 스타터'])
    rule = profile_rules.get(detected_intent, profile_rules['default'])
    results = []
    if len(card_df) > 0 and 'card_category' in card_df.columns:
        for kw in rule['card']:
            matches = card_df[card_df['card_category'].str.contains(kw, case=False, na=False)]
            for _, r in matches.head(1).iterrows():
                results.append(
                    {'product_id': r['product_id'], 'product_name': r['product_name'], 'reason': f"'{kw}' 키워드 매칭",
                     'source': 'rule'})
    if len(dep_df) > 0 and 'product_name' in dep_df.columns:
        for kw in rule['deposit']:
            matches = dep_df[dep_df['product_name'].str.contains(kw, case=False, na=False)]
            for _, r in matches.head(1).iterrows():
                results.append(
                    {'product_id': r['product_id'], 'product_name': r['product_name'], 'reason': f"'{kw}' 키워드 매칭",
                     'source': 'rule'})
    seen = set()
    return [r for r in results if not (r['product_id'] in seen or seen.add(r['product_id']))][:5]


class HybridEngine:
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

    def get_tom_profile(self, cid):
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
            return {"Trend": f"{trend_raw:.1%}", "YOLO": f"{row.get('TOM_YOLO', 0):.2f}",
                    "Digital": f"{row.get('TOM_Digital', 0):.2f}", "Weekend": f"{row.get('TOM_Weekend', 0):.2f}"}
        except Exception as e:
            logger.warning(f"TOM 프로필 조회 실패: {e}")
            return {"status": "조회 실패"}

    def get_persona_name(self, cid):
        if self.raw_customers.empty:
            return "실속 스타터"
        row = self.raw_customers[self.raw_customers['customer_id'] == cid]
        if row.empty:
            return "실속 스타터"
        base = self.zero.PROFILES.get(int(row.iloc[0].get('Persona_Cluster', 3)), {}).get('name', '실속 스타터')
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
        """기존 회원용 하이브리드 추천"""
        if self.zero.is_cold(cid):
            return {'recs': self.zero.recommend(cid, k), 'is_cold': True, 'ctx': {'log_sum': "신규 고객 - 기본 페르소나 기반 추천"}}
        similar = self.knn.get_similar(cid)
        log_recs = self.log.recommend(cid, k=10)
        sat_recs = self.sat.recommend(cid, similar, k=10)
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
        """비회원용 페르소나 기반 추천"""
        recs = self.zero.recommend_by_persona_name(persona_name, k)
        return {
            'recs': recs,
            'is_cold': True,
            'ctx': {'log_sum': f"비회원 - '{normalize_persona_name(persona_name)}' 페르소나 기반 추천"}
        }


# -----------------------------------------
# 🎨 7. Streamlit UI
# -----------------------------------------
st.title("🤖 FirstFin - 사회초년생을 위한 은행 상품 추천 Agent")
st.markdown("**:blue[TOM(Time-Occasion-Method)]** 및 **:green[Lifestyle]** 기반 하이브리드 추천 시스템")

# 데이터 로드
data = load_all_data()

# OpenAI 클라이언트 초기화
client = get_openai_client()

# 제품 DB 생성
product_db = build_product_db(data)

# FAISS 인덱스 생성
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
    if client is not None and len(product_db) > 0:
        with st.spinner("🔄 임베딩 인덱스 생성 중... (최초 1회)"):
            st.session_state.faiss_index = build_faiss_index(product_db, client)

index = st.session_state.faiss_index

# 엔진 초기화
engine = HybridEngine(data)


# -----------------------------------------
# 🔧 8. Tool Functions
# -----------------------------------------
def validate_tool_args(fn_name, args):
    validators = {
        'run_hybrid': lambda a: 'cid' in a and isinstance(a.get('cid'), str),
        'run_rule': lambda a: 'profile' in a and 'intent' in a,
        'get_details': lambda a: 'pids' in a and isinstance(a.get('pids'), (list, str)),
        'search_info': lambda a: 'query' in a and isinstance(a.get('query'), str)
    }
    validator = validators.get(fn_name)
    if validator is None:
        return False, f"알 수 없는 함수: {fn_name}"
    if not validator(args):
        return False, f"잘못된 인자: {fn_name}({args})"
    return True, None


def run_hybrid(cid, intent=""):
    """기존 회원용 추천"""
    try:
        r = engine.recommend(cid, 3)
        return json.dumps({"recommendations": r['recs'], "context": r['ctx'], "is_cold_start": r['is_cold']},
                          ensure_ascii=False)
    except Exception as e:
        logger.error(f"하이브리드 추천 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def run_rule(profile, intent):
    """비회원용 룰 기반 추천"""
    try:
        results = run_rule_engine(profile, intent, data['card'], data['deposit'])
        return json.dumps({"recommendations": results, "profile": profile, "detected_intent": intent},
                          ensure_ascii=False)
    except Exception as e:
        logger.error(f"룰 기반 추천 실패: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_details(pids):
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


def run_agent(user_input, user_mode, cid=None, persona=None):
    """
    에이전트 실행
    - user_mode: 'member' (기존 회원) 또는 'guest' (비회원)
    - cid: 회원일 경우 고객 ID
    - persona: 비회원일 경우 선택한 페르소나
    """
    if client is None:
        return "⚠️ OpenAI API가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 확인해주세요."

    # 모드에 따른 컨텍스트 설정
    if user_mode == 'member' and cid:
        analyzed_persona = engine.get_persona_name(cid)
        tom_info = engine.get_tom_profile(cid)
        log_summary = engine.log.summary(cid)
        tom_insight = f"(TOM지표: {json.dumps(tom_info, ensure_ascii=False)})\n[최근 행동 로그]: {log_summary}"
        context_type = "기존 회원"
        tool_instruction = "반드시 `run_hybrid` 도구를 사용하여 개인화된 추천을 제공하세요."
    else:
        analyzed_persona = persona or "실속 스타터"
        tom_insight = "(비회원 - 페르소나 기반 추천)"
        context_type = "비회원/신규"
        tool_instruction = "반드시 `run_rule` 도구를 사용하여 페르소나 기반 추천을 제공하세요."

    sys_msg = f"""
# Role: 금융 AI 파트너 'FirstFin'

## Context
- 사용자 유형: {context_type}
- 고객 ID: {cid or 'Guest'}
- 페르소나: "{analyzed_persona}"
- 데이터 분석: {tom_insight}

## Guidelines
1. {tool_instruction}
2. 추천 상품은 반드시 `get_details`로 혜택 확인 후 설명
3. 허위 혜택 절대 금지, Tool 결과만 사용
4. 친근하고 전문적인 톤, 이모지 적절히 사용
5. 사회초년생 눈높이에 맞춰 쉽게 설명
"""

    msgs = [{"role": "system", "content": sys_msg}]
    mem = load_memory()
    if mem:
        msgs.append({"role": "user", "content": f"[이전 대화 요약]\n{mem[-600:]}"})
    msgs.append({"role": "user", "content": user_input})

    tools = [
        {"type": "function",
         "function": {"name": "run_hybrid", "description": "기존 회원용: 고객 ID 기반 개인화 추천 (로그, 만족도, 유사고객 분석)",
                      "parameters": {"type": "object", "properties": {"cid": {"type": "string", "description": "고객 ID"},
                                                                      "intent": {"type": "string",
                                                                                 "description": "사용자 의도"}},
                                     "required": ["cid"]}}},
        {"type": "function", "function": {"name": "run_rule", "description": "비회원용: 페르소나 기반 룰 추천",
                                          "parameters": {"type": "object", "properties": {"profile": {"type": "string",
                                                                                                      "description": "페르소나 이름 (실속 스타터, 스마트 플렉서 등)"},
                                                                                          "intent": {"type": "string",
                                                                                                     "description": "사용자 의도"}},
                                                         "required": ["profile", "intent"]}}},
        {"type": "function", "function": {"name": "get_details", "description": "상품 ID로 상세 혜택 조회",
                                          "parameters": {"type": "object", "properties": {
                                              "pids": {"type": "array", "items": {"type": "string"}}},
                                                         "required": ["pids"]}}},
        {"type": "function", "function": {"name": "search_info", "description": "키워드로 상품 검색",
                                          "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                         "required": ["query"]}}}
    ]

    try:
        res = client.chat.completions.create(model="gpt-4o", messages=msgs, tools=tools, tool_choice="auto",
                                             temperature=0.1)
        msg = res.choices[0].message
    except Exception as e:
        logger.error(f"OpenAI API 호출 실패: {e}")
        return f"⚠️ AI 응답 생성 중 오류가 발생했습니다: {e}"

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
                # 기존 회원용
                result = run_hybrid(args.get('cid') or cid, args.get('intent', ''))
            elif fn == "run_rule":
                # 비회원용
                result = run_rule(args.get('profile') or analyzed_persona, args.get('intent', ''))
            elif fn == "get_details":
                result = get_details(args.get('pids', []))
            elif fn == "search_info":
                result = search_info(args.get('query', ''))
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
# 🎛️ 9. Sidebar - 명확한 모드 분리
# -----------------------------------------
with st.sidebar:
    st.header("⚙️ 사용자 설정")

    if client is None:
        st.error("⚠️ API Key 미설정")
        st.caption("`.env` 파일에 `OPENAI_API_KEY=sk-...` 추가")
    else:
        st.success("✅ API 연결됨")

    st.divider()

    # ✅ 모드 선택 (라디오 버튼으로 명확히 분리)
    user_mode = st.radio(
        "🔐 사용자 유형 선택",
        options=["guest", "member"],
        format_func=lambda x: "👤 비회원 / 신규 방문자" if x == "guest" else "🏦 기존 회원 (ID 로그인)",
        index=0,
        help="기존 회원은 축적된 데이터 기반 초개인화 추천, 비회원은 페르소나 기반 추천"
    )

    st.divider()

    # 모드에 따른 UI 분기
    cid_input = None
    selected_persona = None

    if user_mode == "member":
        # 🏦 기존 회원 모드
        st.subheader("🏦 기존 회원 로그인")
        cid_input = st.text_input(
            "고객 ID 입력",
            placeholder="예: C00001",
            help="고객 ID를 입력하면 거래 내역, 관심 상품 등을 분석하여 맞춤 추천을 제공합니다."
        )

        if cid_input:
            # ID 유효성 검증
            if cid_input in data['customers']['customer_id'].values:
                st.success(f"✅ 로그인 성공: {cid_input}")

                # 고객 정보 미리보기
                persona_name = engine.get_persona_name(cid_input)
                st.info(f"🎯 분석된 페르소나: **{persona_name}**")
            else:
                st.warning(f"⚠️ '{cid_input}' ID를 찾을 수 없습니다. 비회원 모드로 전환하세요.")
                cid_input = None  # 유효하지 않은 ID는 무시
        else:
            st.caption("💡 고객 ID를 입력해주세요.")

    else:
        # 👤 비회원 모드
        st.subheader("👤 비회원 / 신규 방문자")
        st.caption("본인의 소비 성향과 가장 가까운 페르소나를 선택해주세요.")

        persona_map = {
            "실속 스타터": "🎓 사회초년생 | 목돈 마련, 교통/통신비 할인 중시",
            "스마트 플렉서": "✈️ YOLO | 여행, 호캉스, 명품 소비 선호",
            "디지털 힙스터": "📱 트렌드 | 넷플릭스, 간편결제 혜택 필수",
            "알뜰 지킴이": "💰 절약 | 마트/공과금 할인 최우선",
            "밸런스 메인스트림": "☕ 직장인 | 점심/커피 등 무난한 혜택"
        }

        selected_persona = st.selectbox(
            "나의 소비 성향은?",
            options=list(persona_map.keys()),
            index=0
        )
        st.info(f"💡 {persona_map[selected_persona]}")

    st.divider()

    # 시스템 상태
    with st.expander("📊 시스템 상태"):
        st.write(f"• 고객: {len(data['customers']):,}명")
        st.write(f"• 로그: {len(data['logs']):,}건")
        st.write(f"• 상품: {len(product_db):,}개")
        st.write(f"• 인덱스: {'✅' if index else '❌'}")
        st.write(f"• 현재 모드: **{'기존회원' if user_mode == 'member' else '비회원'}**")

    if st.button("🗑️ 대화 초기화"):
        clear_memory()
        st.session_state.session = []
        st.rerun()

# -----------------------------------------
# 💬 10. Chat Interface
# -----------------------------------------
if "session" not in st.session_state:
    st.session_state.session = []

# 현재 모드 표시
if user_mode == "member" and cid_input:
    st.caption(f"🏦 **기존 회원 모드** | 고객 ID: `{cid_input}` | 개인화 추천 활성화")
else:
    st.caption(f"👤 **비회원 모드** | 페르소나: `{selected_persona}` | 페르소나 기반 추천")

for role, msg in st.session_state.session:
    with st.chat_message(role):
        st.write(msg)

if user_msg := st.chat_input("금융 상품에 대해 물어보세요..."):
    st.session_state.session.append(("user", user_msg))
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("🔍 분석 중..."):
            # 모드에 따라 다른 파라미터 전달
            reply = run_agent(
                user_input=user_msg,
                user_mode=user_mode,
                cid=cid_input if user_mode == "member" else None,
                persona=selected_persona if user_mode == "guest" else None
            )
            st.write(reply)
    st.session_state.session.append(("assistant", reply))

# -----------------------------------------
# 📊 11. Dashboard (기존 회원 전용)
# -----------------------------------------
if user_mode == "member" and cid_input and 'customers_train' in data and not data['customers_train'].empty:
    user_vec = data['customers_train'][data['customers_train']['customer_id'] == cid_input]

    if not user_vec.empty:
        st.divider()
        st.subheader(f"📊 FirstFin Insight: {cid_input}")

        col1, col2 = st.columns(2)

        target_cols = ['TOM_Invest', 'TOM_YOLO', 'TOM_Weekend', 'TOM_Digital', 'TOM_Cafe']
        valid_cols = [c for c in target_cols if c in user_vec.columns]

        if valid_cols:
            tom_metrics = user_vec[valid_cols].T
            tom_metrics.columns = ['Score']
            name_map = {'TOM_Invest': '투자성향', 'TOM_YOLO': 'YOLO지수', 'TOM_Weekend': '주말소비', 'TOM_Digital': '디지털친화',
                        'TOM_Cafe': '취향(카페)'}
            tom_metrics.index = [name_map.get(c, c) for c in valid_cols]
            with col1:
                st.markdown("**🕵️‍♂️ TOM 라이프스타일 프로필**")
                st.bar_chart(tom_metrics, color="#4A90E2")

        with col2:
            st.markdown("**📈 소비 증감 추세**")
            trend_col = 'TOM_Trend_Raw' if 'TOM_Trend_Raw' in user_vec.columns else 'TOM_Trend'
            if trend_col in user_vec.columns:
                trend_val = user_vec[trend_col].values[0]
                base_point = 100
                df_trend = pd.DataFrame([base_point * (1 - trend_val), base_point, base_point * (1 + trend_val)],
                                        columns=['예상 소비 흐름'], index=['지난달', '이번달', '다음달(예측)'])
                if trend_val > 0.05:
                    st.warning(f"🚨 소비 급증! (+{trend_val:.1%})")
                    st.line_chart(df_trend, color="#FF4B4B")
                elif trend_val < -0.05:
                    st.success(f"✅ 절약 모드 ({trend_val:.1%})")
                    st.line_chart(df_trend, color="#2ECC71")
                else:
                    st.info(f"⚖️ 안정적 ({trend_val:.1%})")
                    st.line_chart(df_trend, color="#808495")