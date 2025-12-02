"""
FINZ 추천시스템 - 프롬프트용 데이터 요약 유틸리티
로그 데이터와 만족도 데이터를 LLM 프롬프트용 텍스트로 변환
"""

import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

class PromptDataSummarizer:
    """로그/만족도 데이터를 LLM 프롬프트용 텍스트로 변환"""
    
    def __init__(self, log_df, satisfaction_df):
        self.log_df = log_df
        self.satisfaction_df = satisfaction_df
        
        # 타임스탬프를 datetime으로 변환
        self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'])
    
    def generate_log_summary(self, customer_id, days=7):
        """
        고객의 최근 N일 로그를 프롬프트용 텍스트로 요약
        
        Returns:
            str: LLM 프롬프트에 삽입할 로그 요약 텍스트
        """
        # 최근 N일 로그 필터링
        cutoff_date = datetime(2024, 11, 30) - timedelta(days=days)
        customer_logs = self.log_df[
            (self.log_df['customer_id'] == customer_id) & 
            (self.log_df['timestamp'] >= cutoff_date)
        ]
        
        if len(customer_logs) == 0:
            return "최근 활동 기록이 없습니다."
        
        # 행동별 집계
        action_summary = customer_logs.groupby(['action_type', 'product_type']).size().to_dict()
        
        # 자주 본 상품
        product_views = customer_logs.groupby(['product_name', 'product_type']).agg({
            'log_id': 'count',
            'duration_seconds': 'sum'
        }).reset_index()
        product_views.columns = ['product_name', 'product_type', 'view_count', 'total_duration']
        top_products = product_views.nlargest(3, 'view_count')
        
        # 카테고리별 관심도
        category_interest = customer_logs.groupby('product_category').size().sort_values(ascending=False)
        
        # 텍스트 생성
        summary_lines = [f"[최근 {days}일 활동 요약]"]
        
        # 1. 전체 활동량
        summary_lines.append(f"- 총 {len(customer_logs)}회 활동 (클릭 {action_summary.get(('click', 'card'), 0) + action_summary.get(('click', 'deposit'), 0)}회, 상세조회 {action_summary.get(('view', 'card'), 0) + action_summary.get(('view', 'deposit'), 0)}회)")
        
        # 2. 관심 상품
        if len(top_products) > 0:
            summary_lines.append("- 관심 상품:")
            for _, row in top_products.iterrows():
                ptype = "카드" if row['product_type'] == 'card' else "예적금"
                summary_lines.append(f"  · {row['product_name']} ({ptype}) - {row['view_count']}회 조회, 총 {row['total_duration']}초 체류")
        
        # 3. 관심 카테고리
        if len(category_interest) > 0:
            top_cats = category_interest.head(3).index.tolist()
            summary_lines.append(f"- 관심 카테고리: {', '.join(top_cats)}")
        
        # 4. 신청 시도 여부
        apply_count = len(customer_logs[customer_logs['action_type'] == 'apply'])
        if apply_count > 0:
            applied_products = customer_logs[customer_logs['action_type'] == 'apply']['product_name'].tolist()
            summary_lines.append(f"- 신청 시도: {', '.join(applied_products[:3])}")
        
        return '\n'.join(summary_lines)
    
    def generate_satisfaction_summary(self, customer_id, similar_customer_ids=None):
        """
        고객 및 유사 고객의 만족도 데이터를 프롬프트용 텍스트로 요약
        
        Args:
            customer_id: 대상 고객 ID
            similar_customer_ids: 유사 고객 ID 리스트 (KNN 결과)
            
        Returns:
            str: LLM 프롬프트에 삽입할 만족도 요약 텍스트
        """
        # 본인 만족도
        my_satisfaction = self.satisfaction_df[self.satisfaction_df['customer_id'] == customer_id]
        
        summary_lines = ["[만족도 분석]"]
        
        # 1. 본인 가입 상품 만족도
        if len(my_satisfaction) > 0:
            summary_lines.append("▶ 본인 가입 상품 평가:")
            high_rated = my_satisfaction[my_satisfaction['rating'] >= 4.0].nlargest(3, 'rating')
            for _, row in high_rated.iterrows():
                ptype = "카드" if row['product_type'] == 'card' else "예적금"
                factors = row['satisfaction_factors'].replace(',', ', ')
                summary_lines.append(f"  · {row['product_name']} ({ptype}): {row['rating']}점 - 만족요인: {factors}")
        else:
            summary_lines.append("▶ 본인 가입 상품: 없음 (신규 고객)")
        
        # 2. 유사 고객 만족도 분석
        if similar_customer_ids and len(similar_customer_ids) > 0:
            similar_satisfaction = self.satisfaction_df[
                self.satisfaction_df['customer_id'].isin(similar_customer_ids)
            ]
            
            if len(similar_satisfaction) > 0:
                summary_lines.append("\n▶ 유사 고객들의 선호 상품 (만족도 4.0 이상):")
                
                # 유사 고객들의 고만족 상품 집계
                high_rated_similar = similar_satisfaction[similar_satisfaction['rating'] >= 4.0]
                product_scores = high_rated_similar.groupby(['product_name', 'product_type']).agg({
                    'rating': 'mean',
                    'customer_id': 'count',
                    'would_recommend': 'mean'
                }).reset_index()
                product_scores.columns = ['product_name', 'product_type', 'avg_rating', 'user_count', 'recommend_rate']
                product_scores = product_scores.nlargest(5, 'user_count')
                
                for _, row in product_scores.iterrows():
                    ptype = "카드" if row['product_type'] == 'card' else "예적금"
                    summary_lines.append(
                        f"  · {row['product_name']} ({ptype}): "
                        f"평균 {row['avg_rating']:.1f}점, "
                        f"{int(row['user_count'])}명 이용, "
                        f"추천율 {row['recommend_rate']*100:.0f}%"
                    )
        
        return '\n'.join(summary_lines)
    
    def check_cold_start(self, customer_id, min_logs=5, min_products=2):
        """
        콜드스타트(신규/이력부족) 고객 여부 판단
        
        Returns:
            bool: True면 콜드스타트 고객
            str: 판단 근거
        """
        log_count = len(self.log_df[self.log_df['customer_id'] == customer_id])
        product_count = len(self.satisfaction_df[self.satisfaction_df['customer_id'] == customer_id])
        
        is_cold_start = (log_count < min_logs) or (product_count < min_products)
        
        reason = f"로그 {log_count}개, 가입상품 {product_count}개"
        if is_cold_start:
            reason += " → 콜드스타트 고객 (제로샷 추천 필요)"
        else:
            reason += " → 일반 고객"
        
        return is_cold_start, reason
    
    def generate_zeroshot_context(self, customer_row):
        """
        제로샷 추천을 위한 인구통계 기반 컨텍스트 생성
        
        Args:
            customer_row: customers 데이터프레임의 한 행
            
        Returns:
            str: 제로샷 프롬프트용 고객 컨텍스트
        """
        age = customer_row.get('AGE', 'Unknown')
        job_type = customer_row.get('JB_TP', 'Unknown')
        persona = customer_row.get('Persona_Cluster', 'Unknown')
        
        # 직업 코드 해석
        job_map = {
            420: '사무직/회사원',
            910: '학생',
            510: '전문직',
            410: '공무원',
            440: '자영업',
            520: '프리랜서',
            430: '기술직'
        }
        job_desc = job_map.get(job_type, '기타')
        
        # 페르소나 해석
        persona_map = {
            0: '안정추구형 (저위험 선호, 안정적 수익 중시)',
            1: '디지털네이티브 (모바일 친화, 간편한 서비스 선호)',
            2: '실속소비형 (혜택 중시, 가성비 추구)',
            3: '프리미엄지향 (품질 중시, 고급 서비스 선호)',
            4: '사회초년생 (첫 금융상품, 기초 재테크 관심)'
        }
        persona_desc = persona_map.get(persona, '미분류')
        
        context = f"""[고객 프로필 - 제로샷 추천용]
- 연령: {age}세
- 직업: {job_desc}
- 성향: {persona_desc}
- 상태: 금융 상품 가입 이력 부족 (신규 고객으로 추정)

[추천 가이드라인]
- 연회비/수수료가 낮은 입문용 상품 우선
- 해당 연령대/직업군에서 인기 있는 상품 고려
- 향후 업그레이드 가능한 상품 경로 제시"""
        
        return context


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    # 데이터 로드
    log_df = pd.read_csv('/mnt/user-data/outputs/customer_logs.csv')
    satisfaction_df = pd.read_csv('/mnt/user-data/outputs/customer_satisfaction.csv')
    customers_df = pd.read_csv('/mnt/user-data/uploads/customers_with_id.csv')
    
    # 요약기 초기화
    summarizer = PromptDataSummarizer(log_df, satisfaction_df)
    
    print("="*70)
    print("📝 프롬프트용 데이터 요약 예시")
    print("="*70)
    
    # 테스트 고객
    test_customer = 'C000001'
    similar_customers = ['C002488', 'C002937', 'C000181', 'C000210', 'C002061']  # KNN 결과
    
    # 1. 로그 요약
    print("\n[1] 로그 데이터 요약")
    print("-"*50)
    log_summary = summarizer.generate_log_summary(test_customer, days=30)
    print(log_summary)
    
    # 2. 만족도 요약
    print("\n[2] 만족도 데이터 요약")
    print("-"*50)
    satisfaction_summary = summarizer.generate_satisfaction_summary(test_customer, similar_customers)
    print(satisfaction_summary)
    
    # 3. 콜드스타트 체크
    print("\n[3] 콜드스타트 체크")
    print("-"*50)
    is_cold, reason = summarizer.check_cold_start(test_customer)
    print(f"고객 {test_customer}: {reason}")
    
    # 4. 제로샷 컨텍스트 (가정: 콜드스타트 고객인 경우)
    print("\n[4] 제로샷 컨텍스트 예시")
    print("-"*50)
    customer_row = customers_df[customers_df['customer_id'] == test_customer].iloc[0]
    zeroshot_context = summarizer.generate_zeroshot_context(customer_row)
    print(zeroshot_context)
