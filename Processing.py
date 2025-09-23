import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataPreprocessor:
    """치료 이력 포함 강화된 데이터 전처리 클래스"""
    
    def __init__(self, picture_strategy='zero', decibel_strategy='period_avg', lux_strategy='period_avg'):
        """
        누락 데이터 처리 전략 설정
        
        Parameters:
        -----------
        picture_strategy : str
            - 'zero': 누락시 0으로 설정
            - 'period_avg': 대상자 해당 기간 평균으로 대체
            - 'missing': 측정 안됨을 별도 값(-999)으로 표시
        """
        self.picture_strategy = picture_strategy
        self.decibel_strategy = decibel_strategy
        self.lux_strategy = lux_strategy
        
        print(f"누락 데이터 처리 전략: 사진={self._get_strategy_name(picture_strategy)}, "
            f"DECIBEL={self._get_strategy_name(decibel_strategy)}, LUX={self._get_strategy_name(lux_strategy)}")

    def _get_strategy_name(self, strategy):
        """전략 이름 변환"""
        names = {
            'zero': '0으로 설정', 
            'period_avg': '기간 평균 대체',
            'missing': '측정안됨(-999)'
        }
        return names.get(strategy, strategy)    
    
    def load_data(self, data_file, label_file):
        print("1: 데이터 로드")
        
        # 라벨 데이터 로드 (시트명이 'MZ'임)
        labels_df = pd.read_excel(label_file, sheet_name='MZ')
        if 'group' in labels_df.columns:
            labels_df['group'] = labels_df['group'].str.strip()
            print(f"라벨 분포: {labels_df['group'].value_counts().to_dict()}")
        
        # 각 데이터 시트 로드
        data_sheets = {}
        sheet_names = ['STEP', 'PICTURE', 'LUX', 'DECIBEL', 'JOIN SURVEY']
        
        for sheet in sheet_names:
            try:
                data_sheets[sheet] = pd.read_excel(data_file, sheet_name=sheet)
                print(f"{sheet}: {len(data_sheets[sheet])}개 레코드")
            except Exception as e:
                print(f"{sheet} 로드 실패: {e}")
                data_sheets[sheet] = pd.DataFrame()
        
        return data_sheets, labels_df
    
    def process_step_data(self, step_data):
        """Step 2: STEP 데이터 전처리"""
        print("\n=== Step 2: STEP 데이터 전처리 ===")
        
        clean_step_data = []
        
        for _, row in step_data.iterrows():
            # STEP_DATE를 기본으로 사용 (YYYYMMDD)
            step_date = int(row['STEP_DATE'])
            
            clean_step_data.append({
                'USER_ID': row['USER_ID'],
                'DATE': step_date,
                'DAILY_STEPS': row['STEP_COUNT']
            })
        
        print(f"정리된 STEP 데이터: {len(clean_step_data)}개 일별 행")
        return clean_step_data
    
    def extract_date_from_timestamp(self, timestamp):
        """타임스탬프에서 날짜 추출 (YYYYMMDDHHMM -> YYYYMMDD)"""
        timestamp_str = str(timestamp)
        if len(timestamp_str) >= 8:
            return int(timestamp_str[:8])
        return None
    
    def find_densest_7day_period(self, user_id, clean_step_data, lux_data, decibel_data, picture_data):
        """사용자별 가장 밀도 높은 7일 구간 찾기"""
        
        # 사용자의 모든 데이터 날짜 수집
        user_step_data = [r for r in clean_step_data if r['USER_ID'] == user_id]
        if len(user_step_data) < 3:
            return None
        
        # 사용자의 모든 활동 날짜 수집
        all_dates = set()
         
        # STEP 날짜들
        step_dates = [r['DATE'] for r in user_step_data]
        all_dates.update(step_dates)
        
        # LUX 날짜들
        if not lux_data.empty:
            user_lux = lux_data[lux_data['USER_ID'].astype(str) == str(user_id)]
            for _, row in user_lux.iterrows():
                if pd.notna(row.get('LUX_TIME')):
                    date = self.extract_date_from_timestamp(row['LUX_TIME'])
                    if date:
                        all_dates.add(date)
        
        # DECIBEL 날짜들
        if not decibel_data.empty:
            user_decibel = decibel_data[decibel_data['USER_ID'].astype(str) == str(user_id)]
            for _, row in user_decibel.iterrows():
                if pd.notna(row.get('DB_TIME')):
                    date = self.extract_date_from_timestamp(row['DB_TIME'])
                    if date:
                        all_dates.add(date)
        
        # PICTURE 날짜들
        if not picture_data.empty:
            user_pictures = picture_data[picture_data['USER_ID'].astype(str) == str(user_id)]
            for _, row in user_pictures.iterrows():
                if pd.notna(row.get('PICTURE_DATE')):
                    date = self.extract_date_from_timestamp(row['PICTURE_DATE'])
                    if date:
                        all_dates.add(date)
        
        # 날짜들을 정렬
        sorted_dates = sorted(list(all_dates))
        
        if len(sorted_dates) < 7:
            return sorted_dates
        
        # 모든 가능한 7일 구간에 대해 밀도 계산
        best_period = None
        best_density = 0
        
        for i in range(len(sorted_dates) - 6):
            period_dates = sorted_dates[i:i+7]
            
            start_date = datetime.strptime(str(period_dates[0]), '%Y%m%d')
            end_date = datetime.strptime(str(period_dates[-1]), '%Y%m%d')
            
            if (end_date - start_date).days <= 10:
                density = self.calculate_period_density(user_id, period_dates, 
                                                      clean_step_data, lux_data, decibel_data, picture_data)
                
                if density > best_density:
                    best_density = density
                    best_period = period_dates
        
        return best_period if best_period else sorted_dates[:7]
    
    def calculate_period_density(self, user_id, period_dates, clean_step_data, lux_data, decibel_data, picture_data):
        """특정 기간의 데이터 밀도 계산"""
        start_date = min(period_dates)
        end_date = max(period_dates)
        
        density_score = 0
        total_possible = len(period_dates)
        
        # STEP 데이터 밀도
        step_days = len([r for r in clean_step_data 
                        if r['USER_ID'] == user_id and start_date <= r['DATE'] <= end_date])
        step_density = step_days / total_possible
        density_score += step_density * 0.4
        
        # LUX 데이터 밀도
        if not lux_data.empty:
            user_lux = lux_data[lux_data['USER_ID'].astype(str) == str(user_id)]
            lux_dates = set()
            for _, row in user_lux.iterrows():
                if pd.notna(row.get('LUX_TIME')):
                    date = self.extract_date_from_timestamp(row['LUX_TIME'])
                    if date and start_date <= date <= end_date:
                        lux_dates.add(date)
            lux_density = len(lux_dates) / total_possible
            density_score += lux_density * 0.2
        
        # DECIBEL 데이터 밀도
        if not decibel_data.empty:
            user_decibel = decibel_data[decibel_data['USER_ID'].astype(str) == str(user_id)]
            decibel_dates = set()
            for _, row in user_decibel.iterrows():
                if pd.notna(row.get('DB_TIME')):
                    date = self.extract_date_from_timestamp(row['DB_TIME'])
                    if date and start_date <= date <= end_date:
                        decibel_dates.add(date)
            decibel_density = len(decibel_dates) / total_possible
            density_score += decibel_density * 0.2
        
        # PICTURE 데이터 밀도
        if not picture_data.empty:
            user_pictures = picture_data[picture_data['USER_ID'].astype(str) == str(user_id)]
            picture_dates = set()
            for _, row in user_pictures.iterrows():
                if pd.notna(row.get('PICTURE_DATE')):
                    date = self.extract_date_from_timestamp(row['PICTURE_DATE'])
                    if date and start_date <= date <= end_date:
                        picture_dates.add(date)
            picture_density = len(picture_dates) / total_possible
            density_score += picture_density * 0.2
        
        return density_score
    
    def create_7day_features(self, clean_step_data, lux_data, decibel_data, picture_data, all_user_ids):
        """Step 3: 밀도 기반 7일 특성 생성"""
        print("\n=== Step 3: 밀도 기반 7일 특성 생성 ===")
        
        all_features = {}
        processed_count = excluded_count = 0
        
        for user_id in all_user_ids:
            best_period = self.find_densest_7day_period(user_id, clean_step_data, lux_data, decibel_data, picture_data)
            
            if best_period is None:
                excluded_count += 1
                continue
            
            features = self.extract_7day_features(user_id, best_period, clean_step_data, lux_data, decibel_data, picture_data)
            
            if features:
                all_features[user_id] = features
                processed_count += 1
            else:
                excluded_count += 1
        
        print(f"처리된 사용자: {processed_count}명, 제외된 사용자: {excluded_count}명")
        return all_features
    
    def extract_7day_features(self, user_id, best_period, clean_step_data, lux_data, decibel_data, picture_data):
        """특정 7일 기간의 모든 특성 추출"""
        
        start_date = min(best_period)
        end_date = max(best_period)
        
        base_start = datetime.strptime(str(start_date), '%Y%m%d')
        seven_days = [(base_start + timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]
        seven_days = [int(date) for date in seven_days]
        
        features = {}
        # 1. 걸음수 특성 (기존과 동일)
        user_step_dict = {r['DATE']: r['DAILY_STEPS'] for r in clean_step_data if r['USER_ID'] == user_id}
        period_steps = [user_step_dict[d] for d in user_step_dict.keys() if start_date <= d <= end_date]
        period_step_avg = np.mean(period_steps) if period_steps else 0
        
        daily_steps = []
        for day in seven_days:
            if day in user_step_dict:
                daily_steps.append(user_step_dict[day])
            else:
                daily_steps.append(period_step_avg)
        
        for i, steps in enumerate(daily_steps):
            features[f'day{i+1}_steps'] = steps
    
        # LUX 특성 (전략 적용)
        daily_lux = self._process_sensor_data_with_strategy(
            user_id, seven_days, start_date, end_date, lux_data, 'LUX_TIME', 'LUX_VALUE', self.lux_strategy
        )
        for i, lux in enumerate(daily_lux):
            features[f'day{i+1}_lux'] = lux
        
        # DECIBEL 특성 (전략 적용)
        daily_decibel = self._process_sensor_data_with_strategy(
            user_id, seven_days, start_date, end_date, decibel_data, 'DB_TIME', 'DB_VALUE', self.decibel_strategy
        )
        for i, decibel in enumerate(daily_decibel):
            features[f'day{i+1}_decibel'] = decibel
        
        # PICTURE 특성 (전략 적용)
        daily_pictures = self._process_picture_data_with_strategy(
            user_id, seven_days, start_date, end_date, picture_data, self.picture_strategy
        )
        for i, pic_count in enumerate(daily_pictures):
            features[f'day{i+1}_pictures'] = pic_count
        
        # 요약 통계 (missing 값 제외하고 계산)
        clean_lux = [x for x in daily_lux if x != -999]
        clean_decibel = [x for x in daily_decibel if x != -999]
        clean_pictures = [x for x in daily_pictures if x != -999]
        
        features['avg_lux'] = np.mean(clean_lux) if clean_lux else 0
        features['std_lux'] = np.std(clean_lux) if len(clean_lux) > 1 else 0
        features['avg_decibel'] = np.mean(clean_decibel) if clean_decibel else 0
        features['std_decibel'] = np.std(clean_decibel) if len(clean_decibel) > 1 else 0
        features['avg_pictures'] = np.mean(clean_pictures) if clean_pictures else 0
        features['std_pictures'] = np.std(clean_pictures) if len(clean_pictures) > 1 else 0
        
        return features

    def _process_sensor_data_with_strategy(self, user_id, seven_days, start_date, end_date, sensor_data, time_col, value_col, strategy):
        """센서 데이터 처리 (3가지 전략 지원)"""
        if sensor_data.empty:
            return [0 if strategy == 'zero' else -999 if strategy == 'missing' else 0] * 7
        
        user_sensor = sensor_data[sensor_data['USER_ID'].astype(str) == str(user_id)]
        if len(user_sensor) == 0:
            return [0 if strategy == 'zero' else -999 if strategy == 'missing' else 0] * 7
        
        # period_avg 전략을 위한 대체값 계산
        period_values = []
        for _, row in user_sensor.iterrows():
            if pd.notna(row.get(time_col)):
                date = self.extract_date_from_timestamp(row[time_col])
                if date and start_date <= date <= end_date:
                    period_values.append(row[value_col])
        
        period_avg_value = np.mean(period_values) if period_values else 0
        
        # 전략별 대체값 설정
        if strategy == 'zero':
            default_value = 0
        elif strategy == 'period_avg':
            default_value = period_avg_value
        elif strategy == 'missing':
            default_value = -999  # 측정 안됨을 나타내는 특수값
        else:
            default_value = 0
        
        # 일별 값 생성
        daily_values = []
        for day in seven_days:
            day_values = []
            for _, row in user_sensor.iterrows():
                if pd.notna(row.get(time_col)):
                    date = self.extract_date_from_timestamp(row[time_col])
                    if date == day:
                        day_values.append(row[value_col])
            
            if day_values:
                daily_values.append(np.mean(day_values))
            else:
                daily_values.append(default_value)
        
        return daily_values

    def _process_picture_data_with_strategy(self, user_id, seven_days, start_date, end_date, picture_data, strategy):
        """사진 데이터 처리 (3가지 전략 지원)"""
        if picture_data.empty:
            return [0 if strategy == 'zero' else -999 if strategy == 'missing' else 0] * 7
        
        user_pictures = picture_data[picture_data['USER_ID'].astype(str) == str(user_id)]
        if len(user_pictures) == 0:
            return [0 if strategy == 'zero' else -999 if strategy == 'missing' else 0] * 7
        
        # period_avg 전략을 위한 계산
        period_counts = {}
        for _, row in user_pictures.iterrows():
            if pd.notna(row.get('PICTURE_DATE')):
                date = self.extract_date_from_timestamp(row['PICTURE_DATE'])
                if date and start_date <= date <= end_date:
                    period_counts[date] = period_counts.get(date, 0) + 1
        
        period_avg_value = np.mean(list(period_counts.values())) if period_counts else 0
        
        # 전략별 대체값 설정
        if strategy == 'zero':
            default_value = 0
        elif strategy == 'period_avg':
            default_value = period_avg_value
        elif strategy == 'missing':
            default_value = -999
        else:
            default_value = 0
        
        # 일별 값 생성
        daily_pictures = []
        for day in seven_days:
            day_picture_count = 0
            has_data = False
            
            for _, row in user_pictures.iterrows():
                if pd.notna(row.get('PICTURE_DATE')):
                    date = self.extract_date_from_timestamp(row['PICTURE_DATE'])
                    if date == day:
                        day_picture_count += 1
                        has_data = True
            
            if has_data:
                daily_pictures.append(day_picture_count)
            else:
                daily_pictures.append(default_value)
        
        return daily_pictures
    def create_medical_features(self, join_survey_data, all_user_ids):
        """Step 4: 의료/상담/치료 특성 생성 (단순화)"""
        print("\n=== Step 4: 의료/상담/치료 특성 생성 ===")
        
        medical_features = {}
        
        for user_id in all_user_ids:
            features = {}
            
            if join_survey_data.empty:
                # 기본값 설정
                features['consult_level'] = 0
                features['has_treatment'] = 0
                features['has_past_diagnosis'] = 0
            else:
                user_survey = join_survey_data[join_survey_data['USER_ID'].astype(str) == str(user_id)]
                
                if len(user_survey) == 0:
                    # 해당 사용자 데이터 없음
                    features['consult_level'] = 0
                    features['has_treatment'] = 0
                    features['has_past_diagnosis'] = 0
                else:
                    user_data = user_survey.iloc[0]
                    
                    # 1. 상담 레벨 (0,1,2,3,4로 수정)
                    consult_value = user_data.get('USER_CONSULT', '없음')
                    consult_str = str(consult_value).strip() if pd.notna(consult_value) else '없음'
                    
                    if consult_str == '없음' or pd.isna(consult_value):
                        consult_level = 0
                    elif '1~5회' in consult_str:
                        consult_level = 1
                    elif '6~10회' in consult_str:
                        consult_level = 2
                    elif '11회 이상' in consult_str:
                        if '지속적' in consult_str:
                            consult_level = 4  # 11회 이상 + 지속적
                        else:
                            consult_level = 3  # 11회 이상
                    else:
                        consult_level = 0
                    
                    features['consult_level'] = consult_level
                    
                    # 2. 치료 여부 (이진)
                    treat_value = user_data.get('USER_TREAT', '아니요')
                    features['has_treatment'] = 1 if str(treat_value).strip() == '예' else 0
                    
                    # 3. 과거 진단 경험 (단순 이진) - 뭔가 적혀있으면 1, 없으면 0
                    diagnosis_value = user_data.get('USER_TREAT_CATEGORY', '없음')
                    diagnosis_str = str(diagnosis_value).strip() if pd.notna(diagnosis_value) else '없음'
                    
                    if diagnosis_str in ['없음', '', 'nan', 'NaN'] or pd.isna(diagnosis_value):
                        features['has_past_diagnosis'] = 0
                    else:
                        features['has_past_diagnosis'] = 1
            
            medical_features[user_id] = features
        
        # 통계 출력
        consult_counts = {}
        treatment_count = 0
        diagnosis_count = 0
        
        for user_features in medical_features.values():
            level = user_features['consult_level']
            consult_counts[level] = consult_counts.get(level, 0) + 1
            
            if user_features['has_treatment'] == 1:
                treatment_count += 1
                
            if user_features['has_past_diagnosis'] == 1:
                diagnosis_count += 1
        
        print(f"\n상담 레벨 분포:")
        level_names = {0: "없음", 1: "1-5회", 2: "6-10회", 3: "11회 이상", 4: "11회 이상+지속적"}
        for level, count in sorted(consult_counts.items()):
            print(f"  {level_names.get(level, f'레벨{level}')}: {count}명")
        
        print(f"치료 받은 경험: {treatment_count}명")
        print(f"과거 진단 경험: {diagnosis_count}명")
        
        return medical_features
    
    def combine_features(self, seven_day_features, medical_features, labels_df):
        """Step 5: 특성 통합"""
        print("\n=== Step 5: 특성 통합 ===")
        
        # 교집합 확인
        seven_day_users = set(seven_day_features.keys())
        medical_users = set(medical_features.keys())
        label_users = set(labels_df['id'].values)
        
        print(f"7일 특성 사용자: {len(seven_day_users)}명")
        print(f"의료 특성 사용자: {len(medical_users)}명")
        print(f"라벨 파일 사용자: {len(label_users)}명")
        
        final_users = seven_day_users & medical_users & label_users
        print(f"최종 사용자: {len(final_users)}명")
        
        feature_matrix = []
        labels = []
        user_ids = []
        
        for user_id in final_users:
            # 7일 특성과 의료 특성 통합
            features = {}
            features.update(seven_day_features[user_id])
            features.update(medical_features[user_id])
            
            user_label = labels_df[labels_df['id'] == user_id]['group'].iloc[0]
            
            feature_matrix.append(list(features.values()))
            labels.append(user_label)
            user_ids.append(user_id)
        
        # 특성 이름들 생성
        if final_users:
            sample_user = next(iter(final_users))
            sample_features = {}
            sample_features.update(seven_day_features[sample_user])
            sample_features.update(medical_features[sample_user])
            feature_names = list(sample_features.keys())
        else:
            feature_names = []
        
        print(f"총 특성 수: {len(feature_names)}개")
        print("특성 구성:")
        print("  - 일별 걸음수 (7개): day1_steps ~ day7_steps")
        print("  - 일별 LUX (7개): day1_lux ~ day7_lux")
        print("  - 일별 DECIBEL (7개): day1_decibel ~ day7_decibel")
        print("  - 일별 사진수 (7개): day1_pictures ~ day7_pictures")
        print("  - 요약 통계 (9개): avg/std 특성들")
        print("  - 상담 레벨 (1개, 0-4): consult_level")
        print("  - 치료 여부 (1개, 0-1): has_treatment")
        print("  - 과거 진단 (1개, 0-1): has_past_diagnosis")
        
        return np.array(feature_matrix), np.array(labels), user_ids, feature_names


class RandomForestTrainer:
    """랜덤포레스트 학습 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
    
    def train_model(self, X, y, feature_names):
        """랜덤포레스트 모델 학습"""
        print("\n=== 랜덤포레스트 모델 학습 ===")
        
        # 라벨 인코딩
        y_encoded = self.label_encoder.fit_transform(y)
        label_names = self.label_encoder.classes_
        print(f"클래스: {label_names}")
        
        # 클래스 분포 확인
        unique_labels, label_counts = np.unique(y, return_counts=True)
        print("\n클래스 분포:")
        for label, count in zip(unique_labels, label_counts):
            print(f"  {label}: {count}명 ({count/len(y)*100:.1f}%)")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 랜덤포레스트 학습
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n=== 모델 성능 ===")
        print(f"훈련 정확도: {self.model.score(X_train_scaled, y_train):.3f}")
        print(f"테스트 정확도: {self.model.score(X_test_scaled, y_test):.3f}")
        
        # 교차 검증
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"5-fold CV 평균: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        # 분류 리포트
        print("\n분류 리포트:")
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n특성 중요도 TOP 15:")
        print(feature_importance.head(15))
        
        return self.model, feature_importance


class PipelineRunner:
    """완전한 파이프라인 실행 클래스"""
    
    def __init__(self, picture_strategy='zero', decibel_strategy='period_avg', lux_strategy='period_avg'):
        self.preprocessor = EnhancedDataPreprocessor(picture_strategy, decibel_strategy, lux_strategy)
        self.trainer = RandomForestTrainer()
    
    def save_processed_data(self, X, y, user_ids, feature_names):
        """전처리된 데이터 저장"""
        print("\n=== 전처리 데이터 저장 ===")
        
        processed_df = pd.DataFrame(X, columns=feature_names)
        processed_df.insert(0, 'USER_ID', user_ids)
        processed_df.insert(1, 'GROUP', y)
        
        # 상담/치료 의미 추가
        level_meaning = {0: "없음", 1: "1-5회", 2: "6-10회", 3: "11회 이상", 4: "11회 이상+지속적"}
        if 'consult_level' in processed_df.columns:
            processed_df['상담레벨의미'] = processed_df['consult_level'].map(level_meaning)
        
        if 'has_treatment' in processed_df.columns:
            processed_df['치료여부의미'] = processed_df['has_treatment'].map({0: "없음", 1: "있음"})
            
        if 'has_past_diagnosis' in processed_df.columns:
            processed_df['과거진단의미'] = processed_df['has_past_diagnosis'].map({0: "없음", 1: "있음"})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"단순화_바이오마커_데이터_{timestamp}.xlsx"
        processed_df.to_excel(filename, index=False)
        
        print(f"저장 완료: {filename}")
        return filename
    
    def run_pipeline(self, data_file, label_file):
        """전체 파이프라인 실행"""
        print("완전한 디지털 바이오마커 분류 파이프라인 시작")
        print("포함 특성:")
        print("  - 밀도 기반 최적 7일 구간 선택")
        print("  - 7일간 일별 패턴 (걸음수, LUX, DECIBEL, 사진수)")
        print("  - 상담 레벨 (0,1,2,3)")
        print("  - 치료 여부 (0,1)")
        print("  - 진단명별 이진 특성")
        
        # 1. 데이터 로드
        data_sheets, labels_df = self.preprocessor.load_data(data_file, label_file)
        
        # 2. STEP 데이터 전처리
        clean_step_data = self.preprocessor.process_step_data(data_sheets['STEP'])
        all_user_ids = labels_df['id'].unique()
        
        # 3. 밀도 기반 7일 특성 생성
        seven_day_features = self.preprocessor.create_7day_features(
            clean_step_data, data_sheets['LUX'], data_sheets['DECIBEL'], 
            data_sheets['PICTURE'], all_user_ids
        )
        
        # 4. 의료/상담/치료 특성 생성
        medical_features = self.preprocessor.create_medical_features(
            data_sheets['JOIN SURVEY'], all_user_ids
        )
        
        # 5. 특성 통합
        X, y, user_ids, feature_names = self.preprocessor.combine_features(
            seven_day_features, medical_features, labels_df
        )
        
        if len(X) == 0:
            print("처리된 데이터가 없습니다.")
            return None, None, None, None, None
        
        # 6. 모델 학습
        model, feature_importance = self.trainer.train_model(X, y, feature_names)
        
        # 7. 데이터 저장
        self.save_processed_data(X, y, user_ids, feature_names)
        
        # 8. 결과 분석
        self.analyze_results(feature_importance)
        
        print("\n완전한 파이프라인 완료!")
        return model, feature_importance, X, y, user_ids
    
    def analyze_results(self, feature_importance):
        """상세 결과 분석"""
        print("\n변수별 중요도 분석:")
        
        # 카테고리별 중요도 집계
        categories = {
            'steps': [f for f in feature_importance['feature'] if 'steps' in f],
            'lux': [f for f in feature_importance['feature'] if 'lux' in f],
            'decibel': [f for f in feature_importance['feature'] if 'decibel' in f],
            'pictures': [f for f in feature_importance['feature'] if 'pictures' in f],
            'medical': [f for f in feature_importance['feature'] if f in ['consult_level', 'has_treatment'] or f.startswith('diagnosis_')]
        }
        
        category_importance = {}
        for category, features in categories.items():
            importance_sum = feature_importance[feature_importance['feature'].isin(features)]['importance'].sum()
            category_importance[category] = importance_sum
        
        # 중요도순으로 정렬
        sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n📊 카테고리별 중요도:")
        category_names = {
            'steps': '걸음수',
            'lux': '조도(LUX)', 
            'decibel': '소음(DECIBEL)',
            'pictures': '사진수',
            'medical': '의료/상담'
        }
        
        for i, (category, importance) in enumerate(sorted_categories, 1):
            print(f"   {i}. {category_names[category]}: {importance:.3f}")
        
        # 의료 특성 세부 분석
        medical_features = feature_importance[
            (feature_importance['feature'] == 'consult_level') |
            (feature_importance['feature'] == 'has_treatment') |
            (feature_importance['feature'] == 'has_past_diagnosis')
        ].sort_values('importance', ascending=False)
        
        if len(medical_features) > 0:
            print("\n💊 의료 특성별 중요도:")
            for _, row in medical_features.iterrows():
                feature_name = row['feature']
                if feature_name == 'consult_level':
                    display_name = '상담 레벨 (0-4)'
                elif feature_name == 'has_treatment':
                    display_name = '치료 여부'
                elif feature_name == 'has_past_diagnosis':
                    display_name = '과거 진단 경험'
                else:
                    display_name = feature_name
                
                print(f"   {display_name}: {row['importance']:.3f}")
        
        print(f"\n분석 결과:")
        top_category = sorted_categories[0][0]
        print(f"   {category_names[top_category]}이 가장 중요한 예측 인자입니다!")


# 실행 예시
if __name__ == "__main__":
    # 다양한 설정으로 실행 가능
    
    # 설정 1: 사진은 0, 나머지는 기간 평균 (기본)
    #print("=== 설정 1: 기본 (사진=0, 센서=기간평균) ===")
    runner1 = PipelineRunner(picture_strategy='period_avg', decibel_strategy='period_avg', lux_strategy='period_avg') #0.757 (±0.114)
    
    # 설정 2: 모든 변수를 기간 평균으로 대체
    #print("\n=== 설정 2: 모든 변수 기간평균 대체 ===")
    #runner2 = PipelineRunner(picture_strategy='period_avg', decibel_strategy='period_avg', lux_strategy='period_avg')# 5-fold CV 평균: 0.743 (±0.146)
    runner2 = PipelineRunner(picture_strategy='zero', decibel_strategy='zero', lux_strategy='zero') #5-fold CV 평균: 0.786 (±0.090)
    runner_missing = PipelineRunner(
    picture_strategy='missing', 
    decibel_strategy='missing', 
    lux_strategy='missing'
    )
    # 파일 경로 설정
    data_file = r"C:\Users\parkm\OneDrive - dgu.ac.kr\AI_LAB\U-health\lifelog\Lifelog\data\231123-DATA.xlsx"
    label_file = r"C:\Users\parkm\OneDrive - dgu.ac.kr\AI_LAB\U-health\lifelog\Lifelog\data\231123-LABEL.xlsx"
    
    try:
        # 원하는 설정 선택해서 실행
        model, feature_importance, X, y, user_ids = runner1.run_pipeline(data_file, label_file)
        
        if model is not None:
            print(f"\n성공적으로 완료되었습니다!")
            print(f"최종 데이터셋: {len(X)}명의 사용자, {X.shape[1]}개 특성")
            print("특성 구성:")
            print("  - 7일 생체신호 패턴: 56개 (28개 값 + 28개 측정여부)")
            print("  - 요약 통계: 13개")
            print("  - 상담/치료 정보: 3개")
        else:
            print("파이프라인 실행 중 오류가 발생했습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        print("파일 경로와 데이터 형식을 확인해주세요.")