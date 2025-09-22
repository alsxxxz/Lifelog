import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

class EnhancedPatternVisualizer:
    """완전한 파이프라인용 패턴 시각화 디버거"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def analyze_user_pattern(self, user_id, data_sheets, labels_df):
        """특정 사용자의 완전한 패턴 분석 및 시각화"""
        
        print(f"=== 사용자 {user_id} 완전한 패턴 분석 ===")
        
        # 1. 전처리된 STEP 데이터 생성
        clean_step_data = self.preprocessor.process_step_data(data_sheets['STEP'])
        
        # 2. 밀도 기반 최적 7일 구간 찾기
        best_period = self.preprocessor.find_densest_7day_period(
            user_id, clean_step_data, data_sheets['LUX'], 
            data_sheets['DECIBEL'], data_sheets['PICTURE']
        )
        
        if best_period is None:
            print(f"❌ 사용자 {user_id}는 분석할 데이터가 충분하지 않습니다.")
            return None
        
        # 3. 7일 특성 추출
        seven_day_features = self.preprocessor.extract_7day_features(
            user_id, best_period, clean_step_data, 
            data_sheets['LUX'], data_sheets['DECIBEL'], data_sheets['PICTURE']
        )
        
        # 4. 의료 특성 추출
        medical_features = {}
        if not data_sheets['JOIN SURVEY'].empty:
            user_survey = data_sheets['JOIN SURVEY'][
                data_sheets['JOIN SURVEY']['USER_ID'].astype(str) == str(user_id)
            ]
            
            if len(user_survey) > 0:
                user_data = user_survey.iloc[0]
                
                # 상담 레벨 (0-4)
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
                        consult_level = 4
                    else:
                        consult_level = 3
                else:
                    consult_level = 0
                
                # 치료 여부
                treat_value = user_data.get('USER_TREAT', '아니요')
                has_treatment = 1 if str(treat_value).strip() == '예' else 0
                
                # 과거 진단 여부
                diagnosis_value = user_data.get('USER_TREAT_CATEGORY', '없음')
                diagnosis_str = str(diagnosis_value).strip() if pd.notna(diagnosis_value) else '없음'
                has_past_diagnosis = 0 if diagnosis_str in ['없음', '', 'nan'] else 1
                
                medical_features = {
                    'consult_level': consult_level,
                    'consult_raw': consult_str,
                    'has_treatment': has_treatment,
                    'has_past_diagnosis': has_past_diagnosis,
                    'diagnosis_raw': diagnosis_str
                }
            else:
                medical_features = {
                    'consult_level': 0,
                    'consult_raw': '없음',
                    'has_treatment': 0,
                    'has_past_diagnosis': 0,
                    'diagnosis_raw': '없음'
                }
        
        # 5. 라벨 확인
        user_label = self._get_user_label(user_id, labels_df)
        
        # 7. 상세 패턴 분석
        detailed_patterns = self._extract_detailed_patterns(
            user_id, best_period, clean_step_data, data_sheets
        )
        
        # 8. 시각화
        self._visualize_complete_pattern(
            user_id, best_period, detailed_patterns, seven_day_features, 
            medical_features, user_label
        )
        
        return {
            'best_period': best_period,
            'seven_day_features': seven_day_features,
            'medical_features': medical_features,
            'detailed_patterns': detailed_patterns,
            'label': user_label
        }
    
    def _extract_detailed_patterns(self, user_id, best_period, clean_step_data, data_sheets):
        """상세한 일별 패턴 추출 (향상된 대체 전략 반영)"""
        
        start_date = min(best_period)
        end_date = max(best_period)
        
        # 연속된 7일 생성
        base_start = datetime.strptime(str(start_date), '%Y%m%d')
        seven_days = [(base_start + timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]
        seven_days = [int(date) for date in seven_days]
        
        patterns = {
            'dates': seven_days,
            'steps': {'values': [], 'sources': []},
            'lux': {'values': [], 'sources': []},
            'decibel': {'values': [], 'sources': []},
            'pictures': {'values': [], 'sources': []}
        }
        
        # 걸음수 패턴
        user_step_dict = {r['DATE']: r['DAILY_STEPS'] for r in clean_step_data if r['USER_ID'] == user_id}
        period_steps = [user_step_dict[d] for d in user_step_dict.keys() if start_date <= d <= end_date]
        period_step_avg = np.mean(period_steps) if period_steps else 0
        
        for day in seven_days:
            if day in user_step_dict:
                patterns['steps']['values'].append(user_step_dict[day])
                patterns['steps']['sources'].append('실제')
            else:
                patterns['steps']['values'].append(period_step_avg)
                patterns['steps']['sources'].append('기간평균')
        
        # LUX 패턴 (향상된 전략)
        lux_values_sources = self._get_sensor_pattern_with_strategy(
            user_id, seven_days, start_date, end_date, data_sheets['LUX'], 'LUX_TIME', 'LUX_VALUE'
        )
        for value, source in lux_values_sources:
            patterns['lux']['values'].append(value)
            patterns['lux']['sources'].append(source)
        
        # DECIBEL 패턴 (향상된 전략)
        decibel_values_sources = self._get_sensor_pattern_with_strategy(
            user_id, seven_days, start_date, end_date, data_sheets['DECIBEL'], 'DB_TIME', 'DB_VALUE'
        )
        for value, source in decibel_values_sources:
            patterns['decibel']['values'].append(value)
            patterns['decibel']['sources'].append(source)
        
        # 사진수 패턴 (향상된 전략)
        picture_values_sources = self._get_picture_pattern_with_strategy(
            user_id, seven_days, start_date, end_date, data_sheets['PICTURE']
        )
        for value, source in picture_values_sources:
            patterns['pictures']['values'].append(value)
            patterns['pictures']['sources'].append(source)
        
        return patterns
    
    def _get_sensor_pattern_with_strategy(self, user_id, seven_days, start_date, end_date, sensor_data, time_col, value_col):
        """센서 패턴 추출 (향상된 전략 반영)"""
        if sensor_data.empty:
            return [(0, '없음')] * 7
        
        user_sensor = sensor_data[sensor_data['USER_ID'].astype(str) == str(user_id)]
        if len(user_sensor) == 0:
            return [(0, '없음')] * 7
        
        # 기간 내 측정일 계산
        period_dates = []
        period_values = []
        for _, row in user_sensor.iterrows():
            if pd.notna(row.get(time_col)):
                date = self._extract_date_from_timestamp(row[time_col])
                if date and start_date <= date <= end_date:
                    period_dates.append(date)
                    period_values.append(row[value_col])
        
        # 전체 측정값
        all_values = []
        for _, row in user_sensor.iterrows():
            if pd.notna(row.get(time_col)) and pd.notna(row.get(value_col)):
                all_values.append(row[value_col])
        
        # 대체 전략 결정
        if len(set(period_dates)) == 0:
            default_value = np.mean(all_values) if len(all_values) > 0 else 0
            strategy = '전체평균' if len(all_values) > 0 else '없음'
        elif len(set(period_dates)) <= 2:
            default_value = np.mean(all_values) if len(all_values) > 0 else np.mean(period_values)
            strategy = '전체평균(기간부족)'
        else:
            default_value = np.mean(period_values)
            strategy = '기간평균'
        
        # 일별 값 생성
        results = []
        for day in seven_days:
            day_values = []
            for _, row in user_sensor.iterrows():
                if pd.notna(row.get(time_col)):
                    date = self._extract_date_from_timestamp(row[time_col])
                    if date == day:
                        day_values.append(row[value_col])
            
            if day_values:
                results.append((np.mean(day_values), f'실제({len(day_values)}회)'))
            else:
                results.append((default_value, strategy))
        
        return results
    
    def _get_picture_pattern_with_strategy(self, user_id, seven_days, start_date, end_date, picture_data):
        """사진 패턴 추출 (향상된 전략 반영)"""
        if picture_data.empty:
            return [(0, '없음')] * 7
        
        user_pictures = picture_data[picture_data['USER_ID'].astype(str) == str(user_id)]
        if len(user_pictures) == 0:
            return [(0, '없음')] * 7
        
        # 기간 내 일별 사진 수
        period_counts = {}
        for _, row in user_pictures.iterrows():
            if pd.notna(row.get('PICTURE_DATE')):
                date = self._extract_date_from_timestamp(row['PICTURE_DATE'])
                if date and start_date <= date <= end_date:
                    period_counts[date] = period_counts.get(date, 0) + 1
        
        # 전체 일별 사진 수
        all_counts = {}
        for _, row in user_pictures.iterrows():
            if pd.notna(row.get('PICTURE_DATE')):
                date = self._extract_date_from_timestamp(row['PICTURE_DATE'])
                if date:
                    all_counts[date] = all_counts.get(date, 0) + 1
        
        # 대체 전략 결정
        if len(period_counts) == 0:
            default_value = np.mean(list(all_counts.values())) if len(all_counts) > 0 else 0
            strategy = '전체평균' if len(all_counts) > 0 else '없음'
        elif len(period_counts) <= 2:
            default_value = np.mean(list(all_counts.values())) if len(all_counts) > 0 else np.mean(list(period_counts.values()))
            strategy = '전체평균(기간부족)'
        else:
            default_value = np.mean(list(period_counts.values()))
            strategy = '기간평균'
        
        # 일별 값 생성
        results = []
        for day in seven_days:
            day_count = 0
            has_data = False
            
            for _, row in user_pictures.iterrows():
                if pd.notna(row.get('PICTURE_DATE')):
                    date = self._extract_date_from_timestamp(row['PICTURE_DATE'])
                    if date == day:
                        day_count += 1
                        has_data = True
            
            if has_data:
                results.append((day_count, '실제'))
            else:
                results.append((default_value, strategy))
        
        return results
    
    def _extract_date_from_timestamp(self, timestamp):
        """타임스탬프에서 날짜 추출"""
        timestamp_str = str(timestamp)
        if len(timestamp_str) >= 8:
            return int(timestamp_str[:8])
        return None
    
    def _get_user_label(self, user_id, labels_df):
        """사용자 라벨 확인"""
        user_label_data = labels_df[labels_df['id'] == user_id]
        if len(user_label_data) > 0:
            return user_label_data['group'].iloc[0]
        else:
            return '라벨없음'
    
    def _visualize_complete_pattern(self, user_id, best_period, detailed_patterns, 
                                   seven_day_features, medical_features, user_label):
        """완전한 패턴 시각화"""
        
        fig = plt.figure(figsize=(18, 14))
        
        # 날짜 라벨 생성
        date_labels = [f"Day{i+1}\n({date})" for i, date in enumerate(detailed_patterns['dates'])]
        
        # 색상 매핑
        color_map = {'실제': '#2E8B57', '평균': '#FF6B6B', '없음': '#CCCCCC'}
        
        # 1. 걸음수 패턴
        plt.subplot(3, 3, 1)
        colors = [color_map.get(source.split('(')[0], '#2E8B57') for source in detailed_patterns['steps']['sources']]
        plt.bar(range(7), detailed_patterns['steps']['values'], color=colors)
        plt.title('걸음수 (Steps)', fontweight='bold')
        plt.xticks(range(7), date_labels, rotation=45, ha='right')
        plt.ylabel('걸음수')
        
        # 소스 정보 표시
        for i, (value, source) in enumerate(zip(detailed_patterns['steps']['values'], detailed_patterns['steps']['sources'])):
            plt.text(i, value + max(detailed_patterns['steps']['values']) * 0.02, 
                    source, ha='center', va='bottom', fontsize=8,
                    color='red' if source == '평균' else 'black')
        
        # 2. LUX 패턴
        plt.subplot(3, 3, 2)
        colors = [color_map.get(source.split('(')[0], '#2E8B57') for source in detailed_patterns['lux']['sources']]
        plt.bar(range(7), detailed_patterns['lux']['values'], color=colors)
        plt.title('조도 (LUX)', fontweight='bold')
        plt.xticks(range(7), date_labels, rotation=45, ha='right')
        plt.ylabel('LUX')
        
        for i, (value, source) in enumerate(zip(detailed_patterns['lux']['values'], detailed_patterns['lux']['sources'])):
            if max(detailed_patterns['lux']['values']) > 0:
                plt.text(i, value + max(detailed_patterns['lux']['values']) * 0.02,
                        source, ha='center', va='bottom', fontsize=8,
                        color='red' if source == '평균' else 'black')
        
        # 3. DECIBEL 패턴
        plt.subplot(3, 3, 3)
        colors = [color_map.get(source.split('(')[0], '#2E8B57') for source in detailed_patterns['decibel']['sources']]
        plt.bar(range(7), detailed_patterns['decibel']['values'], color=colors)
        plt.title('소음 (DECIBEL)', fontweight='bold')
        plt.xticks(range(7), date_labels, rotation=45, ha='right')
        plt.ylabel('dB')
        
        for i, (value, source) in enumerate(zip(detailed_patterns['decibel']['values'], detailed_patterns['decibel']['sources'])):
            if max(detailed_patterns['decibel']['values']) > 0:
                plt.text(i, value + max(detailed_patterns['decibel']['values']) * 0.02,
                        source, ha='center', va='bottom', fontsize=8,
                        color='red' if source == '평균' else 'black')
        
        # 4. 사진수 패턴
        plt.subplot(3, 3, 4)
        colors = [color_map.get(source.split('(')[0], '#2E8B57') for source in detailed_patterns['pictures']['sources']]
        plt.bar(range(7), detailed_patterns['pictures']['values'], color=colors)
        plt.title('사진수 (Pictures)', fontweight='bold')
        plt.xticks(range(7), date_labels, rotation=45, ha='right')
        plt.ylabel('장수')
        
        for i, (value, source) in enumerate(zip(detailed_patterns['pictures']['values'], detailed_patterns['pictures']['sources'])):
            if max(detailed_patterns['pictures']['values']) > 0:
                plt.text(i, value + max(detailed_patterns['pictures']['values']) * 0.02,
                        source, ha='center', va='bottom', fontsize=8,
                        color='red' if source == '평균' else 'black')
        
        # 5. 의료 정보 (막대 그래프)
        plt.subplot(3, 3, 5)
        medical_data = [
            medical_features['consult_level'],
            medical_features['has_treatment'] * 4,  # 스케일 조정
            medical_features['has_past_diagnosis'] * 4  # 스케일 조정
        ]
        colors = ['#9370DB', '#32CD32', '#FF4500']
        labels = ['상담\n레벨', '치료\n여부', '과거\n진단']
        
        bars = plt.bar(range(3), medical_data, color=colors)
        plt.title('의료 정보', fontweight='bold')
        plt.xticks(range(3), labels)
        plt.ylabel('값')
        plt.ylim(0, 5)
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, medical_data)):
            original_values = [medical_features['consult_level'], 
                             medical_features['has_treatment'],
                             medical_features['has_past_diagnosis']]
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{original_values[i]}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 7일 특성 요약 (히트맵)
        plt.subplot(3, 3, 6)
        
        # 주요 특성들만 선택
        key_features = [
            ('avg_steps', '평균걸음수'),
            ('std_steps', '걸음수편차'),
            ('avg_lux', '평균럭스'),
            ('avg_decibel', '평균데시벨'),
            ('avg_pictures', '평균사진수')
        ]
        
        feature_values = []
        feature_labels = []
        
        for key, label in key_features:
            if key in seven_day_features:
                feature_values.append(seven_day_features[key])
                feature_labels.append(label)
        
        # 정규화
        if feature_values:
            max_val = max(feature_values) if max(feature_values) > 0 else 1
            normalized_values = [v / max_val for v in feature_values]
            
            heatmap_data = np.array(normalized_values).reshape(-1, 1)
            sns.heatmap(heatmap_data, annot=[[f'{v:.1f}'] for v in feature_values],
                       fmt='', cmap='YlOrRd', yticklabels=feature_labels, 
                       xticklabels=['값'], cbar=False)
            plt.title('요약 특성', fontweight='bold')
        
        # 7-9. 상세 정보 텍스트
        plt.subplot(3, 3, (7, 9))
        plt.axis('off')
        
        # 상세 정보 텍스트
        info_text = f"""
사용자 ID: {user_id}
라벨: {user_label}
최적 기간: {best_period[0]} ~ {best_period[-1]} ({len(best_period)}일)

=== 데이터 소스 통계 ===
• 걸음수 실제: {detailed_patterns['steps']['sources'].count('실제')}일
• 걸음수 평균: {detailed_patterns['steps']['sources'].count('평균')}일
• LUX 실제: {sum(1 for s in detailed_patterns['lux']['sources'] if '실제' in s)}일
• DECIBEL 실제: {sum(1 for s in detailed_patterns['decibel']['sources'] if '실제' in s)}일
• 사진 실제: {detailed_patterns['pictures']['sources'].count('실제')}일

=== 의료 정보 ===
• 상담 레벨: {medical_features['consult_level']} ({medical_features['consult_raw']})
• 치료 여부: {'있음' if medical_features['has_treatment'] else '없음'}
• 과거 진단: {'있음' if medical_features['has_past_diagnosis'] else '없음'}
• 진단 내용: {medical_features['diagnosis_raw']}

=== 생성된 특성 (총 34개) ===
• 일별 걸음수 (7개): {seven_day_features['day1_steps']:.0f} ~ {seven_day_features['day7_steps']:.0f}
• 일별 LUX (7개): {seven_day_features['day1_lux']:.1f} ~ {seven_day_features['day7_lux']:.1f}
• 일별 DECIBEL (7개): {seven_day_features['day1_decibel']:.1f} ~ {seven_day_features['day7_decibel']:.1f}
• 일별 사진수 (7개): {seven_day_features['day1_pictures']:.0f} ~ {seven_day_features['day7_pictures']:.0f}
• 요약 통계 (9개): 평균, 표준편차, 일관성 지수
• 상담 레벨 (1개): {medical_features['consult_level']}
• 치료 여부 (1개): {medical_features['has_treatment']}
• 과거 진단 (1개): {medical_features['has_past_diagnosis']}
        """
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))
        
        # 범례
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#2E8B57', label='실제 측정값'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', label='기간 평균 대체'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', label='데이터 없음')
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.suptitle(f'사용자 {user_id} - 완전한 7일 패턴 분석 (라벨: {user_label})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
        
        # 콘솔 출력
        print(f"\n📊 완전한 분석 결과:")
        print(f"최적 7일 구간: {best_period[0]} ~ {best_period[-1]}")
        print(f"라벨: {user_label}")
        print(f"총 특성 개수: 34개")
        
        print(f"\n📈 일별 상세 패턴:")
        for i in range(7):
            date = detailed_patterns['dates'][i]
            print(f"Day {i+1} ({date}):")
            print(f"  걸음수: {detailed_patterns['steps']['values'][i]:.0f} ({detailed_patterns['steps']['sources'][i]})")
            print(f"  LUX: {detailed_patterns['lux']['values'][i]:.1f} ({detailed_patterns['lux']['sources'][i]})")
            print(f"  DECIBEL: {detailed_patterns['decibel']['values'][i]:.1f} ({detailed_patterns['decibel']['sources'][i]})")
            print(f"  사진수: {detailed_patterns['pictures']['values'][i]:.0f} ({detailed_patterns['pictures']['sources'][i]})")


# 사용 함수
def debug_enhanced_pattern(user_id, data_file, label_file, picture_strategy='zero'):
    """완전한 파이프라인으로 사용자 패턴 디버깅"""
    
    # 완전한 파이프라인 import
    try:
        from Processing import EnhancedDataPreprocessor  # 파일명 맞게 수정
    except ImportError:
        print("❌ 완전한 파이프라인 파일을 찾을 수 없습니다.")
        return None
    
    # 전처리기 생성
    preprocessor = EnhancedDataPreprocessor(picture_strategy=picture_strategy)
    
    # 데이터 로드
    try:
        data_sheets, labels_df = preprocessor.load_data(data_file, label_file)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None
    
    # 시각화기 생성 및 분석
    visualizer = EnhancedPatternVisualizer(preprocessor)
    result = visualizer.analyze_user_pattern(user_id, data_sheets, labels_df)
    
    return result


# 실행 코드
if __name__ == "__main__":
    # 파일 경로
    data_file = r"C:\Users\parkm\OneDrive - dgu.ac.kr\AI_LAB\U-health\lifelog\Lifelog\data\231123-DATA.xlsx"
    label_file = r"C:\Users\parkm\OneDrive - dgu.ac.kr\AI_LAB\U-health\lifelog\Lifelog\data\231123-LABEL.xlsx"
    
    # 분석할 사용자 ID
    target_user = 23101802
    
    print(f"🔍 사용자 {target_user} 완전한 패턴 분석을 시작합니다...")
    
    try:
        result = debug_enhanced_pattern(target_user, data_file, label_file, picture_strategy='zero')
        
        if result:
            print("\n✅ 분석 완료!")
            print("이제 34개 특성이 어떻게 생성되었는지 확인할 수 있습니다.")
        else:
            print("\n❌ 분석 실패")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("파일 경로와 사용자 ID를 확인해주세요.")