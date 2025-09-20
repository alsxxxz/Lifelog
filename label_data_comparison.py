import pandas as pd

def label_data_comparison(label_file_path, data_file_path):
    """
    라벨, 데이터 파일을 비교 > 일치/불일치 거르기
    """

    # 라벨
    label_df = pd.read_excel(label_file_path, sheet_name='MZ')
    # 데이터 - USER_ACCOUNT 시트 
    data_df = pd.read_excel(data_file_path, sheet_name='USER_ACCOUNT')

# 라벨 파일 내 중복 확인
    label_id_counts = label_df['id'].value_counts()
    label_duplicates = label_id_counts[label_id_counts > 1]
    
    if len(label_duplicates) > 0:
        print(f"라벨 파일에 중복된 ID가 {len(label_duplicates)}개 있습니다:")
        for duplicate_id, count in label_duplicates.items():
            print(f"   ID: {duplicate_id} -> {count}번 중복")
    else:
        print("라벨 파일에 중복 ID 없음")
    
    # 데이터 파일 내 중복 확인
    data_id_counts = data_df['USER_ID'].value_counts()
    data_duplicates = data_id_counts[data_id_counts > 1]
    
    if len(data_duplicates) > 0:
        print(f"데이터 파일에 중복된 USER_ID가 {len(data_duplicates)}개 있습니다:")
        for duplicate_id, count in data_duplicates.items():
            print(f"   USER_ID: {duplicate_id} -> {count}번 중복")
    else:
        print("데이터 파일에 중복 USER_ID 없음")
    
    
# LABEL ID list 만들기
    # 라벨에 있는 ID 문자열로 불러와야함 notna
    # 라벨 있/없 사용자 구분
    label_ids= set(str(id_val).strip() for id_val in label_df['id'] if pd.notna(id_val))
    print(f"라벨에 있는 고유 ID수: {len(label_ids)}")
    data_df['IN_LABEL'] = data_df['USER_ID'].apply(lambda x: str(x).strip() in label_ids if pd.notna(x) else False)
    #o
    matched_users = data_df[data_df['IN_LABEL'] == True]
    #x
    not_matched_users = data_df[data_df['IN_LABEL'] == False] 
    




    # 결과 출력
    print("\n" + "="*50)
    print("분석 결과")
    print("="*50)
    print(f"라벨과 일치하는 사용자 수: {len(matched_users)}명")
    print(f"라벨과 일치하지 않는 사용자 수: {len(not_matched_users)}명")
    print(f"전체 데이터 사용자 수: {len(data_df)}명")
    
    print(f"\n일치율: {len(matched_users)/len(data_df)*100:.1f}%")
    
    # 라벨에 없는 사용자들의 상세 정보 출력
    print("\n" + "="*50)
    print("🚫 라벨에 없는 사용자 목록")
    print("="*50)
    
    if len(not_matched_users) > 0:
        for idx, (_, user) in enumerate(not_matched_users.iterrows(), 1):
            print(f"{idx:2d}. ID: {user['USER_ID']:<15} 이름: {user['USER_NAME']}")
    else:
        print("라벨에 없는 사용자가 없습니다.")
    
    # 라벨에 있는 사용자들 중 일부 출력 (처음 10명)
    print(f"\n" + "="*50)
    print("라벨과 일치하는 사용자 (처음 10명)")
    print("="*50)
    
    if len(matched_users) > 0:
        for idx, (_, user) in enumerate(matched_users.head(10).iterrows(), 1):
            print(f"{idx:2d}. ID: {user['USER_ID']:<15} 이름: {user['USER_NAME']}")
        
        if len(matched_users) > 10:
            print(f"... 외 {len(matched_users)-10}명 더")
    
    # 라벨에 없는 사용자 ID만 리스트로 반환
    not_matched_ids = not_matched_users['USER_ID'].tolist()
    
    return {
        'matched_count': len(matched_users),
        'not_matched_count': len(not_matched_users),
        'not_matched_users': not_matched_users[['USER_ID', 'USER_NAME']].to_dict('records'),
        'not_matched_ids': not_matched_ids
    }

# 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    data_file_path = r"C:\Users\parkm\OneDrive - dgu.ac.kr\AI_LAB\U-health\lifelog\data\231123-DATA.xlsx"
    label_file_path = r"C:\Users\parkm\OneDrive - dgu.ac.kr\AI_LAB\U-health\lifelog\data\231123-LABEL.xlsx"
    
    # 분석 실행
    result = label_data_comparison(label_file_path, data_file_path)
    
    # 라벨에 없는 사용자 ID만 별도로 출력
    print(f"\n" + "="*50)
    print("라벨에 없는 USER_ID 목록")
    print("="*50)
    for user_id in result['not_matched_ids']:
        print(user_id)