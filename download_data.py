import os
from kaggle.api.kaggle_api_extended import KaggleApi

# API 인증
api = KaggleApi()
api.authenticate()

# 데이터 다운로드
print("Downloading competition data...")
competition = 'arc-prize-2025'

try:
    # 모든 파일 다운로드
    api.competition_download_files(competition, path='data/')
    print("✓ Download completed successfully!")

    # 파일 압축 해제
    import zipfile
    zip_file = 'data/arc-prize-2025.zip'
    if os.path.exists(zip_file):
        print("Extracting files...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data/')
        os.remove(zip_file)
        print("✓ Extraction completed!")

    # 다운로드된 파일 목록
    print("\nDownloaded files:")
    for file in os.listdir('data/'):
        file_path = os.path.join('data/', file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to download individual files...")

    # 개별 파일 다운로드
    files = [
        'arc-agi_training_challenges.json',
        'arc-agi_training_solutions.json',
        'arc-agi_evaluation_challenges.json',
        'arc-agi_evaluation_solutions.json',
        'arc-agi_test_challenges.json',
        'sample_submission.json'
    ]

    for file in files:
        try:
            api.competition_download_file(competition, file, path='data/')
            print(f"✓ Downloaded: {file}")
        except Exception as e:
            print(f"✗ Failed to download {file}: {e}")
