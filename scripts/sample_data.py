"""
RAG 구축용 데이터 샘플링 스크립트
- 법령: 898개 (전체)
- 해석: 58개 (전체)
- 판례: 750개 (랜덤)
- 결정문: 294개 (랜덤)
총 2,000개
"""

import os
import shutil
import random
from pathlib import Path

# 경로 설정
BASE_DIR = Path("/Users/seoyeonmun/Justi-Q/data")
OUTPUT_DIR = Path("/Users/seoyeonmun/Justi-Q/data_sampled")

# 소스 경로
SOURCES = {
    "judgement": [
        BASE_DIR / "Training/1_source_data/TS_judgement",
        BASE_DIR / "Validation/1_source_data/VS_judgement",
    ],
    "decision": [
        BASE_DIR / "Training/1_source_data/TS_decision",
        BASE_DIR / "Validation/1_source_data/VS_dicision",  # 오타 그대로 유지
    ],
    "statute": [
        BASE_DIR / "Training/1_source_data/TS_statute",
        BASE_DIR / "Validation/1_source_data/VS_statute",
    ],
    "interpretation": [
        BASE_DIR / "Training/1_source_data/TS_interpretation",
        BASE_DIR / "Validation/1_source_data/VS_interpretation",
    ],
}

# 샘플 수 설정
SAMPLE_COUNTS = {
    "judgement": 750,
    "decision": 294,
    "statute": None,  # 전체
    "interpretation": None,  # 전체
}

def get_all_files(paths):
    """여러 경로에서 모든 파일 수집"""
    files = []
    for path in paths:
        if path.exists():
            files.extend([f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')])
    return files

def sample_files(files, count):
    """파일 리스트에서 랜덤 샘플링"""
    if count is None or count >= len(files):
        return files
    return random.sample(files, count)

def main():
    random.seed(42)  # 재현성을 위한 시드 고정

    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_copied = 0

    for data_type, source_paths in SOURCES.items():
        # 출력 폴더 생성
        output_path = OUTPUT_DIR / data_type
        output_path.mkdir(exist_ok=True)

        # 파일 수집
        all_files = get_all_files(source_paths)
        print(f"\n[{data_type}] 전체 파일: {len(all_files)}개")

        # 샘플링
        sample_count = SAMPLE_COUNTS[data_type]
        sampled_files = sample_files(all_files, sample_count)
        print(f"[{data_type}] 샘플링: {len(sampled_files)}개")

        # 파일 복사
        for f in sampled_files:
            shutil.copy2(f, output_path / f.name)

        total_copied += len(sampled_files)

    print(f"\n{'='*40}")
    print(f"총 샘플링 완료: {total_copied}개")
    print(f"저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
