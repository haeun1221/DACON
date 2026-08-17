# -*- coding: utf-8 -*-
"""
script.py
---------
DACON 236694 - AI Agent 행동(Action) 의사결정 예측 챌린지
[Baseline/Inference] 추론 스크립트 (코드 제출 대회 규격: script.py)

- 입력: data/test.jsonl, model.pkl (train.py로 사전 학습된 모델)
- 출력: submission.csv (id, action)

실행:
    python script.py

제약 조건 (대회 규칙):
    - 추론 코드 실행 시간 <= 10분
    - 오프라인 환경(인터넷 연결 불가) 실행
    - T4 GPU(16GB), 3 vCPU, 12GB RAM
  -> TF-IDF + LogisticRegression 조합은 CPU만으로 충분히 빠르게 동작하므로
     GPU 없이도 시간 제약을 여유있게 만족합니다.
"""

import time

import joblib
import pandas as pd

from features import build_feature_df, load_jsonl, get_feature_columns, ACTIONS

import os

DATA_DIR = "./data"  # 채점 서버가 zip 압축 해제 위치에 자동으로 만들어주는 상대경로
TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, "sample_submission.csv")
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
OUT_DIR = "./output"
OUTPUT_PATH = os.path.join(OUT_DIR, "submission.csv")

DEFAULT_ACTION = "respond_only"  # 예외 발생 시 사용할 안전한 기본값


def main():
    t0 = time.time()
    print("[1/4] 모델 로딩...")
    pipe = joblib.load(MODEL_PATH)

    print("[2/4] 평가 데이터 로딩 및 피처 엔지니어링...")
    records = load_jsonl(TEST_JSONL)
    df = build_feature_df(records)

    cat_cols, num_cols, text_col = get_feature_columns()
    X = df[[text_col] + cat_cols + num_cols]

    print("[3/4] 추론...")
    preds = pipe.predict(X)

    # 혹시 모를 미지 클래스 방어 (14개 클래스 외 값이 나오지 않도록 보정)
    preds = pd.Series(preds).apply(lambda a: a if a in ACTIONS else DEFAULT_ACTION)

    submission = pd.DataFrame({"id": df["id"], "action": preds})

    print("[4/4] 제출 양식(sample_submission.csv)과 정렬/검증 후 저장...")
    try:
        sample = pd.read_csv(SAMPLE_SUBMISSION)
        submission = sample[["id"]].merge(submission, on="id", how="left")
        submission["action"] = submission["action"].fillna(DEFAULT_ACTION)
    except FileNotFoundError:
        pass  # sample_submission.csv 없이도 동작 가능하도록 방어

    os.makedirs(OUT_DIR, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"저장 완료: {OUTPUT_PATH} ({len(submission)}행)")
    print(f"총 소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
