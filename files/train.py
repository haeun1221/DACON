# -*- coding: utf-8 -*-
"""
train.py
--------
DACON 236694 - AI Agent 행동(Action) 의사결정 예측 챌린지
[Baseline/Train] 학습 스크립트

- 입력: data/train.jsonl, data/train_labels.csv
- 출력: model.pkl (학습된 파이프라인 - 전처리 + TF-IDF + LogisticRegression)

실행:
    python train.py
"""

import time
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import build_feature_df, load_jsonl, get_feature_columns

warnings.filterwarnings("ignore")

import os

DATA_DIR = r"C:\Users\user\Desktop\DACON\open\data"
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
TRAIN_LABELS = os.path.join(DATA_DIR, "train_labels.csv")
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

RANDOM_STATE = 42


def build_pipeline(cat_cols, num_cols, text_col):
    """텍스트(char+word TF-IDF 앙상블) + 범주형(OneHot) + 수치형(Scale) + LogisticRegression.

    실측 실험 결과 공유:
    - HistGradientBoostingClassifier(HGB)로 교체 + 텍스트를 TruncatedSVD로 압축하는
      방식을 시도했으나, 20k 서브셋 기준 Macro-F1이 0.564 -> 0.533/0.517로 오히려
      하락함 (SVD 압축 시 텍스트 정보 손실이 커서). 텍스트 없이 문맥 피처만 쓴 HGB는
      더 약함(0.417). LogReg + 앙상블도 강한 모델(LogReg)이 약한 모델에 끌려 내려가
      효과 없음(0.554). 이 문제는 current_prompt 텍스트 신호가 압도적으로 중요해서
      선형모델 + 고차원 TF-IDF 조합이 여전히 더 낫다고 판단, LogReg 유지.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text_char",
                TfidfVectorizer(
                    analyzer="char_wb",   # 한국어/영어/코드 혼재 텍스트에 강건한 char n-gram
                    ngram_range=(2, 3),
                    max_features=20000,
                    sublinear_tf=True,
                    min_df=3,
                ),
                text_col,
            ),
            (
                "text_word",
                TfidfVectorizer(
                    analyzer="word",   # "run", "test", "찾아" 같은 의미 단위 키워드 신호 보강
                    ngram_range=(1, 2),
                    max_features=20000,
                    sublinear_tf=True,
                    min_df=2,
                ),
                text_col,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            ),
            (
                "num",
                StandardScaler(),
                num_cols,
            ),
        ],
        sparse_threshold=1.0,
    )

    clf = LogisticRegression(
        max_iter=700,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",  # sklearn >=1.5부터 lbfgs가 다항(multinomial) 처리를 기본 수행
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
    return pipe


def main():
    t0 = time.time()
    print("[1/5] 데이터 로딩...")
    records = load_jsonl(TRAIN_JSONL)
    labels = pd.read_csv(TRAIN_LABELS)  # columns: id, action

    print(f"  - train.jsonl: {len(records)}건, train_labels.csv: {len(labels)}행")

    print("[2/5] 피처 엔지니어링...")
    df = build_feature_df(records)
    df = df.merge(labels, on="id", how="inner")
    print(f"  - 병합 후: {len(df)}행, 결측 라벨 제외됨: {len(records) - len(df)}건")

    cat_cols, num_cols, text_col = get_feature_columns()
    X = df[[text_col] + cat_cols + num_cols]
    y = df["action"].astype(str)

    print("[3/5] train/valid 분할 (stratified, 90/10)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
    )

    print("[4/5] 모델 학습 (TF-IDF(char+word 앙상블) + LogisticRegression)...")
    pipe = build_pipeline(cat_cols, num_cols, text_col)
    pipe.fit(X_tr, y_tr)

    val_pred = pipe.predict(X_val)
    macro_f1 = f1_score(y_val, val_pred, average="macro")
    print(f"  - Validation Macro-F1: {macro_f1:.4f}")
    n_iter = pipe.named_steps["clf"].n_iter_
    max_iter = pipe.named_steps["clf"].max_iter
    print(f"  - LogisticRegression n_iter_: {n_iter} (max_iter={max_iter}; "
          f"n_iter_ < max_iter 이면 수렴 완료)")
    print(classification_report(y_val, val_pred, digits=3))

    print("[5/5] 전체 데이터로 최종 재학습 후 모델 저장...")
    final_pipe = build_pipeline(cat_cols, num_cols, text_col)
    final_pipe.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_pipe, MODEL_PATH)

    print(f"모델 저장 완료: {MODEL_PATH}")
    final_n_iter = final_pipe.named_steps["clf"].n_iter_
    final_max_iter = final_pipe.named_steps["clf"].max_iter
    print(f"  - (최종 모델) LogisticRegression n_iter_: {final_n_iter} "
          f"(max_iter={final_max_iter})")
    print(f"총 소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
