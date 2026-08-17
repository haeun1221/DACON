# -*- coding: utf-8 -*-
"""
features.py
-----------
AI Agent Action Decision Challenge (DACON 236694) 공용 피처엔지니어링 모듈.

train.py(학습)와 script.py(추론)에서 동일하게 import하여 사용합니다.
동일한 로직을 두 곳에서 재사용해야 train/test 간 피처 불일치 문제를 막을 수 있습니다.
"""

import json
from collections import Counter

import pandas as pd

import re

# 14개 액션 클래스 (train_labels.csv / 문제 설명 기준)
ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

CAT_COLS = ["user_tier", "language_pref", "last_ci_status", "top_lang",
            "last_action", "last_action_2", "last_result_status"]

NUM_COLS = [
    "budget_tokens_remaining", "turn_index", "elapsed_session_sec",
    "loc", "num_open_files", "top_lang_ratio", "py_ratio",
    "n_history", "n_user_turns", "n_action_turns", "n_consecutive_last_action",
] + [f"n_prev_{a}" for a in ACTIONS] + [
    "mentions_specific_file", "mentions_open_file", "mentions_broad_scope",
    "kw_run", "kw_test", "kw_edit", "kw_write", "kw_search",
]

TEXT_COL = "text"

FILE_EXT_RE = re.compile(
    r"[\w\-/]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|rb|php|c|cpp|h|hpp|"
    r"yml|yaml|json|md|sql|sh|toml|css|scss|html|vue|txt|cfg|ini|env)\b",
    re.IGNORECASE,
)
BROAD_SCOPE_RE = re.compile(
    r"전부|모든|모두|여러|다\s|all\b|every\b|across\b|전체", re.IGNORECASE
)
KW_RUN_RE = re.compile(r"실행|돌려|돌리|run\b|execute", re.IGNORECASE)
KW_TEST_RE = re.compile(r"테스트|test\b|tests\b", re.IGNORECASE)
KW_EDIT_RE = re.compile(r"고쳐|수정|fix\b|patch\b|edit\b", re.IGNORECASE)
KW_WRITE_RE = re.compile(r"새로\s*만들|새\s*파일|write\b|create\b|작성", re.IGNORECASE)
KW_SEARCH_RE = re.compile(r"찾아|검색|search\b|find\b|grep\b", re.IGNORECASE)


def load_jsonl(path):
    """jsonl 파일을 읽어 dict 리스트로 반환"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _parse_session_meta(sm):
    # 주의: 실제 데이터에서 workspace는 session_meta 안에 중첩되어 있음
    # (session_meta.workspace), 최상위 workspace 키가 아님.
    sm = sm or {}
    return {
        "user_tier": sm.get("user_tier", "unknown"),
        "language_pref": sm.get("language_pref", "unknown"),
        "budget_tokens_remaining": sm.get("budget_tokens_remaining", 0) or 0,
        "turn_index": sm.get("turn_index", 0) or 0,
        "elapsed_session_sec": sm.get("elapsed_session_sec", 0) or 0,
    }


def _parse_workspace(ws):
    ws = ws or {}
    lang_mix = ws.get("language_mix") or {}
    if lang_mix:
        top_lang = max(lang_mix, key=lang_mix.get)
        top_lang_ratio = float(lang_mix.get(top_lang, 0.0))
    else:
        top_lang = "none"
        top_lang_ratio = 0.0
    open_files = ws.get("open_files") or []
    return {
        "loc": ws.get("loc", 0) or 0,
        "git_dirty": int(bool(ws.get("git_dirty", False))),
        "num_open_files": len(open_files),
        "last_ci_status": ws.get("last_ci_status", "none") or "none",
        "top_lang": top_lang,
        "top_lang_ratio": top_lang_ratio,
        "py_ratio": float(lang_mix.get("py", 0.0)),
        "_open_files": open_files,  # record_to_row에서 매칭용으로만 사용, 최종 피처엔 미포함
    }


def _parse_history(hist):
    hist = hist or []
    texts = []
    action_seq = []  # 시간순 action 이름 목록 (bigram/streak 계산용)
    last_result_summary = ""
    n_user_turns = 0
    n_action_turns = 0
    action_counts = Counter()

    for turn in hist:
        role = turn.get("role")
        if role == "user":
            content = turn.get("content", "") or ""
            texts.append(content)
            n_user_turns += 1
        elif role == "assistant_action":
            name = turn.get("name", "none") or "none"
            result_summary = turn.get("result_summary", "") or ""
            texts.append(f"[ACTION:{name}] {result_summary}")
            action_seq.append(name)
            last_result_summary = result_summary
            action_counts[name] += 1
            n_action_turns += 1

    last_action = action_seq[-1] if action_seq else "none"
    last_action_2 = action_seq[-2] if len(action_seq) >= 2 else "none"

    # 마지막 action이 몇 번 연속으로 반복됐는지 (예: read_file을 계속 반복 -> 다음도 이어질 가능성)
    streak = 0
    for a in reversed(action_seq):
        if a == last_action:
            streak += 1
        else:
            break

    rs_lower = last_result_summary.lower()
    if "fail" in rs_lower or "실패" in last_result_summary:
        last_result_status = "failed"
    elif "pass" in rs_lower or "ok" in rs_lower or "성공" in last_result_summary:
        last_result_status = "passed"
    else:
        last_result_status = "unknown"

    return {
        "hist_text": " ".join(texts),
        "last_action": last_action,
        "last_action_2": last_action_2,
        "n_consecutive_last_action": streak,
        "last_result_status": last_result_status,
        "n_history": len(hist),
        "n_user_turns": n_user_turns,
        "n_action_turns": n_action_turns,
    }, action_counts


def record_to_row(rec):
    """단일 jsonl 레코드(dict) -> 피처 dict로 변환"""
    session_meta = rec.get("session_meta") or {}
    row = {"id": rec.get("id")}
    row.update(_parse_session_meta(session_meta))
    # workspace는 최상위가 아니라 session_meta.workspace에 중첩되어 있음
    row.update(_parse_workspace(session_meta.get("workspace")))

    hist_feats, action_counts = _parse_history(rec.get("history"))
    row.update(hist_feats)

    for a in ACTIONS:
        row[f"n_prev_{a}"] = action_counts.get(a, 0)

    current_prompt = rec.get("current_prompt", "") or ""

    open_files = row.pop("_open_files", [])
    basenames = [f.rsplit("/", 1)[-1] for f in open_files if f]
    row["mentions_open_file"] = int(
        any(bn and bn in current_prompt for bn in basenames)
    )
    row["mentions_specific_file"] = int(bool(FILE_EXT_RE.search(current_prompt)))
    row["mentions_broad_scope"] = int(bool(BROAD_SCOPE_RE.search(current_prompt)))
    row["kw_run"] = int(bool(KW_RUN_RE.search(current_prompt)))
    row["kw_test"] = int(bool(KW_TEST_RE.search(current_prompt)))
    row["kw_edit"] = int(bool(KW_EDIT_RE.search(current_prompt)))
    row["kw_write"] = int(bool(KW_WRITE_RE.search(current_prompt)))
    row["kw_search"] = int(bool(KW_SEARCH_RE.search(current_prompt)))

    # current_prompt을 반복 삽입해 TF-IDF에서 가중치를 더 주도록 함
    row[TEXT_COL] = (current_prompt + " ") * 3 + " [HIST] " + row.pop("hist_text")
    row["current_prompt_len"] = len(current_prompt)

    return row


def build_feature_df(records):
    """레코드 리스트 -> DataFrame (id 포함, 모델 입력용 컬럼 전부 포함)"""
    rows = [record_to_row(r) for r in records]
    df = pd.DataFrame(rows)

    # 결측 방지 (혹시 모를 스키마 누락 대비)
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str)

    if "n_history" not in NUM_COLS:
        pass
    all_num_cols = NUM_COLS + ["git_dirty", "current_prompt_len"]
    for c in all_num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if TEXT_COL not in df.columns:
        df[TEXT_COL] = ""
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    return df


def get_feature_columns():
    return CAT_COLS, NUM_COLS + ["git_dirty", "current_prompt_len"], TEXT_COL
