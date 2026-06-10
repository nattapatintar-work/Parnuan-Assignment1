"""
tiered.py — Regex fast-path + LLM fallback

Goal: skip the LLM on easy cases (free) and route everything uncertain to extract_transactions().

fast_path() handles:
  1. empty / whitespace / null        → {"transactions": []} for free
  2. no number at all                 → {"transactions": []} for free
  3. single clean transaction         → extract with regex for free

Everything else → LLM

extract_tiered() has the same signature as extract_transactions().
meta["path"] tells you whether a row was "fast_path" (free) or "llm" (paid).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner import extract_transactions, validate_transactions, MAX_AMOUNT

# --- Regex patterns ---

# Amount: 1,234.50 / 1234 / 49.50
AMOUNT = re.compile(r"(?<![\d,])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)")

# Currency suffix ที่ตัดออกได้
CURRENCY_TAIL = re.compile(r"\s*(บาท|baht|บ\.?|฿|-\.)\s*$", re.IGNORECASE)

# Risk markers → ส่ง LLM เสมอ
RISK = re.compile(
    r"(ignore|instruction|forget|system|prompt|ลืม|คำสั่ง|select\s|insert\s|"
    r"\[INST\]|</s>|\{.*transactions.*\})",
    re.IGNORECASE
)

# เลขที่ไม่ใช่เงิน เช่น อายุ 25, เบอร์โทร, ฟอง
NOT_MONEY = re.compile(
    r"(ปี|อายุ|โมง|นาฬิกา|เบอร์|โทร|ฟอง|ชิ้น|อัน|คน|ตัว|ครั้ง|ซม|กม|กิโล|%|"
    r"เปอร์เซ็นต์|ขวบ|องศา|หวย|ล็อตเตอรี่|งบ|งวด|x\d|\d+x)"
)


def _parse_amount(s: str) -> float | int | None:
    try:
        val = float(s.replace(",", ""))
        return int(val) if val == int(val) else val
    except ValueError:
        return None


def fast_path(text: str) -> tuple[dict | None, str]:
    """
    ลอง answer โดยไม่ใช้ LLM
    Returns (result, reason)
    - result is None  → route to LLM
    - result is dict  → confident enough to return directly
    """

    # --- Guard: type check ---
    if not isinstance(text, str):
        return {"transactions": []}, "non_string"

    t = text.strip()

    # --- Case 1: empty / whitespace / null ---
    if not t or t.lower() == "null":
        return {"transactions": []}, "empty"

    # --- Case 2: risk markers → LLM ---
    if RISK.search(t):
        return None, "risk_marker"

    # --- Case 3: not-money cues → LLM ---
    if NOT_MONEY.search(t):
        return None, "not_money_cue"

    # --- Case 4: no number at all → empty for free ---
    amounts = AMOUNT.findall(t)
    if not amounts:
        return {"transactions": []}, "no_amount"

    # --- Case 5: มีมากกว่า 1 amount → multi-transaction → LLM ---
    if len(amounts) > 1:
        return None, "multi_amount"

    # --- Case 6: single amount → ลอง extract ---
    raw_amount = amounts[0]
    amount = _parse_amount(raw_amount)
    if amount is None:
        return None, "amount_parse_failed"

    # ตัด currency suffix ออก
    detail = t.replace(raw_amount, "", 1)
    detail = CURRENCY_TAIL.sub("", detail).strip()
    detail = re.sub(r"\s+", " ", detail).strip()

    # detail ต้องไม่ว่างและต้องมีตัวอักษรจริงๆ
    if not detail or not re.search(r"[A-Za-z฀-๿]", detail):
        return {"transactions": []}, "no_detail"

    # ยาวเกินไป → น่าจะเป็น context ยาว → LLM
    if len(detail) > 80:
        return None, "long_detail"

    # validate
    transactions = validate_transactions([{"amount": amount, "detail": detail}])
    if not transactions:
        return None, "validate_failed"

    return {"transactions": transactions}, "single_clean"


def extract_tiered(
    text: str,
    model: str = "google/gemini-2.5-flash",
) -> tuple[dict, dict]:
    """
    Drop-in tiered version ของ extract_transactions()
    Returns: (result, meta)
    meta["path"] = "fast_path" หรือ "llm"
    meta["reason"] = เหตุผลที่ fast_path ตัดสินใจ
    """
    fp, reason = fast_path(text)

    if fp is not None:
        return fp, {
            "path": "fast_path",
            "reason": reason,
            "model": "regex",
            "cost": 0.0,
        }

    result = extract_transactions(text, model=model)
    return result, {
        "path": "llm",
        "reason": reason,
        "model": model,
        "cost": result.get("cost", 0),
    }