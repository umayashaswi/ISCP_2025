#!/usr/bin/env python3
"""
detector_full_candidate_name.py
Usage:
    python3 detector_full_candidate_name.py iscp_pii_dataset_-_Sheet1.csv

Output:
    redacted_output_candidate_full_name.csv
"""

import sys
import json
import re
from pathlib import Path

try:
    import pandas as pd
except Exception as e:
    print("Missing dependency: pandas. Install with: pip install pandas")
    raise

# -------------------------
# Regexes and mask helpers
# -------------------------
PHONE_RE = re.compile(r'(?<!\d)(\d{10})(?!\d)')
AADHAR_RE = re.compile(r'(?<!\d)(\d{12})(?!\d)')  # strict 12 digits (no spaces) - we'll also allow spaced forms below
AADHAR_SPACED_RE = re.compile(r'(?<!\d)(\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)')
PASSPORT_RE = re.compile(r'\b([A-PR-WY][0-9]{7})\b', re.IGNORECASE)  # common passport pattern
EMAIL_RE = re.compile(r'([a-zA-Z0-9._%+\-]+)@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})')
IP_RE = re.compile(r'\b((?:\d{1,3}\.){3}\d{1,3})\b')
UPI_RE = re.compile(r'\b([\w.\-]{2,}@[\w]{2,})\b', re.IGNORECASE)  # conservative
LONG_DIGITS_RE = re.compile(r'\d{10,}')

# Masking functions
def mask_phone(s: str) -> str:
    def repl(m):
        v = m.group(1)
        return f"{v[:2]}{'X'*6}{v[-2:]}"
    return PHONE_RE.sub(repl, s)

def mask_aadhar(s: str) -> str:
    # normalize digits then mask last 4
    m = AADHAR_SPACED_RE.search(s)
    if not m:
        return s
    digits = re.sub(r'\D', '', m.group(1))
    if len(digits) != 12:
        return s
    masked = "XXXX XXXX " + digits[-4:]
    return AADHAR_SPACED_RE.sub(masked, s)

def mask_passport(s: str) -> str:
    def repl(m):
        v = m.group(1)
        return v[0].upper() + 'X'*6 + v[-1]
    return PASSPORT_RE.sub(repl, s)

def mask_email(s: str) -> str:
    def repl(m):
        user, domain = m.group(1), m.group(2)
        if len(user) <= 2:
            masked_user = user[0] + 'X'*(len(user)-1)
        else:
            masked_user = user[:2] + 'X'*(len(user)-2)
        return masked_user + "@" + domain
    return EMAIL_RE.sub(repl, s)

def mask_upi(s: str) -> str:
    # only treat as UPI if the domain part is not a normal TLD-containing domain (we'll be conservative)
    def repl(m):
        full = m.group(1)
        if '@' in full:
            user, handle = full.split('@',1)
            # consider it UPI when handle has no dot (like ybl) OR looks numeric
            if '.' not in handle:
                if len(user) <= 2:
                    masked_user = user[0] + 'X'*(len(user)-1)
                else:
                    masked_user = user[:2] + 'X'*(len(user)-3 if len(user)>3 else 1) + user[-1]
                return masked_user + '@' + handle
        return "[REDACTED_UPI]"
    return UPI_RE.sub(repl, s)

def mask_ip(s: str) -> str:
    def repl(m):
        ip = m.group(1)
        parts = ip.split('.')
        parts[-1] = 'X'
        return '.'.join(parts)
    return IP_RE.sub(repl, s)

def mask_full_name(name: str) -> str:
    # Mask each name part keeping first letter and replacing rest with X
    parts = [p for p in name.strip().split() if p]
    masked = []
    for p in parts:
        if len(p) == 1:
            masked.append(p + 'X')
        else:
            masked.append(p[0] + 'X'*(len(p)-1))
    return " ".join(masked)

# -------------------------
# Detection & redaction
# -------------------------
def looks_like_full_name(s: str) -> bool:
    # Consider full name if there are two or more words and each is alphabetic-ish
    parts = [p for p in s.strip().split() if p]
    if len(parts) < 2:
        return False
    alpha_count = sum(1 for p in parts if re.search(r'[A-Za-z]', p))
    return alpha_count >= 2

def classify_and_redact(record: dict):
    """
    record: dict of fields -> values (may be strings or other)
    returns: (redacted_record_dict, is_pii_bool)
    """
    # track combinatorial flags
    combo = {
        "name": False,
        "email": False,
        "address": False,
        "device_or_ip": False
    }
    standalone_hit = False

    redacted = {}

    # Helper to examine a field value and possibly redact standalone PII
    def process_field(k, v):
        nonlocal standalone_hit
        # leave non-str types as-is
        if not isinstance(v, str):
            return v, False

        orig = v
        changed = False

        # Standalone: phone
        if PHONE_RE.search(orig):
            v = mask_phone(v)
            changed = True
            standalone_hit = True

        # Standalone: Aadhaar
        if AADHAR_SPACED_RE.search(orig) or AADHAR_RE.search(orig):
            # confirm 12 digits inside
            digits = re.sub(r'\D', '', orig)
            if len(digits) == 12:
                v = mask_aadhar(v)
                changed = True
                standalone_hit = True

        # Standalone: Passport
        if PASSPORT_RE.search(orig):
            v = mask_passport(v)
            changed = True
            standalone_hit = True

        # Standalone: UPI (conservative)
        # ensure it's not a normal email-like domain (we'll only treat handle without dot as UPI)
        upi_hits = re.findall(UPI_RE, orig)
        if upi_hits:
            chosen = False
            for u in upi_hits:
                if '@' in u:
                    _, handle = u.split('@',1)
                    if '.' not in handle:
                        chosen = True
            if chosen:
                v = mask_upi(v)
                changed = True
                standalone_hit = True

        # Track email (for combinatorials)
        if EMAIL_RE.search(orig):
            combo["email"] = True

        # Track name (full name) — prefer explicit 'name' field, but also sensible heuristics
        if k.lower() == "name" and looks_like_full_name(orig):
            combo["name"] = True
        # Also consider first_name+last_name combination detected by keys later

        # Track address field presence
        if k.lower() == "address" and len(orig.strip()) >= 8:
            # detect presence of numbers/street/pincode heuristic (pincode 6 digits anywhere)
            if re.search(r'\d{3,}', orig) or re.search(r'\b(pin|pincode|postcode)\b', orig, re.I) or re.search(r'\d{6}\b', orig):
                combo["address"] = True
            else:
                # presence of comma-separated tokens and words can be address; mark conservatively
                if len(orig.split()) >= 3:
                    combo["address"] = True

        # Track device/ip
        if k.lower() in ("ip_address", "device_id") or IP_RE.search(orig):
            combo["device_or_ip"] = True

        # Return possibly masked value
        return v, changed

    # First pass: process all fields and collect combinatorial signals
    for k, v in record.items():
        new_v, field_changed = process_field(k, v)
        redacted[k] = new_v

    # Additional combinatorial detection: if separate first_name and last_name exist
    if ("first_name" in record and "last_name" in record and
        isinstance(record.get("first_name"), str) and isinstance(record.get("last_name"), str) and
        record.get("first_name").strip() and record.get("last_name").strip()):
        combo["name"] = True

    # If name-like present in other fields, attempt to detect
    for k in record:
        if k.lower() in ("name", "first_name", "last_name") and isinstance(record[k], str):
            if looks_like_full_name(record[k]):
                combo["name"] = True

    # Evaluate combinatorial PII condition (>=2)
    combo_count = sum(1 for v in combo.values() if v)
    combo_pii = combo_count >= 2

    # If combo_pii, redact the combinatorial fields (if not already)
    if combo_pii:
        # name
        if combo["name"]:
            if "name" in redacted and isinstance(redacted["name"], str) and looks_like_full_name(redacted["name"]):
                redacted["name"] = mask_full_name(redacted["name"])
            else:
                # try combine first+last
                fn = record.get("first_name")
                ln = record.get("last_name")
                if isinstance(fn, str) and isinstance(ln, str) and fn.strip() and ln.strip():
                    redacted["first_name"] = mask_full_name(fn)
                    redacted["last_name"] = mask_full_name(ln)
        # email
        for k, v in list(redacted.items()):
            if isinstance(v, str) and EMAIL_RE.search(v):
                redacted[k] = mask_email(v)
        # address
        if "address" in redacted and isinstance(redacted["address"], str) and redacted["address"].strip():
            redacted["address"] = "[REDACTED_ADDRESS]"
        # device_id
        if "device_id" in redacted and isinstance(redacted["device_id"], str) and redacted["device_id"].strip():
            redacted["device_id"] = "[REDACTED_DEVICE]"
        # mask IPs found anywhere
        for k, v in list(redacted.items()):
            if isinstance(v, str) and IP_RE.search(v):
                redacted[k] = mask_ip(v)

    is_pii = bool(standalone_hit or combo_pii)
    return redacted, is_pii

# -------------------------
# Main script
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detector_full_candidate_name.py <input_csv>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    # Read CSV and normalize column names to lowercase for robustness
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in df.columns}
    # expected record id column
    if 'record_id' not in cols_lower:
        print("Error: expected 'record_id' column in CSV.")
        sys.exit(1)
    # determine JSON column (support Data_json or data_json)
    json_col = None
    for candidate in ('data_json', 'datajson', 'Data_json', 'DataJson'):
        if candidate.lower() in cols_lower:
            json_col = cols_lower[candidate.lower()]
            break
    if json_col is None:
        # fallback: if only two columns, assume second is JSON
        if len(df.columns) == 2:
            json_col = df.columns[1]
            print(f"Assuming JSON column is: {json_col}")
        else:
            print("Error: could not find the JSON column (expected 'Data_json' or 'data_json').")
            print("Found columns:", df.columns.tolist())
            sys.exit(1)

    out_rows = []
    for _, row in df.iterrows():
        record_id = row[cols_lower['record_id']]
        raw_json_text = row.get(json_col, "{}")
        # parse JSON safely
        data = {}
        if isinstance(raw_json_text, str):
            try:
                data = json.loads(raw_json_text)
                if not isinstance(data, dict):
                    # if not object, try again by wrapping or skip
                    data = {}
            except Exception:
                # try replacing single quotes with double quotes then parse
                try:
                    data = json.loads(raw_json_text.replace("'", '"'))
                except Exception:
                    data = {}
        elif isinstance(raw_json_text, dict):
            data = raw_json_text
        else:
            data = {}

        redacted, is_pii = classify_and_redact(data)
        # redacted_data_json must be a JSON string in the CSV cell
        redacted_json_str = json.dumps(redacted, ensure_ascii=False)
        out_rows.append({
            "record_id": record_id,
            "redacted_data_json": redacted_json_str,
            "is_pii": bool(is_pii)
        })

    out_df = pd.DataFrame(out_rows, columns=["record_id", "redacted_data_json", "is_pii"])
    out_file = Path("redacted_output_candidate_full_name.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Saved -> {out_file.resolve()}")

if __name__ == "__main__":
    # local import to avoid top-level dependency if not used
    import pandas as pd
    main()
