#!/usr/bin/env python3
"""Reproducible Session 4 cleaner for AI4Bharat parallel and monolingual corpora.

Default mode is bilingual-pair aware (Samanantar): normalize Indic text without
deleting ZWJ/ZWNJ, validate source/target structure, apply quality rules, check
scripts, mask structured PII, deduplicate normalized pairs, and decontaminate
against held-out fingerprints.

With --monolingual (auto-enabled for ai4bharat/IndicCorpV2), clean single-text
records from --text-field instead of src/tgt pairs.

Use Samanantar v0.3 LaBSE metadata via --semantic-score-field for semantic
alignment filtering in bilingual mode only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Iterable

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IPV4_RE = re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
URL_RE = re.compile(r"https?://[^\s<>'\"]+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{11,}")
HTML_TAG_RE = re.compile(r"<[^>]{1,120}>")
PLACEHOLDER_RE = re.compile(r"(?:%\d*\$?[a-zA-Z]|\{\w*\}|&[A-Za-z][A-Za-z0-9]+;)")
REMOVE_CODEPOINTS = {
    0x200B, 0xFEFF,  # zero-width space, BOM
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embedding/override
    0x2066, 0x2067, 0x2068, 0x2069,  # bidi isolates
}
PRESERVE_FORMAT = {0x200C, 0x200D}  # ZWNJ / ZWJ are meaningful in Indic scripts

SCRIPT_RANGES = {
    "Latin": [(0x0041,0x005A),(0x0061,0x007A),(0x00C0,0x024F)],
    "Devanagari": [(0x0900,0x097F)],
    "Bengali": [(0x0980,0x09FF)],
    "Gurmukhi": [(0x0A00,0x0A7F)],
    "Gujarati": [(0x0A80,0x0AFF)],
    "Odia": [(0x0B00,0x0B7F)],
    "Tamil": [(0x0B80,0x0BFF)],
    "Telugu": [(0x0C00,0x0C7F)],
    "Kannada": [(0x0C80,0x0CFF)],
    "Malayalam": [(0x0D00,0x0D7F)],
}
EXPECTED_SCRIPT = {
    "as":"Bengali", "bn":"Bengali", "gu":"Gujarati", "hi":"Devanagari",
    "kn":"Kannada", "ml":"Malayalam", "mr":"Devanagari", "or":"Odia",
    "pa":"Gurmukhi", "ta":"Tamil", "te":"Telugu",
}
INDICCORP_V2_CONFIG = "indiccorp_v2"
INDICCORP_V2_LANGUAGE_SPLITS = {
    "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "or": "ory_Orya",
    "pa": "pan_Guru", "ta": "tam_Taml", "te": "tel_Telu",
    "brx": "brx_Deva", "doi": "doi_Deva", "gom": "gom_Deva", "ks": "kas_Arab",
    "mai": "mai_Deva", "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Deva",
    "ur": "urd_Arab", "kha": "khasi", "sat": "santhali",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: str) -> tuple[str, Counter[str]]:
    stats: Counter[str] = Counter()
    value = html.unescape(value or "")
    before = value
    value = unicodedata.normalize("NFC", value)
    if value != before: stats["unicode_nfc_changed"] += 1
    out: list[str] = []
    for ch in value:
        cp = ord(ch)
        if cp in PRESERVE_FORMAT:
            out.append(ch); stats["indic_joiner_preserved"] += 1; continue
        if cp in REMOVE_CODEPOINTS:
            stats[f"removed_U+{cp:04X}"] += 1; continue
        if ch == "\ufffd":
            stats["replacement_character_removed"] += 1; continue
        if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in "\n\t":
            stats["other_control_removed"] += 1; continue
        out.append(ch)
    value = "".join(out).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value, stats


def script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ch in text:
        if not ch.isalpha(): continue
        cp = ord(ch); hit = False
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1; hit = True; break
        if not hit: counts["Other"] += 1
    return counts


def script_ratio(text: str, script: str) -> float:
    counts = script_counts(text); total = sum(counts.values())
    return (counts.get(script, 0) / total) if total else 0.0


def mask_pii(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    def apply(pattern: re.Pattern[str], token: str, key: str, s: str) -> str:
        def repl(_: re.Match[str]) -> str:
            counts[key] += 1; return token
        return pattern.sub(repl, s)
    text = apply(EMAIL_RE, "[EMAIL]", "email", text)
    text = apply(IPV4_RE, "[IP_ADDRESS]", "ipv4", text)
    text = apply(PHONE_RE, "[PHONE]", "phone", text)
    text = apply(URL_RE, "[URL]", "url", text)
    return text, counts


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def pair_hash(language: str, src: str, tgt: str) -> str:
    payload = f"{language}\0{normalized_key(src)}\0{normalized_key(tgt)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_hash(language: str, text: str) -> str:
    payload = f"{language}\0{normalized_key(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ngram_hashes(text: str, n: int = 13) -> set[str]:
    words = normalized_key(text).split()
    if len(words) < n:
        return {hashlib.sha256(" ".join(words).encode()).hexdigest()} if words else set()
    return {hashlib.sha256(" ".join(words[i:i+n]).encode()).hexdigest() for i in range(len(words)-n+1)}


def load_heldout(path: Path | None) -> set[str]:
    if not path: return set()
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): hashes.update(ngram_hashes(line))
    return hashes


def iter_local(path: Path) -> Iterator[dict[str, Any]]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for p in paths:
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if line.strip():
                        row = json.loads(line); row.setdefault("_input_file", p.name); row.setdefault("_input_line", line_no); yield row
        elif ext == ".csv":
            with p.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f): row["_input_file"] = p.name; yield row
        elif ext in {".parquet", ".pq"}:
            try: import pandas as pd  # type: ignore
            except ImportError as exc: raise RuntimeError("Parquet requires pandas and pyarrow") from exc
            for row in pd.read_parquet(p).to_dict(orient="records"):
                row["_input_file"] = p.name; yield row


def is_indiccorp_v2(dataset_id: str | None) -> bool:
    return bool(dataset_id and "IndicCorpV2" in dataset_id)


def iter_hf(
    dataset_id: str,
    languages: Iterable[str],
    *,
    monolingual: bool,
) -> Iterator[dict[str, Any]]:
    try: from datasets import load_dataset  # type: ignore
    except ImportError as exc: raise RuntimeError("Hugging Face streaming requires: pip install datasets") from exc
    if monolingual and is_indiccorp_v2(dataset_id):
        for lang in languages:
            split = INDICCORP_V2_LANGUAGE_SPLITS.get(lang)
            if not split:
                known = ", ".join(sorted(INDICCORP_V2_LANGUAGE_SPLITS))
                raise ValueError(f"No IndicCorpV2 split for {lang!r}; known codes: {known}")
            ds = load_dataset(dataset_id, INDICCORP_V2_CONFIG, split=split, streaming=True)
            for row in ds:
                row["_language"] = lang
                yield row
        return
    if monolingual:
        raise ValueError("HF monolingual streaming is supported for ai4bharat/IndicCorpV2; use --input for other sources")
    for lang in languages:
        ds = load_dataset(dataset_id, lang, split="train", streaming=True)
        for row in ds:
            row["_language"] = lang
            yield row


def choose_language(row: dict[str, Any], fixed: str | None) -> str:
    lang = str(row.get("_language") or row.get("language") or row.get("lang") or fixed or "").lower()
    return lang if lang in EXPECTED_SCRIPT else "unknown"


def evaluate_pair(row: dict[str, Any], language: str, args: argparse.Namespace, heldout: set[str]) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
    reasons: list[str] = []; flags: list[str] = []
    norm_stats: Counter[str] = Counter(); pii_stats: Counter[str] = Counter()
    raw_src, raw_tgt = row.get(args.src_field), row.get(args.tgt_field)
    if not isinstance(raw_src, str): reasons.append("missing_or_nonstring_src"); raw_src = ""
    if not isinstance(raw_tgt, str): reasons.append("missing_or_nonstring_tgt"); raw_tgt = ""
    src, ns = normalize_text(raw_src); tgt, nt = normalize_text(raw_tgt)
    norm_stats.update({f"src_{k}":v for k,v in ns.items()}); norm_stats.update({f"tgt_{k}":v for k,v in nt.items()})
    if not src: reasons.append("empty_src")
    if not tgt: reasons.append("empty_tgt")
    if language == "unknown": reasons.append("missing_or_unknown_language")
    if len(src) < args.min_chars or len(tgt) < args.min_chars: flags.append("very_short_pair")
    if len(src) > args.max_chars or len(tgt) > args.max_chars: reasons.append("extreme_sentence_length")
    if src and tgt:
        ratio = len(tgt) / max(1, len(src))
        if ratio < args.min_length_ratio or ratio > args.max_length_ratio: reasons.append("extreme_length_ratio")
        if normalized_key(src) == normalized_key(tgt) and len(src) > 12: flags.append("identical_source_target")
    if REPEATED_CHAR_RE.search(src + tgt): reasons.append("repeated_character_run")
    if HTML_TAG_RE.search(src + tgt): flags.append("html_markup_present")
    src_placeholders = sorted(PLACEHOLDER_RE.findall(src)); tgt_placeholders = sorted(PLACEHOLDER_RE.findall(tgt))
    if src_placeholders != tgt_placeholders: flags.append("placeholder_mismatch")
    src_latin = script_ratio(src, "Latin")
    expected = EXPECTED_SCRIPT.get(language)
    tgt_expected = script_ratio(tgt, expected) if expected else 0.0
    if len([c for c in src if c.isalpha()]) >= 8 and src_latin < args.source_script_threshold:
        (reasons if args.strict_language else flags).append("source_not_predominantly_latin")
    if len([c for c in tgt if c.isalpha()]) >= 8 and tgt_expected < args.target_script_threshold:
        (reasons if args.strict_language else flags).append("target_script_mismatch")
    score_value = row.get(args.semantic_score_field) if args.semantic_score_field else None
    if score_value is not None:
        try:
            score = float(score_value)
            if score < args.semantic_threshold: reasons.append("low_semantic_alignment_score")
        except (TypeError, ValueError): flags.append("invalid_semantic_score")
    else:
        flags.append("semantic_alignment_not_scored")
    src_masked, ps = mask_pii(src); tgt_masked, pt = mask_pii(tgt)
    pii_stats.update({f"src_{k}":v for k,v in ps.items()}); pii_stats.update({f"tgt_{k}":v for k,v in pt.items()})
    if ps or pt: flags.append("structured_pii_masked")
    if bool(ps) != bool(pt): flags.append("pii_alignment_asymmetry")
    if heldout and (ngram_hashes(src) & heldout or ngram_hashes(tgt) & heldout): reasons.append("heldout_ngram_overlap")
    digest = pair_hash(language, src_masked, tgt_masked)
    return {
        "keep": not reasons,
        "reasons": sorted(set(reasons)), "flags": sorted(set(flags)),
        "language": language, "src": src_masked, "tgt": tgt_masked,
        "pair_sha256": digest,
        "script_ratios": {"src_latin": round(src_latin,4), "tgt_expected": round(tgt_expected,4)},
        "semantic_score": score_value,
    }, norm_stats, pii_stats


def evaluate_monolingual(
    row: dict[str, Any],
    language: str,
    args: argparse.Namespace,
    heldout: set[str],
) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
    reasons: list[str] = []
    flags: list[str] = []
    norm_stats: Counter[str] = Counter()
    pii_stats: Counter[str] = Counter()
    raw = row.get(args.text_field)
    if not isinstance(raw, str):
        reasons.append("missing_or_nonstring_text")
        raw = ""
    text, ns = normalize_text(raw)
    norm_stats.update({f"text_{k}": v for k, v in ns.items()})
    if not text:
        reasons.append("empty_text")
    if language == "unknown":
        reasons.append("missing_or_unknown_language")
    if text and len(text) < args.min_chars:
        flags.append("very_short_text")
    if text and len(text) > args.max_chars:
        reasons.append("extreme_text_length")
    if text and REPEATED_CHAR_RE.search(text):
        reasons.append("repeated_character_run")
    if text and HTML_TAG_RE.search(text):
        flags.append("html_markup_present")
    expected = EXPECTED_SCRIPT.get(language)
    script_expected = script_ratio(text, expected) if expected and text else 0.0
    if text and len([c for c in text if c.isalpha()]) >= 8 and expected and script_expected < args.target_script_threshold:
        (reasons if args.strict_language else flags).append("script_mismatch")
    text_masked, ps = mask_pii(text)
    pii_stats.update({f"text_{k}": v for k, v in ps.items()})
    if ps:
        flags.append("structured_pii_masked")
    if heldout and text and (ngram_hashes(text) & heldout):
        reasons.append("heldout_ngram_overlap")
    digest = text_hash(language, text_masked)
    return {
        "keep": not reasons,
        "reasons": sorted(set(reasons)),
        "flags": sorted(set(flags)),
        "language": language,
        "text": text_masked,
        "content_sha256": digest,
        "script_ratios": {"expected_script": round(script_expected, 4)},
    }, norm_stats, pii_stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Parquet/CSV/JSONL file or directory")
    source.add_argument("--hf-dataset", help="Stream a Hugging Face dataset, e.g. ai4bharat/samanantar")
    p.add_argument("--languages", default="as,bn,gu,hi,kn,ml,mr,or,pa,ta,te")
    p.add_argument("--language", choices=sorted(EXPECTED_SCRIPT), help="Language for a local single-language file")
    p.add_argument("--output-dir", type=Path, default=Path("cleaned_samanantar"))
    p.add_argument("--monolingual", action="store_true", help="Clean single-text records (IndicCorpV2)")
    p.add_argument("--text-field", default="text", help="Monolingual text column")
    p.add_argument("--src-field", default="src"); p.add_argument("--tgt-field", default="tgt")
    p.add_argument("--semantic-score-field", default="", help="Optional LaBSE/alignment score column")
    p.add_argument("--semantic-threshold", type=float, default=0.70)
    p.add_argument("--min-chars", type=int, default=2); p.add_argument("--max-chars", type=int, default=5000)
    p.add_argument("--min-length-ratio", type=float, default=0.125); p.add_argument("--max-length-ratio", type=float, default=8.0)
    p.add_argument("--source-script-threshold", type=float, default=0.55)
    p.add_argument("--target-script-threshold", type=float, default=0.35)
    p.add_argument("--strict-language", action="store_true", help="Reject, rather than flag, script mismatches")
    p.add_argument("--heldout", type=Path, help="One immutable benchmark sentence per line")
    p.add_argument("--license-note", default="CC BY-NC 4.0; non-commercial use unless separately cleared")
    p.add_argument("--limit", type=int, help="Deterministic audit limit")
    args = p.parse_args(argv)
    if is_indiccorp_v2(args.hf_dataset):
        args.monolingual = True
    if args.input and not args.input.exists(): p.error(f"Input not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    rows = (
        iter_hf(args.hf_dataset, languages, monolingual=args.monolingual)
        if args.hf_dataset
        else iter_local(args.input)
    )
    heldout = load_heldout(args.heldout)
    counts: Counter[str] = Counter(); reasons: Counter[str] = Counter(); flags: Counter[str] = Counter()
    langs: Counter[str] = Counter(); norm_total: Counter[str] = Counter(); pii_total: Counter[str] = Counter()
    seen_content: set[str] = set()
    source_targets: dict[tuple[str, str], str] = {}
    output_hashes: list[str] = []
    clean_path = args.output_dir / "cleaned.jsonl"
    reject_path = args.output_dir / "rejected.jsonl"
    with clean_path.open("w", encoding="utf-8") as cf, reject_path.open("w", encoding="utf-8") as rf:
        for i, row in enumerate(rows):
            if args.limit is not None and i >= args.limit:
                break
            counts["input_rows"] += 1
            lang = choose_language(row, args.language)
            langs[lang] += 1
            if args.monolingual:
                d, n, pstats = evaluate_monolingual(row, lang, args, heldout)
                norm_total.update(n)
                pii_total.update(pstats)
                content_id = d["content_sha256"]
                dup_reason = "exact_text_duplicate"
                if d["keep"] and content_id in seen_content:
                    d["keep"] = False
                    d["reasons"].append(dup_reason)
                for f in set(d["flags"]):
                    flags[f] += 1
                payload = {
                    "idx": row.get("idx", i),
                    "text": d["text"],
                    "language": lang,
                    "data_source": row.get("data_source"),
                    "cleaning": {
                        "content_sha256": content_id,
                        "flags": sorted(set(d["flags"])),
                        "script_ratios": d["script_ratios"],
                    },
                }
            else:
                d, n, pstats = evaluate_pair(row, lang, args, heldout)
                norm_total.update(n)
                pii_total.update(pstats)
                content_id = d["pair_sha256"]
                dup_reason = "exact_pair_duplicate"
                if d["keep"] and content_id in seen_content:
                    d["keep"] = False
                    d["reasons"].append(dup_reason)
                source_key = (lang, normalized_key(d["src"]))
                target_key = normalized_key(d["tgt"])
                if source_key in source_targets and source_targets[source_key] != target_key:
                    d["flags"].append("source_has_multiple_targets")
                else:
                    source_targets[source_key] = target_key
                for f in set(d["flags"]):
                    flags[f] += 1
                payload = {
                    "idx": row.get("idx", i),
                    "src": d["src"],
                    "tgt": d["tgt"],
                    "language": lang,
                    "data_source": row.get("data_source"),
                    "cleaning": {
                        "pair_sha256": content_id,
                        "flags": sorted(set(d["flags"])),
                        "script_ratios": d["script_ratios"],
                        "semantic_score": d["semantic_score"],
                    },
                }
            if d["keep"]:
                seen_content.add(content_id)
                output_hashes.append(content_id)
                counts["cleaned_rows"] += 1
                cf.write(canonical_json(payload) + "\n")
            else:
                counts["rejected_rows"] += 1
                for r in set(d["reasons"]):
                    reasons[r] += 1
                rf.write(canonical_json({"row": payload, "reasons": sorted(set(d["reasons"]))}) + "\n")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    semantic_filter = "not applicable: monolingual mode"
    if not args.monolingual:
        semantic_filter = (
            f"{args.semantic_score_field} >= {args.semantic_threshold}"
            if args.semantic_score_field
            else "not executed: no score field"
        )
    manifest = {
      "schema_version": 3 if args.monolingual else 2,
      "mode": "monolingual" if args.monolingual else "bilingual_pair",
      "dataset": {
          "id": args.hf_dataset or str(args.input),
          "languages": languages if args.hf_dataset else [args.language or "from_rows"],
          "license_note": args.license_note,
          "text_field": args.text_field if args.monolingual else None,
          "src_field": None if args.monolingual else args.src_field,
          "tgt_field": None if args.monolingual else args.tgt_field,
      },
      "cleaner": {"script_sha256": script_hash, "unicode_form": "NFC", "preserved_codepoints": ["U+200C", "U+200D"],
                 "semantic_filter": semantic_filter,
                 "near_dedup": "not executed in this streaming pass; run merged MinHash/LSH job",
                 "decontamination": "13-word held-out n-gram fingerprints" if heldout else "not executed: no held-out registry"},
      "counts":dict(counts),"language_rows":dict(langs),"rejection_reasons":dict(reasons),"review_flags":dict(flags),
      "normalization_operations":dict(norm_total),"structured_pii_masks":dict(pii_total),
      "output":{"cleaned_file":clean_path.name,"rejected_file":reject_path.name,
                "aggregate_content_sha256":hashlib.sha256("".join(sorted(output_hashes)).encode()).hexdigest()}
    }
    (args.output_dir/"audit_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest["counts"],indent=2,sort_keys=True)); return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (OSError,ValueError,KeyError,RuntimeError,json.JSONDecodeError) as exc:
        print(f"error: {exc}",file=sys.stderr); raise SystemExit(2)
