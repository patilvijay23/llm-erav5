#!/usr/bin/env python3
"""Fetch four India Wikipedia articles, optimize and train one 10k SentencePiece BPE tokenizer, and build dashboard data."""
from __future__ import annotations
import argparse, hashlib, html, json, math, random, re, shutil, sys, time, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import requests
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SITE = ROOT / "site"
PAGES = {
    "English": {"code":"en", "title":"India", "url":"https://en.wikipedia.org/wiki/India"},
    "Hindi": {"code":"hi", "title":"भारत", "url":"https://hi.wikipedia.org/wiki/भारत"},
    "Telugu": {"code":"te", "title":"భారతదేశం", "url":"https://te.wikipedia.org/wiki/భారతదేశం"},
    "Marathi": {"code":"mr", "title":"भारत", "url":"https://mr.wikipedia.org/wiki/भारत"},
}
UA = "India-BPE-assignment/1.0 (educational reproducibility package)"

@dataclass
class Metrics:
    words: int
    tokens: int
    ratio: float
    chars: int
    unique_words: int


def normalize_extract(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Keep article prose and headings, but drop conventional appendices whose size changes frequently.
    stop = re.compile(r"^==\s*(See also|Notes|References|External links|इन्हें भी देखें|सन्दर्भ|बाहरी कड़ियाँ|हेही पहा|संदर्भ|बाह्य दुवे|ఇవి కూడా చూడండి|మూలాలు|బయటి లింకులు)\s*==\s*$", re.I)
    kept = []
    for line in text.splitlines():
        if stop.match(line.strip()):
            break
        line = re.sub(r"^=+\s*|\s*=+$", "", line.strip())
        if line:
            kept.append(line)
    return "\n".join(kept).strip() + "\n"


def fetch_extract(lang: str, title: str, out: Path, refresh: bool=False) -> str:
    if out.exists() and not refresh:
        return out.read_text(encoding="utf-8")
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action":"query", "prop":"extracts", "explaintext":1, "redirects":1, "format":"json", "formatversion":2, "titles":title}
    r = requests.get(api, params=params, timeout=60, headers={"User-Agent":UA})
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    if not pages or "extract" not in pages[0]:
        raise RuntimeError(f"No extract returned for {lang}:{title}")
    text = normalize_extract(pages[0]["extract"])
    out.write_text(text, encoding="utf-8")
    return text


def words(text: str) -> List[str]:
    # Assignment metric: non-whitespace spans. This is script-agnostic and exactly reproducible.
    return re.findall(r"\S+", text, flags=re.UNICODE)


def make_weighted_corpus(texts: Dict[str,str], weights: Dict[str,float], path: Path) -> None:
    # Repeat complete article lines; relative repetition controls merge allocation without altering evaluation text.
    min_words = min(len(words(t)) for t in texts.values())
    target = 90000
    with path.open("w", encoding="utf-8") as f:
        for lang, text in texts.items():
            wc = len(words(text))
            repeats = max(1, round((target / wc) * weights[lang]))
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            for _ in range(repeats):
                for line in lines:
                    f.write(line + "\n")


def train_one(corpus: Path, prefix: Path, vocab_size: int=10000) -> None:
    for ext in (".model", ".vocab"):
        p = Path(str(prefix)+ext)
        if p.exists(): p.unlink()
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), model_type="bpe",
        vocab_size=vocab_size, character_coverage=1.0,
        normalization_rule_name="nmt_nfkc", remove_extra_whitespaces=False,
        add_dummy_prefix=True, split_by_whitespace=True, split_by_unicode_script=True,
        split_by_number=True, split_digits=False, allow_whitespace_only_pieces=True,
        byte_fallback=False, hard_vocab_limit=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        input_sentence_size=0, shuffle_input_sentence=False,
        num_threads=4, minloglevel=2,
    )


def evaluate(model: Path, texts: Dict[str,str]) -> Dict[str,Metrics]:
    sp = spm.SentencePieceProcessor(model_file=str(model))
    result = {}
    for lang, text in texts.items():
        ws = words(text)
        ids = sp.encode(text, out_type=int)
        result[lang] = Metrics(len(ws), len(ids), len(ids)/len(ws), len(text), len(set(ws)))
    return result


def objective(m: Dict[str,Metrics]) -> tuple:
    """Rank only compliant models; English > 1.2 is never eligible."""
    ratios = [x.ratio for x in m.values()]
    spread = max(ratios) - min(ratios)
    english_over = max(0.0, m["English"].ratio - 1.2)
    # First tuple item makes every compliant model better than every failing model.
    return (english_over > 1e-12, english_over, spread, max(ratios), sum(ratios))


def candidates() -> Iterable[Dict[str,float]]:
    """Deterministic allocation search with enough English capacity to meet X1 <= 1.2."""
    keys = list(PAGES)
    # English must receive roughly half of the useful merge budget. The previous
    # search underweighted it and merely penalized failures.
    presets = [
        (4.0, 1.0, 1.0, 1.0),
        (6.0, 1.0, 1.0, 1.0),
        (8.0, 1.0, 1.0, 1.0),
        (10.0, 1.0, 1.0, 1.0),
        (12.0, 1.0, 1.0, 1.0),
        (8.0, 1.4, 1.4, 1.4),
        (10.0, 1.5, 2.0, 1.5),
        (12.0, 2.0, 2.0, 2.0),
        (10.0, 1.0, 2.5, 1.5),
        (10.0, 1.5, 1.5, 2.5),
    ]
    for p in presets:
        yield dict(zip(keys, p))
    rnd = random.Random(20260717)
    english_vals = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    other_vals = [0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]
    for _ in range(38):
        yield {
            "English": rnd.choice(english_vals),
            "Hindi": rnd.choice(other_vals),
            "Telugu": rnd.choice(other_vals),
            "Marathi": rnd.choice(other_vals),
        }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch Wikipedia snapshots")
    ap.add_argument("--quick", action="store_true", help="train only the equal-weight baseline")
    args=ap.parse_args()
    DATA.mkdir(exist_ok=True); ART.mkdir(exist_ok=True); SITE.mkdir(exist_ok=True)
    texts={}
    source_meta={}
    for name,p in PAGES.items():
        out=DATA/f"{p['code']}_india.txt"
        text=fetch_extract(p['code'],p['title'],out,args.refresh)
        texts[name]=text
        source_meta[name]={**p,"file":str(out.relative_to(ROOT)),"sha256":hashlib.sha256(text.encode()).hexdigest()}
        print(f"{name}: {len(words(text)):,} words")

    best=None; best_obj=None; trials=[]
    cand=list(candidates())[:1 if args.quick else None]
    work=ROOT/".work"; work.mkdir(exist_ok=True)
    for i,w in enumerate(cand,1):
        corpus=work/f"train_{i}.txt"; prefix=work/f"model_{i}"
        make_weighted_corpus(texts,w,corpus)
        try:
            train_one(corpus,prefix)
            met=evaluate(Path(str(prefix)+".model"),texts)
        except Exception as e:
            print(f"trial {i} failed: {e}", file=sys.stderr); continue
        obj=objective(met)
        trials.append({"weights":w,"ratios":{k:v.ratio for k,v in met.items()},"objective":obj[0]})
        print(f"trial {i:02d}: " + " ".join(f"{k[:2]}={v.ratio:.4f}" for k,v in met.items()) + f" spread={max(x.ratio for x in met.values())-min(x.ratio for x in met.values()):.4f}")
        if best_obj is None or obj < best_obj:
            best_obj=obj; best=(prefix,w,met)
    if best is None: raise RuntimeError("No tokenizer training trial succeeded")
    if best[2]["English"].ratio > 1.2 + 1e-12:
        raise RuntimeError(
            f"No compliant tokenizer found: best English X={best[2]['English'].ratio:.6f} > 1.2. "
            "The build has intentionally stopped and will not publish a score."
        )
    prefix,w,met=best
    shutil.copy2(Path(str(prefix)+".model"),ART/"india_multilingual_bpe_10000.model")
    shutil.copy2(Path(str(prefix)+".vocab"),ART/"india_multilingual_bpe_10000.vocab")
    sp=spm.SentencePieceProcessor(model_file=str(ART/"india_multilingual_bpe_10000.model"))
    vocab=[{"id":i,"token":sp.id_to_piece(i),"score":float(sp.get_score(i))} for i in range(sp.get_piece_size())]
    (ART/"tokens.json").write_text(json.dumps(vocab,ensure_ascii=False,indent=2),encoding="utf-8")
    with (ART/"tokens.txt").open("w",encoding="utf-8") as f:
        for x in vocab: f.write(f"{x['id']}\t{x['token']}\t{x['score']:.8f}\n")

    ratios={k:v.ratio for k,v in met.items()}; ordered=sorted(ratios.items(),key=lambda x:x[1],reverse=True)
    spread=ordered[0][1]-ordered[-1][1]
    score=math.inf if spread==0 else 1000/spread
    payload={
        "generated_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "methodology":{
            "tokenizer":"SentencePiece BPE, one shared model", "vocab_size":sp.get_piece_size(),
            "normalization":"SentencePiece nmt_nfkc; source snapshots NFC", "word_metric":"count of Unicode non-whitespace spans (regex \\S+)",
            "ratio":"SentencePiece encoded piece count / non-whitespace word count", "special_tokens":4,
            "selection":"deterministic candidate search; models with English X > 1.2 are rejected, then compliant models are ranked by minimum ratio spread",
            "training_weights":w,
        },
        "sources":source_meta,
        "languages":{k:{"words":v.words,"tokens":v.tokens,"ratio":v.ratio,"chars":v.chars,"unique_words":v.unique_words} for k,v in met.items()},
        "ranking":[{"language":k,"ratio":r} for k,r in ordered],
        "max_ratio":ordered[0][1],"min_ratio":ordered[-1][1],"spread":spread,"self_score":score,
        "english_constraint_pass":met["English"].ratio<=1.2,
        "vocab_size_exact":sp.get_piece_size()==10000,
        "trials":trials,
    }
    (ART/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    js="window.SUBMISSION_DATA = "+json.dumps(payload,ensure_ascii=False)+";\nwindow.TOKEN_VOCAB = "+json.dumps(vocab,ensure_ascii=False)+";\n"
    (SITE/"data.js").write_text(js,encoding="utf-8")
    for filename in ["india_multilingual_bpe_10000.model","india_multilingual_bpe_10000.vocab","tokens.json","tokens.txt","metrics.json"]:
        shutil.copy2(ART/filename,SITE/filename)
    for p in DATA.glob("*_india.txt"): shutil.copy2(p,SITE/p.name)
    print(f"\nSelected score: {score:.3f}; spread={spread:.6f}; vocab={sp.get_piece_size()}")
    print("Dashboard: site/index.html")

if __name__=="__main__": main()
