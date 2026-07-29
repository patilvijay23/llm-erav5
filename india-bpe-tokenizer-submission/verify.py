#!/usr/bin/env python3
import json,re,sys
from pathlib import Path
import sentencepiece as spm
R=Path(__file__).resolve().parent
m=json.loads((R/'artifacts/metrics.json').read_text(encoding='utf-8'))
sp=spm.SentencePieceProcessor(model_file=str(R/'artifacts/india_multilingual_bpe_10000.model'))
assert sp.get_piece_size()==10000, sp.get_piece_size()
assert m['english_constraint_pass'], 'Published model violates English X <= 1.2'
for lang,s in m['sources'].items():
    text=(R/s['file']).read_text(encoding='utf-8')
    words=len(re.findall(r'\S+',text,flags=re.UNICODE)); tokens=len(sp.encode(text,out_type=int)); ratio=tokens/words
    expected=m['languages'][lang]
    assert (words,tokens)==(expected['words'],expected['tokens'])
    assert abs(ratio-expected['ratio'])<1e-12
    if lang == 'English': assert ratio <= 1.2 + 1e-12, ratio
    print(f'{lang:8s} words={words:7d} tokens={tokens:7d} X={ratio:.8f}')
ratios=[x['ratio'] for x in m['languages'].values()]
score=1000/(max(ratios)-min(ratios))
assert abs(score-m['self_score'])<1e-8
print(f'PASS vocab=10000 score={score:.6f}')
