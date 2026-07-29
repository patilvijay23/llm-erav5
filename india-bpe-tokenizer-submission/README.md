# India Multilingual BPE Tokenizer — Submission Package

Languages: English, Hindi, Telugu, Marathi. One shared SentencePiece BPE model with exactly 10,000 pieces.

## Reproduce everything

```bash
python -m pip install -r requirements.txt
python build.py --refresh
python verify.py
python -m http.server 8000 --directory site
```

Open `http://localhost:8000`.

The build script fetches the exact Wikipedia article extracts through the MediaWiki API, saves immutable text snapshots and SHA-256 hashes, trains several deterministic allocation candidates, rejects every model with English X > 1.2, then selects the compliant model with the smallest ratio spread, then writes the dashboard and downloadable artifacts.

## Metric definition

For every language:

`X = number of SentencePiece tokens / number of Unicode non-whitespace spans`

The denominator is implemented as Python regex `\\S+`. No score should be quoted without the included snapshots because Wikipedia pages can change.

## Verify / audit

`verify.py` independently reloads the final model, re-encodes every saved source, checks exact word and token counts, checks vocabulary size = 10,000, and recomputes `1000 / (max(X)-min(X))`.

## Deploy to Netlify

1. Run the build commands above.
2. Drag the `site/` folder into Netlify Drop, or connect this folder as a repository.
3. Netlify's publish directory is already set to `site` in `netlify.toml`.

The generated site contains direct downloads for the model, `.vocab`, full token list, metrics, and four source snapshots.


## Hard validity rule

The build exits with an error and does not publish dashboard artifacts unless English `X <= 1.2`. `verify.py` independently asserts the same condition. The dashboard also withholds the self-score if a noncompliant metrics file is supplied.
