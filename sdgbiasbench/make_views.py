"""
Turn the Hugging Face release into one file per evidence view (paper Sec. 3.1).

    python -m sdgbiasbench.make_views --data-root data/SDGBiasBench --out views/

For every context test T<k> it writes
    <task>_T<k>.jsonl        image + context + question   (Full view; IMG+Q view at T5)
    <task>_T<k>_text.jsonl   blank image + context + question + "Ignore the image."
                             (CTX+Q view; Q-only view at T5)
with task = mcq (k = 1..5) and regression (k = 2, 5). Each record has `index` (the
item id, shared by all views of one question), `images` (absolute paths), `question`,
the options, `answer` and `split`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

TEXT_ONLY_SUFFIX = "\nIgnore the image."
BLANK_IMAGE = "images/blank.png"
EXTRA_FIELDS = ("split", "level", "pillar", "indicator", "task", "task_name", "target", "num_options")


def to_view(rec: dict, root: str, text_only: bool) -> dict:
    image = BLANK_IMAGE if text_only else rec["image"]
    out = {
        "id": rec["id"],
        "index": rec["item_id"],  # shared by all views of one question -> views align on it
        "images": [os.path.join(root, image)],
        "question": rec["question"] + (TEXT_ONLY_SUFFIX if text_only else ""),
    }
    for o in ("A", "B", "C"):
        if rec.get(o):
            out[o] = rec[o]
    out["answer"] = rec["answer"] if isinstance(rec["answer"], str) else str(rec["answer"])
    out.update({k: rec[k] for k in EXTRA_FIELDS if k in rec})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="local copy of the Hugging Face dataset repo")
    ap.add_argument("--out", required=True, help="output folder for the view files")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], help="which splits to include")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.data_root)
    os.makedirs(args.out, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(root, "data", "*.jsonl"))):
        with open(src, encoding="utf-8") as f:
            records = [r for r in map(json.loads, f) if r["split"] in args.splits]
        stem = os.path.splitext(os.path.basename(src))[0]            # e.g. mcq_T1
        for text_only in (False, True):
            name = stem + ("_text" if text_only else "")
            path = os.path.abspath(os.path.join(args.out, name + ".jsonl"))
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(to_view(r, root, text_only), ensure_ascii=False) + "\n")
            print(f"[views] {path}  ({len(records)} questions)")


if __name__ == "__main__":
    main()
