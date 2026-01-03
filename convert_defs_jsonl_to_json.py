#!/usr/bin/env python3
import argparse, json, os, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "defs-*.jsonl")))
    if not files:
        raise SystemExit(f"No defs-*.jsonl files found in {args.in_dir}")

    for path in files:
        out_path = os.path.join(
            args.out_dir,
            os.path.basename(path).replace(".jsonl", ".json")
        )
        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                mapping[rec["word"]] = rec.get("senses", [])

        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(mapping, out, ensure_ascii=False)

        print(f"Wrote {out_path} ({len(mapping):,} words)")

if __name__ == "__main__":
    main()
