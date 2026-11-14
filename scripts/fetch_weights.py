"""Download the ten fine-tuned weight files and verify their SHA-256 (no weights are stored in Git itself).

Source of the files is given in `weights_source.json` (a Hugging Face Hub model repository) or, for offline use, a local directory with the same layout.
  python scripts/fetch_weights.py                       # download from the Hub, verify, place under ./weights and ./s43_weights
  python scripts/fetch_weights.py --local-dir /path/to  # copy (or --link) from a local directory with the same relative paths
Exit status is non-zero if any file is missing or its SHA-256 differs from `weights_manifest_10m.json`. Existing correct files are kept.
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "weights_manifest_10m.json"))
    ap.add_argument("--source", default=str(ROOT / "weights_source.json"))
    ap.add_argument("--dest", default=str(ROOT), help="repository root to place weights in")
    ap.add_argument("--local-dir", help="copy from this directory instead of downloading")
    ap.add_argument("--link", action="store_true", help="with --local-dir: symlink instead of copying")
    a = ap.parse_args()
    dest = Path(a.dest)
    manifest = json.loads(Path(a.manifest).read_text())
    members = manifest["seed42_members"] + manifest["seed43_members"]
    state = {}   # download settings are resolved only if a file is actually missing

    def hub():
        if not state:
            src = json.loads(Path(a.source).read_text()); repo, revision = src["huggingface_repo"], src.get("revision", "main")
            if "REPLACE" in repo:
                sys.exit(f"Some weights are missing and weights_source.json still contains a placeholder ({repo}). Set the Hugging Face repository id, or pass --local-dir.")
            from huggingface_hub import hf_hub_download
            state.update(repo=repo, revision=revision, dl=hf_hub_download)
        return state
    bad = []
    for m in members:
        rel, want = m["path"], m["sha256"]; out = dest / rel; out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and sha256(out) == want:
            print("ok (already present)", rel); continue
        if out.exists() or out.is_symlink(): out.unlink()
        if a.local_dir:
            s = Path(a.local_dir) / rel
            if not s.exists(): bad.append((rel, "missing in --local-dir")); continue
            out.symlink_to(s.resolve()) if a.link else shutil.copyfile(s, out)
        else:
            h = hub(); tmp = h["dl"](repo_id=h["repo"], filename=rel, revision=h["revision"], local_dir=str(dest))   # saves to dest/rel
            assert Path(tmp).resolve() == out.resolve() or out.exists(), tmp
        got = sha256(out)
        if got != want: bad.append((rel, f"sha256 {got[:12]}… != {want[:12]}…")); continue
        print("verified", rel)
    if bad:
        for b in bad: print("FAILED", *b, file=sys.stderr)
        return 1
    print(f"all {len(members)} weight files present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
