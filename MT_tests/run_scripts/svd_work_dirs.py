"""List (and optionally delete) the SVD tools' leftover ``*_svd_work`` folders.

Every Panther-EM run (``svd_match_filament.py``, ``svd_score_sites.py``,
``svd_trace_maps.py``) writes a work folder next to its output: the micrograph's whitening,
the template's decomposition (~2 GB at the default rank, 4-7 GB at k 256 or nside 64) and
the calibration tables. Nothing reads them after the run. The tools now delete theirs
when they end (unless ``--keep-work``), but runs from before 2026-11-13, and runs killed
with SIGKILL, leave them behind: they are what filled the disk on 2026-11-12.

Only folders named ``*_svd_work`` under MT_tests/ are considered. A folder touched in the
last ``--min-age-min`` minutes is treated as a run in progress and never deleted.

Usage (from MT_tests/):
    python run_scripts/svd_work_dirs.py            # list, largest first, with the total
    python run_scripts/svd_work_dirs.py --delete   # delete the listed ones
"""

import argparse
import pathlib
import shutil
import time

MT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def folder_size(path: pathlib.Path) -> tuple[int, float]:
    """(bytes, newest modification time) of everything under ``path``."""
    total, newest = 0, path.stat().st_mtime
    for item in path.rglob("*"):
        try:
            info = item.stat()
        except FileNotFoundError:
            continue
        newest = max(newest, info.st_mtime)
        if item.is_file():
            total += info.st_size
    return total, newest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete", action="store_true", help="delete the listed folders")
    parser.add_argument("--min-age-min", type=float, default=60.0,
                        help="skip folders touched more recently than this (a live run)")
    args = parser.parse_args()

    now = time.time()
    rows = []
    for path in MT_ROOT.rglob("*_svd_work"):
        if path.is_dir() and not any(p.name.endswith("_svd_work") for p in path.parents):
            size, newest = folder_size(path)
            rows.append((size, (now - newest) / 60.0, path))
    rows.sort(key=lambda r: r[0], reverse=True)
    old = [r for r in rows if r[1] >= args.min_age_min]
    print(f"{'GB':>6}  {'age':>8}  folder")
    for size, age, path in rows:
        flag = "" if age >= args.min_age_min else "   (recent: a run in progress? kept)"
        print(f"{size / 1e9:6.2f}  {age / 60:6.1f} h  {path.relative_to(MT_ROOT)}{flag}")
    total_old = sum(r[0] for r in old) / 1e9
    print(f"{len(rows)} folder(s), {sum(r[0] for r in rows) / 1e9:.1f} GB; "
          f"{len(old)} older than {args.min_age_min:g} min, {total_old:.1f} GB")
    free = shutil.disk_usage(MT_ROOT).free / 1e9
    print(f"free on this disk: {free:.0f} GB")
    if args.delete:
        for _, _, path in old:
            shutil.rmtree(path, ignore_errors=True)
        print(f"deleted {len(old)} folder(s); free now "
              f"{shutil.disk_usage(MT_ROOT).free / 1e9:.0f} GB")
    elif old:
        print("nothing deleted (--delete to remove them; every one can be regenerated)")


if __name__ == "__main__":
    main()
