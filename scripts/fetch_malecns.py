#!/usr/bin/env python3
"""Download MaleCNS v1.0 flat-connectome tables into data/raw/malecns_v1/.

Uses curl with multi-connection range segments when available (much faster than
urllib on throttled links). Resumes partial downloads.

Does not commit the ~1 GB files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "malecns_v1"
MANIFEST = OUT / "download_manifest.json"

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": f"{BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": f"{BASE}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather": f"{BASE}/body-neurotransmitters-male-cns-v1.0.feather",
}

N_PARTS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None


def curl_range(url: str, dest: Path, start: int, end: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Inclusive byte range for HTTP.
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "-H",
        f"Range: bytes={start}-{end}",
        "-o",
        str(dest),
        url,
    ]
    if dest.exists() and dest.stat().st_size == (end - start + 1):
        return
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def download_multipart(url: str, target: Path, n_parts: int = N_PARTS) -> None:
    total = content_length(url)
    if total is None:
        raise RuntimeError(f"no Content-Length for {url}")

    # Prefer an existing single-stream partial if present.
    legacy_partials = [
        target.with_suffix(target.suffix + ".partial"),
        target.with_suffix(target.suffix + ".part"),
    ]
    parts_dir = target.with_suffix(target.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    # If we already have a contiguous partial from curl -C -, seed part 0 by
    # splitting later; simplest: if legacy partial covers everything, rename.
    for lp in legacy_partials:
        if lp.exists() and lp.stat().st_size == total:
            lp.replace(target)
            return

    size = total // n_parts
    ranges = []
    for i in range(n_parts):
        start = i * size
        end = total - 1 if i == n_parts - 1 else (i + 1) * size - 1
        ranges.append((i, start, end))

    # If a legacy .part exists with some prefix bytes, keep using single-stream
    # resume for that file only when multipart would waste work — but for
    # stuck-slow downloads, multipart from scratch of remaining is better.
    # Here: if legacy partial exists and is substantial, continue single-stream.
    best_legacy = None
    for lp in legacy_partials:
        if lp.exists() and lp.stat().st_size > 0:
            if best_legacy is None or lp.stat().st_size > best_legacy.stat().st_size:
                best_legacy = lp
    if best_legacy is not None and best_legacy.stat().st_size > total * 0.2:
        print(
            f"  resuming single-stream from {best_legacy.stat().st_size}/{total} bytes"
        )
        last_err: Exception | None = None
        for attempt in range(1, 40):
            try:
                subprocess.check_call(
                    [
                        "curl",
                        "-L",
                        "--fail",
                        "--retry",
                        "5",
                        "--retry-all-errors",
                        "--retry-delay",
                        "3",
                        "--connect-timeout",
                        "30",
                        "--speed-time",
                        "60",
                        "--speed-limit",
                        "1000",
                        "--continue-at",
                        "-",
                        "-o",
                        str(best_legacy),
                        url,
                    ]
                )
                if best_legacy.stat().st_size == total:
                    best_legacy.replace(target)
                    return
                print(
                    f"  incomplete after attempt {attempt}: "
                    f"{best_legacy.stat().st_size}/{total}; retrying…"
                )
            except subprocess.CalledProcessError as e:
                last_err = e
                got = best_legacy.stat().st_size if best_legacy.exists() else 0
                print(f"  curl failed attempt {attempt} at {got}/{total}: {e}")
                # brief pause before resume
                import time

                time.sleep(min(2 * attempt, 30))
        raise RuntimeError(
            f"could not finish download after retries: {best_legacy}"
        ) from last_err

    print(f"  multipart {n_parts}× ranges, {total} bytes")

    def one(item: tuple[int, int, int]) -> None:
        i, start, end = item
        part = parts_dir / f"{i:02d}.bin"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return
        curl_range(url, part, start, end)
        if part.stat().st_size != expected:
            raise RuntimeError(f"part {i} size {part.stat().st_size} != {expected}")

    with ThreadPoolExecutor(max_workers=n_parts) as ex:
        futs = [ex.submit(one, r) for r in ranges]
        for fut in as_completed(futs):
            fut.result()

    tmp = target.with_suffix(target.suffix + ".assembling")
    with tmp.open("wb") as out:
        for i, _, _ in ranges:
            part = parts_dir / f"{i:02d}.bin"
            with part.open("rb") as inp:
                shutil.copyfileobj(inp, out, length=1 << 20)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"assembled size {tmp.stat().st_size} != {total}")
    tmp.replace(target)
    shutil.rmtree(parts_dir, ignore_errors=True)
    for lp in legacy_partials:
        if lp.exists():
            lp.unlink()


def download_file(name: str, url: str) -> Path:
    target = OUT / name
    if target.exists():
        print(f"exists: {target}")
        return target
    print(f"download: {name}")
    if shutil.which("curl"):
        download_multipart(url, target)
    else:
        partial = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    return target


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, url in FILES.items():
        target = download_file(name, url)
        digest = sha256_file(target)
        hashes[name] = digest
        print(f"  sha256={digest}")

    MANIFEST.write_text(
        json.dumps({"files": hashes, "source_base": BASE}, indent=2) + "\n"
    )
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
