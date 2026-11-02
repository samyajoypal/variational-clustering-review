"""Fetch only pinned upstream source prerequisites, never datasets or weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def fetch(names=None, verify_only=False):
    pins = json.loads((ROOT/'provenance/upstream_sources.json').read_text())
    result = []
    for pin in pins:
        if names and pin['name'] not in names: continue
        base = ROOT/'experiments/paper1_review/vendor_sources'/pin['name']
        missing = []
        for relative, expected in pin['files'].items():
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe source path')
            target = base/relative
            if target.exists():
                if target.is_symlink() or digest(target.read_bytes()) != expected:
                    raise ValueError(f'Existing source differs; refusing overwrite: {relative}')
            else: missing.append(relative)
        if missing:
            if verify_only: raise FileNotFoundError(f'{pin["name"]}: {len(missing)} prerequisites missing')
            # Full historical archives may contain weights/data; fetch only the
            # explicitly inventoried source files at the exact recorded commit.
            upstream=pin['repository'].removeprefix('https://github.com/')
            def download(relative):
                url=f'https://raw.githubusercontent.com/{upstream}/{pin["commit"]}/{quote(relative)}'
                request=Request(url,headers={'User-Agent':'variational-clustering-review/0.2'})
                with urlopen(request,timeout=120) as stream:content=stream.read(5*1024*1024+1)
                if len(content)>5*1024*1024:raise ValueError('Unexpected source file size')
                if digest(content)!=pin['files'][relative]:raise ValueError(f'File SHA256 mismatch: {relative}')
                return relative,content
            with ThreadPoolExecutor(max_workers=4) as pool:
                prepared=list(pool.map(download,missing))
            # All selected downloads must verify before any are persisted.
            for relative,content in prepared:
                target=base/relative;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(content)
        result.append({'source':pin['name'],'verified_files':len(pin['files']),'downloaded_files':len(missing)})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--source',action='append')
    args=parser.parse_args()
    print(json.dumps(fetch(args.source,args.verify_only),indent=2))
