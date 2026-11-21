#!/usr/bin/env python3
"""Check tracked release files for private paths, credentials and old imports."""
from pathlib import Path
import ast
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    q = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True)
    if q.returncode == 0 and q.stdout:
        files = [ROOT / s.decode() for s in q.stdout.split(b'\0') if s]
    else:
        files = [p for p in ROOT.rglob('*') if p.is_file()
                 and not set(p.relative_to(ROOT).parts) & {'third_party', '__pycache__', '.git', 'data', 'weights', 'runs'}]
    errors = []
    private = re.compile('|'.join(re.escape('/' + part) for part in ('inspire/', 'Users/', 'root/miniconda')))
    secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in {'.pth', '.pt', '.ckpt', '.safetensors', '.pem', '.key'}:
            errors.append(f'Checkpoint/credential must not be tracked: {relative}')
        if path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f'Oversized release file: {relative}')
        if path.suffix not in {'.py', '.json', '.toml', '.md', '.txt', '.sh', '.yaml', '.yml', '.html', '.css', '.js', '.svg'}:
            continue
        text = path.read_text()
        if private.search(text):
            errors.append(f'Private absolute path: {relative}')
        if secret.search(text):
            errors.append(f'Credential-like value: {relative}')
        if path.suffix == '.py':
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                errors.append(f'Python syntax: {relative}: {exc}')
                continue
            for node in ast.walk(tree):
                modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                if any(m == 'ufo' or m.startswith('ufo.') for m in modules):
                    errors.append(f'Old package import: {relative}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f'RELEASE_AUDIT_PASS files={len(files)}')


if __name__ == '__main__':
    main()
