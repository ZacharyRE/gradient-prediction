"""Validate the copied public evidence and navigation before committing it."""
from common import *
import argparse, ast
from urllib.parse import unquote, urlsplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worktree', type=Path, required=True)
    a = ap.parse_args()
    tree = a.worktree.resolve()
    report = tree/'reports/predictor_surrogate_20261015'
    code = tree/'predictor'/EXP.name
    manifest = json.loads((report/'provenance/publication_manifest.json').read_text())
    checked = []
    for item in manifest['files']:
        target = tree/item['destination']
        assert target.is_file() and sha(target) == item['sha256'], str(target)
        if item.get('transformation'):
            assert sha(EXP/item['source']) == item['source_sha256']
            assert json.loads(target.read_text())['_publication_redaction']['original_sha256'] == item['source_sha256']
        checked.append(item['destination'])
    required = ['README.md', 'RESULTS.md', 'PROTOCOL.md', 'METHODS.md',
                'DEVELOPMENT.md', 'LITERATURE.md', 'REPRODUCING.md',
                'results/final_evidence_audit.json',
                'results/frozen_correctness.npz',
                'results/surrogate_analysis_reproduction_audit.json',
                'results/selection_pipeline_summary.json',
                'results/resource_release.json', 'provenance/overview_render.json']
    for relative in required:
        assert (report/relative).is_file(), relative
    assert json.loads((report/'results/final_evidence_audit.json').read_text())['passed']
    assert json.loads((report/'results/resource_release.json').read_text())['released']
    overview = json.loads((report/'provenance/overview_render.json').read_text())
    assert overview['readme_sha256'] == sha(report/'README.md')
    assert overview['final_evidence_audit_sha256'] == sha(report/'results/final_evidence_audit.json')
    files = [p for root in [report, code] for p in root.rglob('*') if p.is_file()]
    forbidden = {'.pt', '.pth', '.safetensors', '.jsonl', '.log', '.pyc', '.pyo'}
    allowed_files=set(checked)|{str((report/'provenance/publication_manifest.json').relative_to(tree))}
    assert {str(p.relative_to(tree)) for p in files}==allowed_files, 'Unexpected unmanifested public artifact'
    for path in files:
        assert path.suffix not in forbidden, f'Unexpected private/bulk artifact: {path}'
        assert path.stat().st_size < 100*1024**2, f'GitHub file size limit: {path}'
    source_files = sorted((code/'scripts').glob('*.py'))
    for path in source_files:
        ast.parse(path.read_text(), filename=str(path))
    def check_public_provenance(value):
        if isinstance(value,list):
            for v in value:check_public_provenance(v)
        elif isinstance(value,dict):
            for k,v in value.items():
                assert k!='pid' and not k.endswith('_pid') and k not in {'process_start','foreign_processes','cmdline'}, k
                check_public_provenance(v)
    for path in (report/'provenance').glob('*.json'):
        check_public_provenance(json.loads(path.read_text()))
    documents = sorted(report.glob('*.md')) + [tree/'README.md', tree/'predictor/README.md', tree/'docs/RESULTS.md']
    links = []
    for path in documents:
        # Skip fenced command examples; check the local links a reader can follow.
        body = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', body):
            target = target.strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            linked = (tree/parsed.path.lstrip('/') if parsed.path.startswith('/') else path.parent/unquote(parsed.path)).resolve()
            assert linked.exists(), f'Broken link in {path}: {target}'
            links.append(dict(document=str(path.relative_to(tree)), target=target))
    result = dict(passed=True, epoch=time.time(), worktree=str(tree),
                  manifest_sha256=sha(report/'provenance/publication_manifest.json'),
                  manifest_files_verified=len(checked), public_files=len(files),
                  parsed_python_sources=len(source_files), local_links_checked=len(links),
                  largest_file_bytes=max(p.stat().st_size for p in files),
                  scope='Exact copied-file hashes, required completed evidence, Python syntax, existing local Markdown link targets, file size limits, and exclusion of raw JSONL, logs and model-weight formats. This does not verify remote publication, external URLs, GPU rerun portability or independent response grading.')
    write(EXP/'provenance/publication_audit.json', result)
    log(result)


if __name__ == '__main__':
    main()
