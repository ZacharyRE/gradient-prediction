"""Freeze previously unused Countdown data and read-only source identities."""
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PRED = HERE.parent
ROOT = PRED.parent
sys.path.insert(0, str(ROOT / 'research/0.5B_countdown_lora/scripts'))
from countdown import check, number_key, problem, solution, solve


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest_path = HERE / 'data/manifest.json'
    assert not manifest_path.exists(), 'Refusing to overwrite frozen data'
    folders = [PRED / name / 'data' for name in (
        'predictor_countdown', 'predictor_countdown2',
        'predictor_dynamic/one-step', 'predictor_dynamic/adaptive-validation',
        'predictor_dynamic/three-layer-adaptive',
        'main_model_forward_transfer_20260923')]
    folders += [ROOT / 'research' / name / 'data' for name in (
        '0.5B_countdown_lora', '0.5B_sft_lora', 'sft_recovery_0p5b_20260911')]
    reserved = {(2, 3, 7)}
    prior = {}
    for folder in folders:
        for p in sorted(folder.glob('*.jsonl')):
            real = p.resolve()
            if str(real) in prior:
                continue
            rows = [json.loads(line) for line in p.read_text().splitlines() if line]
            keys = {number_key(row) for row in rows if 'nums' in row}
            reserved.update(keys)
            if keys:
                prior[str(real)] = dict(sha256=sha(real), number_multisets=len(keys))

    source = ROOT / 'research/0.5B_sft_lora/data/countdown_source.parquet'
    raw = pq.read_table(source).to_pylist()
    rng = random.Random(202609231)
    order = list(range(len(raw)))
    rng.shuffle(order)
    pool = {3: [], 4: []}
    seen = set(reserved)
    # 6 history episodes (20 each), 8 probes (48 each), loss256, accuracy1024.
    target_per_size = (6 * 20 + 8 * 48 + 256 + 1024) // 2
    for index in order:
        row = raw[index]
        key = number_key(row)
        size = len(row['nums'])
        if size not in pool or key in seen or len(pool[size]) >= target_per_size:
            continue
        nums, target = list(map(int, row['nums'])), int(row['target'])
        tree = solve(nums, target)
        if tree is None:
            continue
        answer = solution(tree, nums, target)
        assert check(answer, nums, target)['correct']
        pool[size].append(dict(source_index=index,
            source='Jiayi-Pan/Countdown-Tasks-3to4', nums=nums, target=target,
            problem=problem(nums, target), solution=answer, gold=str(target),
            supervision='exact_solver'))
        seen.add(key)
        if all(len(v) == target_per_size for v in pool.values()):
            break
    assert all(len(v) == target_per_size for v in pool.values())
    cursors = {3: 0, 4: 0}
    splits = {}

    def take(name, n):
        assert n % 2 == 0
        rows = []
        for size in (3, 4):
            start = cursors[size]
            rows.extend(pool[size][start:start+n//2])
            cursors[size] += n//2
        rng.shuffle(rows)
        rows = [{**row, 'split': name} for row in rows]
        path = HERE / 'data' / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        splits[name] = dict(n=n, sha256=sha(path),
            source_indices=[r['source_index'] for r in rows],
            number_counts=dict(Counter(len(r['nums']) for r in rows)))

    for state in (0, 1, 2, 4, 8, 16):
        take(f'history_s{state}_cal', 4)
        take(f'history_s{state}_val', 16)
    for state in (16, 32):
        for probe in range(4):
            prefix = f'probe_s{state}_p{probe}'
            take(prefix + '_cal', 4)
            take(prefix + '_val', 16)
            take(prefix + '_held', 28)
    take('loss256', 256)
    take('accuracy1024', 1024)
    assert all(cursors[n] == target_per_size for n in (3, 4))
    frozen = {}
    for seed in (101, 102, 103):
        predictor = PRED / 'predictor_dynamic/one-step/models' / f'y_mask_pos_s{seed+22}/best.pt'
        states = {}
        for step in (0, 1, 2, 4, 8, 16, 32):
            adapter = (PRED / 'predictor_countdown2/models/teacher/step-32'
                if step == 0 else PRED / 'predictor_dynamic/adaptive-validation/models'
                / f'single_oracle_s{seed+22}_u{seed}' / f'step-{step}')
            states[str(step)] = dict(path=str(adapter),
                sha256=sha(adapter / 'adapter_model.safetensors'),
                config_sha256=sha(adapter / 'adapter_config.json'))
        frozen[str(seed)] = dict(predictor=str(predictor), predictor_sha256=sha(predictor),
                                 model_states=states)
    dependencies = {}
    scripts = PRED / 'predictor_dynamic/adaptive-validation/scripts'
    for p in [scripts / name for name in ('core.py', 'predictor.py', 'single_layer.py')]:
        shutil.copyfile(p, HERE / 'provenance' / p.name)
        dependencies[str(p)] = sha(p)
    for p in [ROOT / 'research/0.5B_countdown_lora/scripts' / name
              for name in ('common.py', 'sft_helpers.py', 'countdown.py',
                           'gradient_geometry/extraction.py')]:
        dependencies[str(p)] = sha(p)
    result = dict(selection_seed=202609231, source=str(source), source_sha256=sha(source),
        source_rows=len(raw), excluded_number_multisets=len(reserved), prior_files=prior,
        splits=splits, new_unique_number_multisets=2*target_per_size,
        overlap_with_prior=0, overlap_across_new_splits=0,
        all_reference_solutions_exact_verified=True, starts=frozen,
        dependencies=dependencies, plan_sha256=sha(HERE / 'PLAN.md'),
        selection_before_model_outputs=True, input_truncation=False,
        pretraining_contamination_assessed=False)
    manifest_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(new_examples=2*target_per_size,
        prior_files=len(prior), excluded_multisets=len(reserved), split_count=len(splits))))


if __name__ == '__main__':
    main()
