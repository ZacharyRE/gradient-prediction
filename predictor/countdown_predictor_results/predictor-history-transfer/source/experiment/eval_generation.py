"""Prespecified primary-state probe-0 accuracy with every input token preserved."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'predictor_dynamic/adaptive-validation/scripts'))
from core import MODEL, prompt
from countdown import check


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=(101, 102, 103), required=True)
    args = ap.parse_args()
    manifest = json.loads((HERE / 'data/manifest.json').read_text())
    result = json.loads((HERE / 'results' / f'u{args.seed}.json').read_text())
    assert result['complete'] and result['frozen_base_unchanged']
    assert sha(HERE / 'PLAN.md') == manifest['plan_sha256']
    test = HERE / 'data/accuracy1024.jsonl'
    assert sha(test) == manifest['splits']['accuracy1024']['sha256']
    rows = [json.loads(line) for line in test.read_text().splitlines()]
    assert len(rows) == 1024
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = [prompt(tokenizer, row) for row in rows]
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in prompts]
    output_cap = 2048
    max_model_len = ((max(lengths) + output_cap + 255)//256)*256
    assert max_model_len <= AutoConfig.from_pretrained(MODEL, local_files_only=True).max_position_embeddings
    probe = next(r for r in result['probes'] if r['model_step'] == 32 and r['probe'] == 0)
    start = manifest['starts'][str(args.seed)]['model_states']['32']
    jobs = [('baseline', start['path'], start['sha256'])]
    jobs += [(method, probe['methods'][method]['adapter_path'],
              probe['methods'][method]['adapter_sha256'])
             for method in ('carry', 'reset', 'frozen', 'oracle')]
    folder = HERE / 'results/accuracy' / f'u{args.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=args.seed, model_step=32, probe=0, test_sha256=sha(test),
        n=1024, input_truncation=False, max_observed_prompt_tokens=max(lengths),
        max_new_output_tokens=output_cap, temperature=0, generation_seed=42,
        base_model=MODEL, dtype='bfloat16', max_num_seqs=128, kv_cache_gib=8,
        backend='native vLLM LoRA', plan_sha256=manifest['plan_sha256'],
        adapters={name: dict(path=path, sha256=digest) for name, path, digest in jobs})
    protocol_path = folder / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        write(protocol_path, protocol)
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if len(summary) == 5:
        return
    llm = LLM(model=MODEL, dtype='bfloat16', max_model_len=max_model_len,
        max_num_seqs=128, kv_cache_memory_bytes=8*1024**3, gpu_memory_utilization=.1,
        seed=42, enable_lora=True, max_lora_rank=64, max_loras=1,
        enable_prefix_caching=False, async_scheduling=False, enforce_eager=True,
        attention_config={'backend': 'TRITON_ATTN'})
    try:
        for index, (name, path, digest) in enumerate(jobs, 1):
            if name in summary:
                continue
            assert sha(Path(path) / 'adapter_model.safetensors') == digest
            started = time.perf_counter()
            outputs = llm.generate(prompts,
                SamplingParams(temperature=0, max_tokens=output_cap, seed=42),
                lora_request=LoRARequest(name, index, path), use_tqdm=False)
            records = []
            for row, text, output in zip(rows, prompts, outputs):
                answer = output.outputs[0]
                records.append(dict(source_index=row['source_index'], nums=row['nums'],
                    target=row['target'], full_prompt=text, prediction=answer.text,
                    output_tokens=len(answer.token_ids), finish_reason=answer.finish_reason,
                    **check(answer.text, row['nums'], row['target'])))
            assert len(records) == 1024
            target = folder / f'{name}.jsonl'
            temporary = target.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(row) + '\n' for row in records))
            temporary.replace(target)
            correct = sum(row['correct'] for row in records)
            summary[name] = dict(n=1024, correct=correct, accuracy=correct/1024,
                output_cap_hits=sum(row['finish_reason'] == 'length' for row in records),
                seconds=time.perf_counter()-started, records_sha256=sha(target))
            write(summary_path, summary)
            print(json.dumps(dict(seed=args.seed, method=name, **summary[name])), flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=15)


if __name__ == '__main__':
    main()
