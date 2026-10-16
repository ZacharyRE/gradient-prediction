"""List measured predictor preparation components, without treating reused work as free."""
from common import *


def collection_seconds(cache):
    receipt=json.loads((cache/'complete.json').read_text())
    return sum(r['seconds'] for r in receipt['audits'] if r['split'] in ['predictor_train','predictor_dev'])


def main():
    source=Path(json.loads((EXP/'configs/protocol.json').read_text())['source_study'])
    rows=[]
    for task in ['paws','mnli']:
        inherited_collection=collection_seconds(source/'cache'/task)
        suffix='_globalfit' if task=='mnli' else ''
        for seed_id in [123,124,125]:
            directory=source/'models'/f'{task}_predictor{suffix}_p{seed_id}'
            if not (directory/'summary.json').exists():continue
            summary=json.loads((directory/'summary.json').read_text())
            rows.append(dict(task=task,predictor=directory.name,predictor_sha256=summary['best_sha256'],origin='inherited',collection_seconds=inherited_collection,fit_seconds=summary['seconds'],base_predictor_seconds=0.,teacher_trajectory_seconds=0.,total_recorded_preparation_seconds=inherited_collection+summary['seconds'],source_summary_sha256=sha(directory/'summary.json')))
        for path in sorted((EXP/'models').glob(f'{task}__refit*/summary.json')):
            summary=json.loads(path.read_text());config=json.loads((path.parent/'config.json').read_text())
            if 'source_cache' not in config or 'best_sha256' not in summary:continue
            cache=Path(config['source_cache']);seconds=collection_seconds(cache)
            rows.append(dict(task=task,predictor=path.parent.name,predictor_sha256=summary['best_sha256'],origin='new_offline_fit',collection_seconds=seconds,fit_seconds=summary['seconds'],base_predictor_seconds=0.,teacher_trajectory_seconds=0.,total_recorded_preparation_seconds=seconds+summary['seconds'],source_summary_sha256=sha(path)))
        by_sha={r['predictor_sha256']:r for r in rows}
        for path in sorted((EXP/'models').glob(f'{task}__multistate*/summary.json')):
            summary=json.loads(path.read_text());config=json.loads((path.parent/'config.json').read_text())
            if 'base_predictor_sha256' not in config or 'best_sha256' not in summary:continue
            base=by_sha[config['base_predictor_sha256']]['total_recorded_preparation_seconds']
            teacher=json.loads((EXP/'models'/f'{task}__fast_oracle32_u101_p123'/'summary.json').read_text())['online_seconds']
            collection=summary['additional_collection_cost']
            rows.append(dict(task=task,predictor=path.parent.name,predictor_sha256=summary['best_sha256'],origin='new_multistate_fit',collection_seconds=collection,fit_seconds=summary['seconds'],base_predictor_seconds=base,teacher_trajectory_seconds=teacher,total_recorded_preparation_seconds=base+teacher+collection+summary['seconds'],source_summary_sha256=sha(path)))
    write(EXP/'results/predictor_preparation_costs.json',dict(updated=time.time(),rows=rows,note='Recorded preparation for each artifact, including reused collection and prerequisite predictors/teacher trajectories. Do not sum rows: collections and base predictors are shared. Excludes shared warmup/model startup and separate held-out gradient-test collection. Fitting timers include checkpoint selection and serialization inside those fit loops. These are historical implementation costs, not theoretical lower bounds. Occupancy/timing replay limitations apply; no clean cold-preparation benchmark is claimed. Online-only RLS has no inherited predictor preparation; its initialization and online calibration are charged in its training receipts.'))
    log(dict(predictors=len(rows)))


if __name__=='__main__':main()
