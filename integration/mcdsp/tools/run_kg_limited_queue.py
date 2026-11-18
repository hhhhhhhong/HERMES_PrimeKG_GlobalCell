"""One seed and at most two user GPU processes, shared across both suites."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
MODES = ('mean', 'single_query', 'task_query', 'structure', 'gating',
         'fine_tokens', 'norm_four', 'temp_four', 'temp_single')
RECIPES = {'aux1': ('none', 0.), 'aux1_original': ('original', .1)}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def proc_args(pid):
    try:
        return Path('/proc/%d/cmdline' % pid).read_bytes().decode().strip('\0').split('\0')
    except (FileNotFoundError, ProcessLookupError):
        return []


def identify(pid, representation, auxiliary):
    argv = proc_args(pid)
    if '--output' not in argv or '--mode' not in argv or '--device' not in argv:
        return None
    try:
        output = Path(argv[argv.index('--output') + 1]).resolve()
        gpu = int(argv[argv.index('--device') + 1].split(':')[-1])
        mode = argv[argv.index('--mode') + 1]
        seed = int(argv[argv.index('--train-seed') + 1]) if '--train-seed' in argv else 0
        if seed != 0:
            return None
        if any(s.endswith('/run_kg_representation_cv5.py') for s in argv):
            dataset = argv[argv.index('--dataset') + 1]
            if output != representation / dataset / 'seed0':
                return None
            return dict(kind='representation', dataset=dataset, seed=0, mode=mode,
                        directory=str(output / mode), gpu=gpu, pid=pid)
        if any(s.endswith('/run_kg_auxiliary_cv5.py') for s in argv):
            if output.parent != auxiliary or output.name not in RECIPES:
                return None
            return dict(kind='auxiliary', recipe=output.name, seed=0, mode=mode,
                        directory=str(output / mode), gpu=gpu, pid=pid)
    except (ValueError, IndexError):
        return None
    return None


def key(job):
    return (job['kind'], job.get('dataset', job.get('recipe')), job['mode'])


def gpu_user_counts():
    # Count every GPU process owned by this user, including other experiment roots.
    gpu_rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {row[1].strip(): int(row[0]) for row in csv.reader(gpu_rows.splitlines())}
    app_rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    counts = {}
    for uuid, pid in csv.reader(app_rows.splitlines()):
        try:
            if Path('/proc/%d' % int(pid)).stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        gpu = mapping[uuid.strip()]
        counts.setdefault(gpu, set()).add(int(pid))
    return counts


def eligible_gpu(active, occupied, gpus):
    # Include freshly launched processes before their CUDA context is visible.
    for gpu in gpus:
        pids = set(occupied.get(gpu, set()))
        pids.update(job['pid'] for job in active.values() if job['gpu'] == gpu)
        if len(pids) < 2:
            return gpu
    return None


def representation_report(root):
    lines = ['# 单 seed KG 表示实验', '',
             '当前范围：train seed0、五折验证。其他历史 seed 保留，但不再继续训练。',
             '所有实验共享每张 GPU 最多两个用户进程的上限。', '',
             '| 数据集 | 模型 | 完成折数 | 验证 AUC | 验证 AP | 验证 ACC |',
             '|---|---|---:|---:|---:|---:|']
    records = []
    for dataset in ('OncologyScreen', 'DrugCombDB'):
        for mode in MODES:
            directory = root / dataset / 'seed0' / mode
            folds = [read(p) for p in sorted(directory.glob('fold*/metrics.json'))]
            complete = len(folds) == 5 and read(directory / 'progress.json').get('status') == 'complete'
            scores = {k: sum(f[k] for f in folds) / 5 for k in ('val_auc', 'val_aupr', 'val_acc')} if complete else None
            records.append(dict(dataset=dataset, mode=mode, seed=0, complete=complete,
                                completed_folds=len(folds), mean=scores))
            values = ['%.4f' % scores[k] for k in ('val_auc', 'val_aupr', 'val_acc')] if scores else ['待完成'] * 3
            lines.append('| %s | %s | %d/5 | %s |' % (dataset, mode, len(folds), ' | '.join(values)))
    lines += ['', '固定测试集不评估。仅一个训练 seed，不据此宣称跨 seed 稳定性。',
              '原 plan.json 保留历史三 seed 计划；当前执行范围以 resource_policy.json 为准。']
    (root / 'RESULTS_SINGLE_SEED.md').write_text('\n'.join(lines) + '\n')
    write(root / 'single_seed_summary.json', dict(seeds=[0], experiments=records))


def command(job, representation, auxiliary, plan):
    common = [sys.executable, '-B']
    if job['kind'] == 'representation':
        dataset = job['dataset']
        data = PROJECT / 'Data' / dataset
        role = 'dual_granularity_embeddings_v2.npz' if dataset == 'OncologyScreen' else 'dual_granularity_embeddings_kg_suite_20261008.npz'
        fact = 'dual_granularity_facts_pooling_20260928.npz' if dataset == 'OncologyScreen' else 'dual_granularity_facts_kg_suite_20261008.npz'
        return common + ['tools/run_kg_representation_cv5.py', '--dataset', dataset,
            '--mode', job['mode'], '--output', str(representation / dataset / 'seed0'),
            '--train-seed', '0', '--split-seed', '0', '--role-cache', str(data / role),
            '--fact-cache', str(data / fact), '--temperature', '.5',
            '--device', 'cuda:%d' % job['gpu'], '--epochs', str(plan['epochs']),
            '--patience', str(plan['patience'])]
    kind, weight = RECIPES[job['recipe']]
    return common + ['tools/run_kg_auxiliary_cv5.py', '--dataset', 'OncologyScreen',
        '--mode', job['mode'], '--output', str(auxiliary / job['recipe']),
        '--train-seed', '0', '--device', 'cuda:%d' % job['gpu'],
        '--auxiliary-ce-weight', '1.', '--contrastive-kind', kind,
        '--contrastive-weight', str(weight), '--temperature', '.5']


def main(args):
    os.chdir(PROJECT)
    representation = Path(args.representation).resolve()
    auxiliary = Path(args.auxiliary).resolve()
    locks = []
    for root in (representation, auxiliary):
        lock = (root / 'scheduler.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    # Fail before scheduling if original representation sources changed.
    import hashlib
    for source, sha in read(representation / 'source_hashes.json').items():
        if hashlib.sha256((PROJECT / source).read_bytes()).hexdigest() != sha:
            raise RuntimeError('frozen source changed: ' + source)
    plan = read(representation / 'plan.json')
    active = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            job = identify(int(p.name), representation, auxiliary)
            if job:
                if key(job) in active:
                    raise RuntimeError('duplicate worker')
                active[key(job)] = job
    # Auxiliary trials take the next freed slots; no additional seeds are queued.
    candidates = [dict(kind='auxiliary', recipe=recipe, seed=0, mode=mode,
                       directory=str(auxiliary / recipe / mode))
                  for recipe in RECIPES for mode in ('mean', 'single_query', 'task_query')]
    candidates += [dict(kind='representation', dataset=dataset, seed=0, mode=mode,
                        directory=str(representation / dataset / 'seed0' / mode))
                   for dataset in plan['datasets'] for mode in MODES]
    pending, finished, failed = [], [], []
    for job in candidates:
        if key(job) in active:
            continue
        if read(Path(job['directory']) / 'progress.json').get('status') == 'complete':
            finished.append(job)
        else:
            pending.append(job)
    children, streams = {}, {}
    write(representation / 'resource_policy.json', dict(seeds=[0], folds=5,
        max_user_processes_per_gpu=2, shared_with=str(auxiliary),
        historical_seeds_preserved=True, controller=str(Path(__file__).resolve())))
    representation_report(representation)
    from tools.run_kg_auxiliary_followup import summarize
    summarize(auxiliary)
    while active or pending:
        changed = False
        for k, job in list(active.items()):
            process = children.get(k)
            code = process.poll() if process else (None if identify(job['pid'], representation, auxiliary) else 0)
            if code is None:
                continue
            if k in streams:
                streams.pop(k).close()
            complete = read(Path(job['directory']) / 'progress.json').get('status') == 'complete'
            (finished if code == 0 and complete else failed).append(job)
            del active[k]
            changed = True
            print('DONE' if complete else 'FAILED', job, flush=True)
        occupied = gpu_user_counts()  # Query failure stops launches, never assumes GPUs free.
        while pending:
            gpu = eligible_gpu(active, occupied, args.gpus)
            if gpu is None:
                break
            job = pending.pop(0)
            job['gpu'] = gpu
            directory = Path(job['directory'])
            root = auxiliary if job['kind'] == 'auxiliary' else representation
            if directory.exists():
                # Trainer has no optimizer/RNG resume: preserve the interrupted attempt,
                # then restart this one-seed job with its original deterministic settings.
                archive = root / 'interrupted_attempts' / directory.relative_to(root)
                archive = archive.with_name(archive.name + '_' + str(time.time_ns()))
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(directory), str(archive))
                job['previous_attempt'] = str(archive)
            logfile = root / 'logs' / ('limited_%s_%s.log' % (key(job)[1], job['mode']))
            stream = logfile.open('a')
            process = subprocess.Popen(command(job, representation, auxiliary, plan), cwd=PROJECT,
                stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                         PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false'))
            job.update(pid=process.pid, log=str(logfile))
            k = key(job)
            active[k], children[k], streams[k] = job, process, stream
            print('START', job, flush=True)
        state = dict(status='running', pid=os.getpid(), seeds=[0], folds=5,
                     max_user_processes_per_gpu=2, active=list(active.values()),
                     pending=pending, completed_jobs=len(finished), failed=failed)
        write(representation / 'limited_queue_progress.json', state)
        if changed:
            representation_report(representation)
            summarize(auxiliary)
        for root, filename, kind in ((representation, 'scheduler_progress.json', 'representation'),
                                     (auxiliary, 'progress.json', 'auxiliary')):
            write(root / filename, dict(status='running' if any(j['kind'] == kind for j in active.values()) else 'queued',
                pid=os.getpid(), scheduling='Shared queue; max 2 user GPU processes; seed0 only',
                active=[j for j in active.values() if j['kind'] == kind],
                completed_jobs=sum(j['kind'] == kind for j in finished),
                pending_jobs=sum(j['kind'] == kind for j in pending),
                failed=[j for j in failed if j['kind'] == kind]))
        if active or pending:
            time.sleep(10)
    write(representation / 'limited_queue_progress.json', dict(status='failed' if failed else 'complete',
        seeds=[0], max_user_processes_per_gpu=2, completed_jobs=len(finished), failed=failed))
    for root, filename, kind in ((representation, 'scheduler_progress.json', 'representation'),
                                 (auxiliary, 'progress.json', 'auxiliary')):
        relevant = [j for j in failed if j['kind'] == kind]
        write(root / filename, dict(status='failed' if relevant else 'complete', seeds=[0],
            active=[], pending_jobs=0, failed=relevant,
            completed_jobs=sum(j['kind'] == kind for j in finished)))
    representation_report(representation)
    summarize(auxiliary)
    if failed:
        raise RuntimeError('Training jobs failed; see limited_queue_progress.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation', required=True)
    parser.add_argument('--auxiliary', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('unique nonnegative GPUs required')
    main(args)
