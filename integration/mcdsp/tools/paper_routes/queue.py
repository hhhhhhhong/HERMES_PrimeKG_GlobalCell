"""Persistent MUG / Proxy-Purifier / HG-Adapter controller.

Counts all GPU processes owned by this user, including unrelated queues.
No metric gate: every implemented route runs even if a previous route loses.
--proxy-now permits the user-authorized overlap of MUG and Proxy-Purifier.
--hg-now permits immediate independent HG-Adapter execution as requested.
"""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paper_routes.common import BASELINE, PROJECT, digest, snapshot_sources, write
from tools.run_kg_limited_queue import gpu_user_counts

DATASETS = ['OncologyScreen', 'DrugCombDB']
STAGES = ['MUG', 'ProxyPurifier', 'HGAdapter']
_input_digest_cache = {}
_auxiliary_keys = None


def verified_input_digest(filename):
    """Avoid rereading unchanged large arrays on every scheduler heartbeat."""
    path = Path(filename)
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if filename not in _input_digest_cache or _input_digest_cache[filename][0] != signature:
        _input_digest_cache[filename] = (signature, digest(path))
    return _input_digest_cache[filename][1]


def sidecar_jobs(root):
    """Small explicitly authorized diagnostics/pilots share this controller."""
    records = read(root / 'auxiliary_jobs.json').get('jobs', [])
    if _auxiliary_keys is not None:
        found = {j['key'] for j in records}
        if not set(_auxiliary_keys) <= found:
            raise ValueError('requested auxiliary keys absent from registry')
        records = [j for j in records if j['key'] in _auxiliary_keys]
    assert len({j['key'] for j in records}) == len(records), 'duplicate auxiliary job key'
    assert len({str(Path(j['directory']).resolve()) for j in records}) == len(records), 'duplicate auxiliary output'
    for job in records:
        assert job['module'] in ['train_hermes_dual_branch','train_hermes_knowledge_decay_oneil_cv5','train_hermes_knowledge_decay','train_hermes_zero_gated_hgnn_cv5','train_hermes_zero_gated_hgnn','train_hermes_global_cell','train_hermes_path_batch_cv5','train_hermes_path_batch','train_hermes_coder_singlepath','train_hermes_relation_scratch_cv5','train_hermes_relation_joint_cv5_v2','train_hermes_relation_joint_cv5','train_hermes_relation_residual_oneil','train_hermes_relation_residual','train_hermes_primekg_hhnn','train_hermes_atom_pair_3d','train_hermes_fragment_geometry','train_hermes_drug_init_2d3d','train_hermes_hg_structure','train_hermes_topk_joint','train_hermes_unimol_paper','train_hermes_unimol_structure','train_hermes_topk_increment','train_mechanism_hhnn_rl','train_hermes_query_mechanism_v2','train_hermes_query_mechanism','train_hermes_medkg_rl','train_higsyn_author','train_hermes_author_cv5','train_hermes_original_check','train_env_syn_original','train_backbone_knowledge_residual','train_no_rgcn_subgraph','prepare_scratch_subgraph_descriptions','prepare_scratch_annotations','train_subgraph_descriptions_scratch','prepare_subgraph_descriptions','train_subgraph_descriptions','train_conditional_subgraph','train_bio_kg_functional_cv5','train_bio_kg_functional','prepare_direct_annotations','train_direct_annotations','prepare_hierarchical_coarse_v2','train_hierarchical_coarse_v2','train_hierarchical_rsj_moe','train_evidence_memory','train_map_ce_dropout','train_map_synergy_genes','train_map_bio_text_moe','train_fixed_ensemble_rank','train_map_bio_ablation','train_map_path_diagnostics','train_map_bio_nograph_cross','train_map_synergy_transformer','train_map_synergy_cell'] or job['module'] in ['diagnose_proxy', 'tune_proxy', 'tune_mug', 'repair_pilot', 'train_mcdsp_repaired', 'train_brics', 'train_context_interaction', 'train_context_identity', 'train_full_context_i2moe', 'train_function_rgcn', 'train_rise_synergy', 'train_agent_evidence', 'train_agent_full', 'train_agent_speed', 'train_hermes', 'train_hermes_tune', 'train_hermes_ablation', 'train_hybrid_ideas', 'train_hermes_full_data', 'train_hermes_full_data_v2', 'train_moco_multisynergy', 'train_molecular_bio', 'train_molecular_bio_cv5', 'train_molecular_bio_author', 'train_molecular_bio_author_cv5', 'train_molecular_bio_adaptation', 'train_molecular_dropout_cv5', 'train_multitoken_mechanism', 'train_mechanism_diversification', 'train_drug_specific_cell', 'train_drug_specific_cell_oncology', 'train_mainpath_molecular', 'train_mainpath_molecular_cv5', 'evaluate_mainpath_molecular_cv5', 'train_mainpath_fusion_cv5', 'evaluate_mainpath_fusion_cv5', 'train_mainpath_moco_pilot', 'train_mainpath_moco_repair', 'train_mainpath_fp_moco_fusion', 'train_task_tcp_fusion', 'train_text_bio_risk_cv5', 'train_text_bio_only', 'prepare_graph_retrieval_resume', 'prepare_graph_retrieval_text', 'train_mainpath_graph_text', 'benchmark_llm_native', 'benchmark_llm_parallel', 'benchmark_llm_graph_speed', 'run_llm_graph_evidence', 'select_llm_graph_text', 'select_llm_graph_text_native', 'select_llm_graph_text_parallel', 'merge_llm_graph_shards', 'encode_llm_graph_text_parallel', 'select_llm_graph_text_fast', 'encode_llm_graph_text_fast', 'select_llm_graph_text_stream', 'encode_llm_graph_text', 'prepare_map_attributes', 'prepare_llm_text_tuning', 'train_llm_text_frozen', 'train_llm_text_adaptation', 'train_map_attention_rdrop', 'train_map_pair_cross_attention', 'train_map_pair_transformer3', 'train_map_pair_transformer', 'train_map_pair_interaction', 'train_map_mechanism_plateau_cv5', 'evaluate_map_mechanism_plateau_cv5', 'train_map_mechanism_optimization', 'train_map_mechanism_earlystop_cv5', 'evaluate_map_mechanism_earlystop_cv5', 'train_map_mechanism_cv5', 'evaluate_map_mechanism_cv5', 'train_map_attribute_repaired', 'train_llm_graph_ensemble_cv5', 'evaluate_llm_graph_ensemble_cv5', 'train_fixed_ensemble_kg_bio_cross', 'train_fixed_ensemble_paper_head', 'train_text_bio_transformer_single', 'train_text_bio_transformer_moe', 'train_text_bio_terminal_moe', 'train_map_attribute_scale', 'train_map_attribute_graph', 'train_mainpath_llm_graph_text', 'evaluate_fp_moco_fusion_cv5', 'evaluate_mainpath_moco_repair']
        resolved_directory=str(Path(job['directory']).resolve())
        # Explicit external storage keeps checkpoints writable when the historical result disk is full.
        assert resolved_directory.startswith(str(root.resolve())+'/tuning_') or (
            job.get('result_storage_root') and resolved_directory.startswith(str(Path(job['result_storage_root']).resolve())+'/'))
        assert job['key'].startswith('aux_')
    return [dict(j, stage='Auxiliary') for j in records]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def jobs(root, stage):
    if stage == 'MUG':
        return [dict(key='MUG_%s_%s' % (d, m), dataset=d, mode=m,
                     directory=str(root / stage / d / m), module='train_mcdsp')
                for d in DATASETS for m in ['M1', 'M2', 'M3']]
    if stage == 'ProxyPurifier':
        return [dict(key='teacher', directory=str(root / stage / 'teacher'), module='teacher')] + [
            dict(key='ProxyPurifier_' + d, dataset=d, directory=str(root / stage / d), module='train_proxy')
            for d in DATASETS]
    return [dict(key='HGAdapter_' + d, dataset=d, directory=str(root / stage / d), module='train_hg')
            for d in DATASETS]


def complete(job):
    directory = Path(job['directory'])
    if job.get('stage') == 'Auxiliary':
        version = directory / 'EXPERIMENT.json'
        if version.exists() and job['module'] in {'repair_pilot', 'train_mcdsp_repaired', 'train_brics', 'train_context_interaction', 'train_context_identity', 'train_full_context_i2moe', 'train_function_rgcn', 'train_rise_synergy', 'train_agent_evidence', 'train_agent_full', 'train_agent_speed', 'train_hermes', 'train_hermes_tune', 'train_hermes_ablation', 'train_hybrid_ideas', 'train_hermes_full_data', 'train_hermes_full_data_v2', 'train_moco_multisynergy', 'train_molecular_bio', 'train_molecular_bio_cv5', 'train_molecular_bio_author', 'train_molecular_bio_author_cv5', 'train_molecular_bio_adaptation', 'train_molecular_dropout_cv5', 'train_multitoken_mechanism', 'train_mechanism_diversification', 'train_drug_specific_cell', 'train_drug_specific_cell_oncology', 'train_mainpath_molecular', 'train_mainpath_molecular_cv5', 'evaluate_mainpath_molecular_cv5', 'train_mainpath_fusion_cv5', 'evaluate_mainpath_fusion_cv5', 'train_mainpath_moco_pilot', 'train_mainpath_moco_repair', 'train_mainpath_fp_moco_fusion', 'train_task_tcp_fusion', 'train_text_bio_risk_cv5', 'train_text_bio_only', 'prepare_graph_retrieval_resume', 'prepare_graph_retrieval_text', 'train_mainpath_graph_text', 'run_llm_graph_evidence', 'select_llm_graph_text', 'select_llm_graph_text_native', 'select_llm_graph_text_parallel', 'merge_llm_graph_shards', 'encode_llm_graph_text_parallel', 'select_llm_graph_text_fast', 'encode_llm_graph_text_fast', 'select_llm_graph_text_stream', 'encode_llm_graph_text', 'prepare_map_attributes', 'prepare_llm_text_tuning', 'train_llm_text_frozen', 'train_llm_text_adaptation', 'train_map_attention_rdrop', 'train_map_pair_cross_attention', 'train_map_pair_transformer3', 'train_map_pair_transformer', 'train_map_pair_interaction', 'train_map_mechanism_plateau_cv5', 'evaluate_map_mechanism_plateau_cv5', 'train_map_mechanism_optimization', 'train_map_mechanism_earlystop_cv5', 'evaluate_map_mechanism_earlystop_cv5', 'train_map_mechanism_cv5', 'evaluate_map_mechanism_cv5', 'train_map_attribute_repaired', 'train_llm_graph_ensemble_cv5', 'evaluate_llm_graph_ensemble_cv5', 'train_fixed_ensemble_kg_bio_cross', 'train_fixed_ensemble_paper_head', 'train_text_bio_transformer_single', 'train_text_bio_transformer_moe', 'train_text_bio_terminal_moe', 'train_map_attribute_scale', 'train_map_attribute_graph', 'train_mainpath_llm_graph_text', 'evaluate_fp_moco_fusion_cv5', 'evaluate_mainpath_moco_repair'}:
            # Refuse changed recipes/sources before skipping a completed job.
            # Historical unversioned jobs retain their original status; the
            # repaired runners independently forbid reusing those directories.
            import importlib
            module = importlib.import_module('tools.paper_routes.' + job['module'])
            args = module.build_parser().parse_args(['--output', str(directory),
                '--result-directory', str(directory), '--device', 'cpu',
                '--dataset', job['dataset']] + job.get('extra_args', []))
            recorded = read(version)
            if recorded['recipe'] != module.task_recipe(args) or recorded['source_sha256'] != module.task_sources():
                raise ValueError('completed auxiliary experiment version changed; choose a new output directory: ' + str(directory))
            if any(verified_input_digest(path) != sha for path, sha in recorded['inputs_sha256'].items()):
                raise ValueError('completed auxiliary experiment inputs changed; choose a new output directory: ' + str(directory))
        progress = read(directory / 'progress.json')
        if job['module'] in {'train_full_context_i2moe', 'train_function_rgcn'} and version.exists():
            expected_folds = recorded['recipe']['fold_limit'] or 5
            return progress.get('status') == 'complete' and progress.get('completed_folds') == expected_folds
        return progress.get('status') == 'complete'
    if job['key'] == 'teacher':
        meta = read(directory / 'COMPLETE.json')
        return meta.get('status') == 'complete' and meta.get('nodes', 0) > 10000
    p = read(directory / 'progress.json')
    required = ('completed_folds', 5) if job['module'] == 'train_mcdsp' else ('completed_fits', 25)
    return p.get('status') == 'complete' and p.get(required[0]) == required[1]


def pid_alive(pid, root, module):
    try:
        args = Path('/proc/%d/cmdline' % pid).read_bytes().decode().split('\0')
        return 'tools.paper_routes.' + module in args and str(root) in args
    except FileNotFoundError:
        return False


def gpu_memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(v) for i, v in csv.reader(rows.splitlines())}


def command(job, root, gpu):
    cmd = [sys.executable, '-u', '-m', 'tools.paper_routes.' + job['module'],
           '--output', str(root), '--device', 'cpu' if gpu is None else 'cuda:%d' % gpu]
    if 'dataset' in job:
        cmd += ['--dataset', job['dataset']]
    if 'mode' in job:
        cmd += ['--mode', job['mode']]
    if job.get('stage') == 'Auxiliary':
        cmd += ['--result-directory', job['directory']] + job.get('extra_args', [])
    return cmd


def scheduling_plan(root, stage, active, checks, proxy_now=False, hg_now=False):
    """Keep teacher dependencies while optionally scheduling Proxy during MUG."""
    stages = [stage] if stage in STAGES else []
    if proxy_now and stage == 'MUG' and not all(complete(j) for j in jobs(root, 'ProxyPurifier')):
        stages.append('ProxyPurifier')
    if hg_now and stage != 'HGAdapter' and not all(complete(j) for j in jobs(root, 'HGAdapter')):
        stages.append('HGAdapter')
    readiness, pending = {}, []
    for candidate in stages:
        check = {'MUG': 'mug', 'ProxyPurifier': 'proxy', 'HGAdapter': 'hg'}[candidate]
        ready = checks.get(check) is True
        if candidate == 'MUG':
            ready = ready and all(read(root / 'MUG' / d / 'pretrain/progress.json').get('status') == 'complete' for d in DATASETS)
        elif candidate == 'ProxyPurifier':
            ready = ready and checks.get('teacher') is True
        readiness[candidate] = ready
        candidates = [j for j in jobs(root, candidate) if not complete(j) and j['key'] not in active]
        if candidate == 'HGAdapter':
            # Place the full-size job first so a small job cannot consume its
            # only available high-memory GPU while another GPU can fit it.
            candidates.sort(key=lambda j: j['dataset'] != 'DrugCombDB')
        if candidate == 'ProxyPurifier' and not complete(jobs(root, candidate)[0]):
            candidates = [j for j in candidates if j['key'] == 'teacher']
        for job in candidates:
            job['stage'] = candidate
            pending.append(job)
    auxiliary = [j for j in sidecar_jobs(root) if not complete(j) and j['key'] not in active]
    if auxiliary:
        stages.append('Auxiliary')
        readiness['Auxiliary'] = True
        pending.extend(auxiliary)
    return stages, readiness, pending


def memory_requirement(job):
    if job.get('stage') == 'Auxiliary':
        return int(job.get('memory_mib', 7000))
    if job['key'] == 'teacher':
        return 23000
    if job['stage'] == 'HGAdapter' and job['dataset'] == 'DrugCombDB':
        return 40000
    return 7000


def report(root, state, runtime=None):
    runtime = root if runtime is None else runtime
    if state.get('auxiliary_only'):
        rows=[]
        lines=['# 已授权的独立单折试验', '',
               '统一队列；seed0/fold0；所有任务合计每卡最多两个用户GPU进程。', '',
               '| 任务 | 状态 | epoch | 最佳内部AUC |', '|---|---|---:|---:|']
        for job in sidecar_jobs(root):
            value=read(Path(job['directory'])/'progress.json')
            rows.append(dict(key=job['key'],directory=job['directory'],**value))
            lines.append('| %s | %s | %s | %s |' % (job['key'],value.get('status','pending'),
                         value.get('epoch',''),value.get('best_inner_auc','')))
        (runtime/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write(runtime/'auxiliary_summary.json',dict(state=state,jobs=rows))
        if any(j['module']=='train_rise_synergy' for j in sidecar_jobs(root)):
            from tools.paper_routes.train_rise_synergy import report as rise_report
            rise_report(root)
        return
    import numpy as np
    import pandas as pd
    rows = []
    execution = ('MUG、Proxy–Purifier、HG-Adapter 按用户最新授权独立并行执行。'
                 if state.get('hg_parallel_now') else
                 'MUG 与 Proxy–Purifier 同时执行；HG-Adapter 等待前两条完成。'
                 if state.get('proxy_parallel_to_mug') else '顺序 MUG → Proxy–Purifier → HG-Adapter。')
    lines = ['# 三路线实验状态', '',
             execution + ' 两个数据集；seed0、CV1 五折。',
             '每张 GPU 最多两个用户进程；不读取封存测试标签，不按前一条路线成绩跳过后续路线。', '',
             '当前状态：%s；阶段：%s。' % (state['status'], state.get('stage')), '',
             '下表仅汇总已完成的 fold，少于五折不能作为完整结果。', '',
             '| 路线 | 数据集 | 对照 | 完成折数 | 验证 AUC 均值 |',
             '|---|---|---|---:|---:|']
    for dataset in DATASETS:
        baseline = pd.read_csv(BASELINE / dataset / 'seed0/paper/fold_metrics.csv')
        auc = float(baseline.val_auc.mean())
        lines.append('| M0 原 MCDSP | %s | baseline | 5 | %.6f |' % (dataset, auc))
        rows.append(dict(route='M0', dataset=dataset, variant='paper', completed_folds=5, auc_mean=auc))
        specs = [('MUG', ['M1', 'M2', 'M3']),
                 ('ProxyPurifier', ['raw', 'random_lift', 'proxy_only', 'full', 'shuffled_teacher']),
                 ('HGAdapter', ['none', 'hom', 'het', 'full', 'no_propagation'])]
        for route, variants in specs:
            for variant in variants:
                found = [read(p) for p in (root / route / dataset / variant).glob('fold*/metrics.json')]
                values = [r['val_auc'] for r in found]
                val = float(np.mean(values)) if values else None
                lines.append('| %s | %s | %s | %d | %s |' % (route, dataset, variant,
                    len(values), '尚未完成' if val is None else '%.6f' % val))
                rows.append(dict(route=route, dataset=dataset, variant=variant,
                    completed_folds=len(values), auc_mean=val, delta_to_original=None if val is None or len(values) != 5 else val - auc))
    lines += ['', 'Proxy/HG 使用冻结原模型的任务适配协议，应优先对比各自 raw/none、随机 lift、打乱教师和组件消融；',
              '与端到端 MCDSP 的差值不是孤立组件的因果增益。预训练边预测成绩不等于协同预测成绩。', '',
              '已完成工作的真实日志、梯度、预测和来源散列均保存在同一结果目录。']
    pd.DataFrame(rows).to_csv(runtime / 'summary.csv', index=False)
    (runtime / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    # Keep the newly authorized full-model experiment visible while its folds run.
    if any(j['module'] == 'train_full_context_i2moe' and '/tuning_20261011_full_context_i2moe/' in j['directory']
           for j in sidecar_jobs(root)):
        from tools.report_full_context_i2moe import report as context_report
        context_report(root / 'tuning_20261011_full_context_i2moe')
    if any(j['module'] == 'train_function_rgcn' and '/tuning_20261011_function_rgcn/' in j['directory']
           for j in sidecar_jobs(root)):
        from tools.report_function_rgcn import report as function_report
        function_report(root / 'tuning_20261011_function_rgcn')



def process_limit_overrides(values):
    result={}
    for value in values or []:
        gpu,limit=map(int,value.split('='))
        if gpu<0 or limit<1 or gpu in result:raise ValueError('Expected unique GPU=positive_limit')
        result[gpu]=limit
    return result


def gpu_slot_available(gpu,occupied,active,overrides):
    pids=set(occupied.get(gpu,set())) | {j['pid'] for j in active.values() if j['gpu']==gpu}
    return len(pids)<overrides.get(gpu,2)


def job_gpu_slot_available(job, gpu, occupied, active, overrides):
    """An optional job ceiling may tighten, never enlarge, the global limit.

    This keeps a temporary exception restricted to the approved jobs while
    other already-queued jobs continue to observe their original limit.
    """
    limit = overrides.get(gpu, 2)
    ceiling = job.get('gpu_process_limits', {}).get(str(gpu))
    if ceiling is not None:
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
            raise ValueError('job GPU process ceiling must be a positive integer')
        limit = min(limit, ceiling)
    return gpu_slot_available(gpu, occupied, active, {gpu: limit})


def run(args):
    global _auxiliary_keys
    _auxiliary_keys = getattr(args, 'auxiliary_keys', None)
    limits=process_limit_overrides(getattr(args,'gpu_process_limit',None))
    selected_stages = [] if getattr(args, 'auxiliary_only', False) else STAGES
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Optional writable runtime keeps logs/state off a full historical disk.
    # Registry, result paths, shared lock and worker --output remain unchanged.
    runtime = Path(args.runtime_directory).resolve() if getattr(args, 'runtime_directory', None) else root
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / 'processes').mkdir(exist_ok=True)
    os.chdir(PROJECT)
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(runtime / 'resource_policy.json', dict(seed=[0], folds=1 if getattr(args, 'auxiliary_only', False) else 5, max_user_processes_per_gpu=2, gpu_process_limit_overrides={str(k):v for k,v in limits.items()},
        auxiliary_only=getattr(args, 'auxiliary_only', False), selected_auxiliary_keys=_auxiliary_keys,
        process_limit_exempt_auxiliary_keys=[j['key'] for j in sidecar_jobs(root) if j.get('ignore_process_limit',False)],
        stage_order=STAGES, no_metric_gate=True, old_results_preserved=True,
        proxy_parallel_to_mug=args.proxy_now, teacher_required_before_proxy_training=True,
        hg_parallel_now=args.hg_now, hg_backbone='independent original baseline; not MUG/Proxy outputs',
        test_policy='sealed', gpus=args.gpus, controller=str(Path(__file__).resolve())))
    # Preserve the original launch snapshot when adopting existing workers.
    snapshot = root if not (root / 'pipeline_source_hashes.json').exists() else root / 'controller_restarts' / datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S_%f')
    no_source_checks = getattr(args, 'no_source_checks', False)
    if not no_source_checks:
        source_hashes = snapshot_sources(snapshot)
        write(snapshot / 'pipeline_source_hashes.json', source_hashes)
        write(root / 'current_controller_source.json', dict(snapshot=str(snapshot), source_hashes=source_hashes))
    originals = ['Model/rgcn_model.py', 'Model/Multimodal_OncologyScreen.py',
                 'Model/graph_semantic_i2moe.py', 'Model/modality_training_recipe.py',
                 'Utils/graph_constrct.py', 'Utils/feature_utils.py']
    protected = {} if no_source_checks else {s: digest(PROJECT / s) for s in originals}
    if not no_source_checks:
        write(root / 'protected_original_hashes.json', protected)
    active, processes, streams = {}, {}, {}
    for group in [jobs(root, s) for s in selected_stages] + [sidecar_jobs(root)]:
        for job in group:
            previous = read(runtime / 'processes' / (job['key'] + '.json'))
            if previous and pid_alive(previous['pid'], root, job['module']):
                active[job['key']] = previous
    completed = set()
    (runtime / 'logs').mkdir(exist_ok=True)
    last_report = 0
    while True:
        # Re-adopt a live worker if a dynamically edited manifest reintroduces
        # it. A controller restart/registry edit must never launch it twice.
        for job in sidecar_jobs(root):
            if job['key'] not in active:
                previous = read(runtime / 'processes' / (job['key'] + '.json'))
                if previous and pid_alive(previous['pid'], root, job['module']):
                    active[job['key']] = previous
        for k, job in list(active.items()):
            child = processes.get(k)
            running = child.poll() is None if child else pid_alive(job['pid'], root, job['module'])
            if running:
                continue
            if k in streams:
                streams.pop(k).close()
            if not complete(job):
                write(runtime / 'pipeline_progress.json', dict(status='failed', pid=os.getpid(),
                    stage=job['stage'], failure=job, progress=read(Path(job['directory']) / 'progress.json'),
                    active=[j for kk, j in active.items() if kk != k],
                    reason='technical failure; not silently skipping or changing the method'))
                raise RuntimeError('worker failed: ' + k + '; see ' + job['log'])
            print('COMPLETE', k, flush=True)
            completed.add(k)
            del active[k]
        for source, sha in protected.items():
            if digest(PROJECT / source) != sha:
                raise RuntimeError('protected original model source changed: ' + source)
        stage = next((s for s in selected_stages if not all(complete(j) for j in jobs(root, s))), None)
        if stage is None and any(not complete(j) for j in sidecar_jobs(root)):
            stage = 'Auxiliary'
        if stage is None:
            state = dict(status='complete', pid=os.getpid(), stage=None,
                         completed_jobs=sum(complete(j) for s in selected_stages for j in jobs(root, s)),
                         completed_auxiliary_jobs=sum(complete(j) for j in sidecar_jobs(root)),
                         auxiliary_only=getattr(args, 'auxiliary_only', False),
                         completion_time_utc=datetime.datetime.utcnow().isoformat(), seed=0, folds=5,
                         proxy_parallel_to_mug=args.proxy_now)
            state['hg_parallel_now'] = args.hg_now
            write(runtime / 'pipeline_progress.json', state)
            report(root, state, runtime=runtime)
            if not no_source_checks:
                write(root / 'protected_original_verification.json', dict(passed=True, after=protected))
            return
        checks = read(root / 'preflight_checks.json')
        eligible_stages, readiness, pending = scheduling_plan(root, stage, active, checks, args.proxy_now, args.hg_now)
        if any(readiness.values()):
            # A failed NVML query aborts scheduling; never assumes GPUs are free.
            occupied, free = gpu_user_counts(), gpu_memory()
            allowed = [g for g in args.gpus if not (g == args.qa_gpu and not all(checks.get(c) for c in ['mug', 'proxy', 'hg', 'teacher']))]
            for job in pending:
                if not readiness[job['stage']]:
                    continue
                if any(read(Path(path)).get('status') != 'complete' for path in job.get('requires', [])):
                    continue
                need = memory_requirement(job)
                cpu_only = job.get('cpu_only', False)
                if cpu_only:
                    # Only this verified graph-only stage may bypass GPU
                    # slots. Hide CUDA entirely in its child environment.
                    assert job['module'] == 'prepare_graph_retrieval_resume'
                    assert job['extra_args'] == ['--stage', 'retrieve', '--device', 'cpu']
                    gpu = None
                else:
                    gpu = next((g for g in allowed if g in job.get('gpus', allowed) and
                        # Explicitly authorized per-job exception; memory placement remains enforced.
                        (job.get('ignore_process_limit',False) or job_gpu_slot_available(job,g,occupied,active,limits)) and free[g] >= need), None)
                    if gpu is None:
                        continue
                cmd = command(job, root, gpu)
                logfile = runtime / 'logs' / (job['key'] + '.log')
                stream = logfile.open('a')
                child_env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                               PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
                if cpu_only:child_env['CUDA_VISIBLE_DEVICES']=''
                process = subprocess.Popen(cmd, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT,
                    env=child_env, start_new_session=True)
                job.update(gpu=gpu, pid=process.pid, command=cmd, log=str(logfile),
                           started_utc=datetime.datetime.utcnow().isoformat())
                active[job['key']], processes[job['key']], streams[job['key']] = job, process, stream
                if not cpu_only:free[gpu] -= need
                write(runtime / 'processes' / (job['key'] + '.json'), job)
                print('START', job, flush=True)
        state = dict(status='running' if active or any(readiness.values()) else 'waiting_preflight', pid=os.getpid(), stage=stage,
            auxiliary_only=getattr(args, 'auxiliary_only', False),
            eligible_stages=eligible_stages, readiness=readiness, proxy_parallel_to_mug=args.proxy_now,
            hg_parallel_now=args.hg_now,
            active=list(active.values()), pending=[j for j in pending if j['key'] not in active],
            completed_jobs=sum(complete(j) for s in STAGES for j in jobs(root, s)),
            max_user_processes_per_gpu=2, gpu_process_limit_overrides={str(k):v for k,v in limits.items()}, seed=0, folds=1 if getattr(args, 'auxiliary_only', False) else 5, checks=checks,
            heartbeat_utc=datetime.datetime.utcnow().isoformat())
        write(runtime / 'pipeline_progress.json', state)
        if time.monotonic() - last_report >= 60:
            report(root, state, runtime=runtime)
            last_report = time.monotonic()
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-directory', help='optional writable logs, processes and controller state directory; shared registry/lock stay at --output')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 2, 3])
    parser.add_argument('--gpu-process-limit',action='append',metavar='GPU=LIMIT',
                        help='explicit per-GPU user process override; other GPUs remain limited to2')
    parser.add_argument('--qa-gpu', type=int, default=3,
                        help='reserve this GPU until all real preflight checks pass')
    parser.add_argument('--proxy-now', action='store_true',
                        help='start Proxy teacher/training during unfinished MUG, as explicitly authorized')
    parser.add_argument('--hg-now', action='store_true',
                        help='start independent HG adapters during unfinished routes, as explicitly authorized')
    parser.add_argument('--auxiliary-only', action='store_true',
                        help='run only explicitly selected auxiliary jobs; never resume historical routes')
    parser.add_argument('--no-source-checks', action='store_true',
                        help='user-requested direct training without controller source hash checks/snapshots')
    parser.add_argument('--auxiliary-keys', nargs='+',
                        help='limit auxiliary registry entries to these exact authorized job keys')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        root = Path(args.output)
        path = (Path(args.runtime_directory) if args.runtime_directory else root) / 'pipeline_progress.json'
        previous = read(path)
        if previous.get('status') != 'failed':
            write(path, dict(status='failed', pid=os.getpid(), error=repr(error),
                previous_state=previous, reason='controller failed; see pipeline.log'))
        raise
