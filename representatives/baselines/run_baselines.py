"""Full Abilene baseline run: 20 agents x 4 iterations (0..3) x 5 models.
Supported by resume/checkpoint (skips already-done models). Result JSON is
compatible with the paper's tab:perf tables and summarize_baselines.py.

Results are written to ./results/baselines_<data_train_path_name>_*/ directory next to this script.
"""
import sys, os, json, time, argparse
from json import JSONEncoder
import numpy as np
from pathlib import Path
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # representatives/ (data_loader.py)

from baselines import LVQ1, GLVQ, SklearnGLVQ, LRVQ, PrototypicalNet
from data_loader import DataLoader, load_maroh_data

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def std_fit_predict(model, Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model.fit((Xtr - mu) / sd, ytr)
    pred, conf, is_conf, n = model.predict((Xte - mu) / sd)
    return pred, conf, is_conf, n


def load_state(dataset_name, n_agents, n_iterations, coverage_target, models_names, checkpoint):
    if os.path.exists(checkpoint):
        with open(checkpoint) as f:
            return json.load(f)
    return {'params': dict(mode='full', classifier='baselines', dataset=dataset_name,
                           n_agents=n_agents, n_iterations=n_iterations,
                           coverage_target=coverage_target, models=models_names),
            'results': {}, 'done': []}


def save_state(state, out_dir, checkpoint):
    os.makedirs(out_dir, exist_ok=True)
    with open(checkpoint, 'w') as f:
        json.dump(state, f, indent=1, cls=NumpyArrayEncoder)


def main(args):
    config_filepath = args.input
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    dataset_name = Path(config['data_train']).name
    target_coverage = config['models'].get('LRVQ', {}).get('target_coverage', 0.56)
    cur_time = datetime.datetime.now()
    cur_time_str = cur_time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, 'results', f'baselines_{dataset_name}_{cur_time_str}')
    checkpoint = os.path.join(out_dir, 'checkpoint.json')

    models_names = ['LVQ1', 'GLVQ', 'GLVQ_sklearn', 'LRVQ', 'ProtoNet']
    models_args_default = {
        'LVQ1':         dict(m=8, epochs=25, lr=0.05),
        'GLVQ':         dict(m=8, epochs=30, lr=0.01),
        'GLVQ_sklearn': dict(m=8, max_iter=300, gtol=1e-5),
        'LRVQ':         dict(m=8, epochs=25, lr=0.05, target_coverage=0.56),
        'ProtoNet':     dict(dim_out=64, epochs=100, lr=0.005, batch=2048,
                                                seed=0, n_support=100),
    }
    models_classes = {
        'LVQ1':         LVQ1,
        'GLVQ':         GLVQ,
        'GLVQ_sklearn': SklearnGLVQ,
        'LRVQ':         LRVQ,
        'ProtoNet':     PrototypicalNet,
    }

    models = {}
    for model_name in models_names:
        params = config['models'].get(model_name, models_args_default[model_name])
        model_class = models_classes[model_name]
        models[model_name] = lambda cls=model_class, kwargs=params: cls(**kwargs)

    os.makedirs(out_dir, exist_ok=True)
    traind = load_maroh_data(config['data_train'], keep_episode=True)
    testd = load_maroh_data(config['data_test'], keep_episode=True)

    all_agents = sorted(np.unique(traind['agent']))
    all_iterations = sorted(set(
        it for a in all_agents
        for it in np.unique(traind[traind['agent'] == a]['iteration'])
    ))

    n_agents = len(all_agents)
    n_iterations = len(all_iterations)
    state = load_state(dataset_name, n_agents, n_iterations, target_coverage, models_names, checkpoint)
    remaining = [name for name in models_names if name not in state['done']]
    print('resume: done=%s remaining=%s' % (state['done'], remaining), flush=True)

    for name in remaining:
        make = models[name]
        t_cfg = time.time()
        results = []
        for a in all_agents:
            for it in all_iterations:
                Xtr, Xte, ytr, yte = DataLoader.prepare_training_data(traind, a, it, data_test=testd)
                model = make()
                t0 = time.time()
                if name == 'ProtoNet':
                    pred, conf, is_conf, n = std_fit_predict(model, Xtr, ytr, Xte, yte)
                else:
                    model.fit(Xtr, ytr)
                    pred, conf, is_conf, n = model.predict(Xte)
                fit_t = time.time() - t0
                is_conf = is_conf.astype(bool)
                total = len(yte)
                acc = float(np.mean(pred[is_conf] == yte[is_conf])) if is_conf.any() else 0.0
                cov = float(is_conf.mean())
                comp = n / len(Xtr)
                results.append(dict(
                    agent_id=a, iteration_id=it,
                    metrics=dict(accuracy=acc, coverage=cov, compression_ratio=comp,
                                 num_reprs=int(n), total_points=total,
                                 correct_classifications=int(is_conf.sum() * acc),
                                 confident_classifications=int(is_conf.sum())),
                    build_time=fit_t, test_time=0.0))
                print('[%s] agent=%2d it=%d acc=%.4f cov=%.3f comp=%.4f (%5.1fs)'
                      % (name, a, it, acc, cov, comp, fit_t), flush=True)
        s = time.time() - t_cfg
        print('=== %s done in %d:%02d' % (name, int(s // 60), int(s % 60)), flush=True)
        state['results'][name] = results
        state['done'].append(name)
        save_state(state, out_dir, checkpoint)

    def agg_stat(vals):
        vals = np.asarray(vals, dtype=float)
        return dict(mean=float(vals.mean()), median=float(np.median(vals)),
                    std=float(vals.std()), min=float(vals.min()), max=float(vals.max()))

    aggregated = []
    for name in models:
        res = state['results'][name]
        by_agent, by_iteration = {}, {}
        for a in all_agents:
            ks = [r for r in res if r['agent_id'] == a]
            by_agent[str(a)] = {m: agg_stat([r['metrics'][m] for r in ks])
                                for m in ['accuracy', 'coverage', 'compression_ratio', 'num_reprs']}
        for it in all_iterations:
            ks = [r for r in res if r['iteration_id'] == it]
            by_iteration[str(it)] = {m: agg_stat([r['metrics'][m] for r in ks])
                                     for m in ['accuracy', 'coverage', 'compression_ratio', 'num_reprs']}
        overall = {m: agg_stat([r['metrics'][m] for r in res])
                   for m in ['accuracy', 'coverage', 'compression_ratio', 'num_reprs']}
        aggregated.append(dict(overall=overall, by_agent=by_agent, by_iteration=by_iteration))

    payload = dict(
        params=state['params'],
        results=state['results'],
        aggregated=aggregated)
    fp = os.path.join(out_dir, 'analysis_%s.json' % cur_time_str)
    with open(fp, 'w') as f:
        json.dump(payload, f, indent=1, cls=NumpyArrayEncoder)
    print('saved', fp)
    for i, name in enumerate(models_names):
        ov = aggregated[i]['overall']
        print('%-13s acc=%.4f cov=%.3f comp=%.4f' % (name, ov['accuracy']['mean'],
                                                     ov['coverage']['mean'],
                                                     ov['compression_ratio']['mean']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', help='Path to json file with experiment configuration')
    args = parser.parse_args()
    main(args)
