"""Full Geant2009 baseline run: 78 agents x 6 iterations (0..5) x 5 models.
Supported by resume/checkpoint (skips already-done models). Result JSON is
compatible with the paper's tab:perf_geant tables and summarize_baselines.py.

Data layout expected under the --data directory:
    train_geant2009/  test_geant2009/
Results are written to ./results/<RUN_ID>/ next to this script.
"""
import sys, os, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # representatives/ (data_loader.py)

from baselines import LVQ1, GLVQ, SklearnGLVQ, LRVQ, PrototypicalNet
from data_loader import DataLoader, load_maroh_data

N_AGENTS = 78
N_ITERS = 6            # geant2009: iterations 0..5
TARGET_COV = 0.56

RUN_ID = 'baselines_geant2009_20260813'
OUT_DIR = os.path.join(HERE, 'results', RUN_ID)
CHECKPOINT = os.path.join(OUT_DIR, 'checkpoint.json')

MODELS = {
    'LVQ1':         lambda: LVQ1(m=8, epochs=25, lr=0.05),
    'GLVQ':         lambda: GLVQ(m=8, epochs=30, lr=0.01),
    'GLVQ_sklearn': lambda: SklearnGLVQ(m=8, max_iter=300, gtol=1e-5),
    'LRVQ':         lambda: LRVQ(m=8, epochs=25, lr=0.05, target_coverage=TARGET_COV),
    'ProtoNet':     lambda: PrototypicalNet(dim_out=64, epochs=100, lr=0.005, batch=2048,
                                            seed=0, n_support=100),
}


def std_fit_predict(model, Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    model.fit((Xtr - mu) / sd, ytr)
    pred, conf, is_conf, n = model.predict((Xte - mu) / sd)
    return pred, conf, is_conf, n


def load_state():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {'params': dict(mode='full', classifier='baselines', dataset='geant2009',
                           n_agents=N_AGENTS, n_iterations=N_ITERS,
                           coverage_target=TARGET_COV, models=list(MODELS.keys())),
            'results': {}, 'done': []}


def save_state(state):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CHECKPOINT, 'w') as f:
        json.dump(state, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('BASELINES_DATA', 'data'),
                    help='directory containing train_geant2009/ and test_geant2009/')
    args = ap.parse_args()
    base = os.path.abspath(args.data)
    for sub in ['train_geant2009', 'test_geant2009']:
        if not os.path.isdir(os.path.join(base, sub)):
            sys.exit('missing data directory: %s' % os.path.join(base, sub))

    os.makedirs(OUT_DIR, exist_ok=True)
    state = load_state()
    traind = load_maroh_data(os.path.join(base, 'train_geant2009'), keep_episode=True)
    testd = load_maroh_data(os.path.join(base, 'test_geant2009'), keep_episode=True)

    remaining = [name for name in MODELS if name not in state['done']]
    print('resume: done=%s remaining=%s' % (state['done'], remaining), flush=True)

    for name in remaining:
        make = MODELS[name]
        t_cfg = time.time()
        results = []
        for a in range(N_AGENTS):
            for it in range(N_ITERS):
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
                print('[%s] agent=%3d it=%d acc=%.4f cov=%.3f comp=%.4f (%5.1fs)'
                      % (name, a, it, acc, cov, comp, fit_t), flush=True)
        s = time.time() - t_cfg
        print('=== %s done in %d:%02d' % (name, int(s // 60), int(s % 60)), flush=True)
        state['results'][name] = results
        state['done'].append(name)
        save_state(state)

    def agg_stat(vals):
        vals = np.asarray(vals, dtype=float)
        return dict(mean=float(vals.mean()), median=float(np.median(vals)),
                    std=float(vals.std()), min=float(vals.min()), max=float(vals.max()))

    aggregated = []
    for name in MODELS:
        res = state['results'][name]
        by_agent, by_iteration = {}, {}
        for a in range(N_AGENTS):
            ks = [r for r in res if r['agent_id'] == a]
            by_agent[str(a)] = {m: agg_stat([r['metrics'][m] for r in ks])
                                for m in ['accuracy', 'coverage', 'compression_ratio', 'num_reprs']}
        for it in range(N_ITERS):
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
    fp = os.path.join(OUT_DIR, 'analysis_%s.json' % time.strftime('%Y%m%d_%H%M%S'))
    with open(fp, 'w') as f:
        json.dump(payload, f, indent=1)
    print('saved', fp)
    for i, name in enumerate(MODELS):
        ov = aggregated[i]['overall']
        print('%-13s acc=%.4f cov=%.3f comp=%.4f' % (name, ov['accuracy']['mean'],
                                                     ov['coverage']['mean'],
                                                     ov['compression_ratio']['mean']))


if __name__ == '__main__':
    main()