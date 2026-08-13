"""Build comparison tables for the paper from baseline results.
Outputs per-iteration accuracy / compression (tab:perf style) and overall
accuracy / coverage / compression, for Abilene and Geant2009.

By default it reads the committed analysis files in ./results/; pass explicit
paths to inspect freshly re-run experiments.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABILE = os.path.join(HERE, 'results', 'baselines_abilene_20260813',
                     'analysis_20260813_135019.json')
GEANT = os.path.join(HERE, 'results', 'baselines_geant2009_20260813',
                     'analysis_20260813_125814.json')

MODELS = ['LVQ1', 'GLVQ', 'GLVQ_sklearn', 'LRVQ', 'ProtoNet']


def per_iteration_table(fn, label):
    d = json.load(open(fn))
    res = d['results']
    print('\n===== %s (per-iteration) =====' % label)
    print('%-13s %-30s | %s' % ('model', 'metric', ' '.join('it.%d' % i for i in range(d['params']['n_iterations']))))
    for name in MODELS:
        rn = res[name]
        for metric in ['accuracy', 'compression_ratio']:
            it_acc = []
            for it in range(d['params']['n_iterations']):
                vals = [r['metrics'][metric] for r in rn if r['iteration_id'] == it]
                it_acc.append(float(np.mean(vals)))
            if metric == 'accuracy':
                print('%-13s %-30s | %s' % (name, 'accuracy', ' '.join('%.4f' % v for v in it_acc)))
            else:
                print('%-13s %-30s | %s' % (name, 'compression', ' '.join('%.5f' % v for v in it_acc)))
    # overall
    print('---- overall (mean over agents & iterations) ----')
    for i, name in enumerate(MODELS):
        ov = d['aggregated'][i]['overall']
        print('%-13s acc=%.4f (std %.4f)  cov=%.4f  comp=%.5f  n_reprs=%.1f'
              % (name, ov['accuracy']['mean'], ov['accuracy']['std'],
                 ov['coverage']['mean'], ov['compression_ratio']['mean'], np.mean([r['metrics']['num_reprs'] for r in res[name]])))


if __name__ == '__main__':
    abile = sys.argv[1] if len(sys.argv) > 1 else ABILE
    geant = sys.argv[2] if len(sys.argv) > 2 else GEANT
    per_iteration_table(abile, 'Abilene (20 agents, it.0..it.3)')
    per_iteration_table(geant, 'Geant2009 (78 agents, it.0..it.5)')