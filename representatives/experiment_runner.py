import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any, Callable, Union
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass
from sklearn.neighbors import BallTree, KDTree
import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import stats
import os
import re
import seaborn as sns
import json
import yaml
from itertools import product
from classifiers import ClassifierFactory, MetricSpace
from data_loader import DataLoader
from types import SimpleNamespace

@dataclass
class AgentIterationResult:
    """Result for a specific agent and iteration"""
    agent_id: int
    iteration_id: int
    metrics: Any  # EvaluationMetrics object
    representatives_count: int
    train_data_size: int
    test_data_size: int
    build_time: float
    test_time: float
    class_distribution: Dict[int, int]  # Class distribution in representatives

@dataclass
class AnalysisResults:
    """Full analysis results"""
    all_results: List[AgentIterationResult]
    aggregated_metrics: Dict[str, Dict[str, float]]  # Aggregated metrics by agents and iterations
    parameters: Dict[str, Any]  # Analysis parameters
    timestamp: str  # Analysis timestamp

def analyze_agent_iteration(
    data: np.ndarray,
    data_test: np.ndarray,
    agent_id: int,
    iteration_id: int,
    args: dict
) -> Tuple[AgentIterationResult, Dict[str, np.ndarray]]:
    """
    Analysis for a specific agent and iteration

    Args:
        data: Loaded data
        agent_id: Agent ID
        iteration_id: Iteration ID
        args: Run arguments

    Returns:
        AgentIterationResult; dictionary with np-arrays pred, confidences,
        confident, covered; trained classifier; or (None, None, None) if no data
    """
    try:
        if args.verbose:
            print(f"Analysis: Agent {agent_id}, Iteration {iteration_id}")
            print("-" * 40)

        train_ratio = None
        if data_test is None:
            train_ratio = args.train_ratio

        # Prepare data
        X_train, X_test, y_train, y_test = DataLoader.prepare_training_data(
            data, agent_id, iteration_id, train_ratio=train_ratio,
            data_test=data_test,
            random_seed=args.seed
        )

        if args.max_test_points and args.max_test_points < len(X_test):
            X_test = X_test[:args.max_test_points]
            y_test = y_test[:args.max_test_points]
            if args.verbose:
                print(f"   Test points limited to: {len(X_test)}")

        if len(X_train) == 0:
            if args.verbose:
                print(f"  No training data")
            return None

        if len(X_test) == 0:
            if args.verbose:
                print(f"  No test data")
            return None

        if args.verbose:
            print("Train data classes:")
            y_train_classes = np.unique(y_train)
            if len(y_train_classes) > 1:
                for label in y_train_classes:
                    class_n = np.sum(y_train == label)
                    fraction = class_n / len(y_train) if len(y_train) > 0 else 0
                    print(f"class {label}: {class_n} / {len(y_train)} points ({fraction})")
            else:
                print(f"!!! all {len(y_train)} points belong to class {y_train_classes[0]}")

            print("Test data classes:")
            y_test_classes = np.unique(y_test)
            if len(y_test_classes) > 1:
                for label in y_test_classes:
                    class_n = np.sum(y_test == label)
                    fraction = class_n / len(y_test) if len(y_test) > 0 else 0
                    print(f"class {label}: {class_n} / {len(y_test)} points ({fraction})")
            else:
                print(f"!!! all {len(y_test)} points belong to class {y_test_classes[0]}")

        # Initialize metric space
        metric_space = MetricSpace(metric=args.metric)

        # Build and train classifier
        if args.verbose:
            print(f"\n2. Building and training classifier {args.classifier}...")
        start_time = time.time()
        clf = ClassifierFactory.create(args.classifier,
                                       metric = metric_space,
                                       agent = agent_id,
                                       iteration = iteration_id,
                                       config = args)
        if hasattr(args, "reprs_path") and args.classifier == "epsilon_net":
            clf.load_representatives(os.path.join(args.reprs_path, f"reprs_a{agent_id:02}_i{iteration_id}.npz"))
        else:
            clf.fit(X_train, y_train)
        build_time = time.time() - start_time

        # Analyze class distribution in representatives
        class_distribution = defaultdict(int)
        to_update = False
        if args.classifier == 'epsilon_net':
            for rep in clf.representatives:
                class_distribution[rep.label] += 1
            if hasattr(args, 'update'):
                to_update = args.update
                if to_update:
                    update_buffer_length = args.update_buffer_length
                    max_updates = args.max_updates if hasattr(args, 'max_updates') else None

        # Test classifier
        if args.verbose:
            print(f"\n3. Testing on {len(X_test)} points...")
        start_time = time.time()
        if not to_update:
            metrics, preds_np = clf.evaluate(X_test, y_test)
            update_durations = []
            # update_duration = 0
            # n_updates = 0
        else:
            metrics, preds_np, update_durations = clf.evaluate(X_test, y_test,
                update_buffer_length=update_buffer_length, max_updates=max_updates)
            # update_duration = np.mean(update_durations)
            # n_updates = len(update_durations)
        test_time = time.time() - start_time

        if args.verbose:
            print(f"  Data: train={len(X_train)}, test={len(X_test)}")
            print(f"  Representatives: {clf.n_representatives}")
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            print(f"  Coverage: {metrics.coverage:.4f}")
            print(f"  Build time: {build_time:.3f}s")
            print(f"  Test time: {test_time:.3f}s")

        return AgentIterationResult(
            agent_id=agent_id,
            iteration_id=iteration_id,
            metrics=metrics,
            representatives_count=clf.n_representatives,
            train_data_size=len(X_train),
            test_data_size=len(X_test),
            build_time=build_time,
            test_time=test_time,
            class_distribution=dict(class_distribution)
        ), preds_np, clf, update_durations

    except Exception as e:
        raise e # DEBUG
        if args.verbose:
            print(f"  Error: {e}")
        return None, None, None


def aggregate_results(results):
    """Aggregate results by agents and iterations"""
    # Statistics
    def stats(vals):
        funcs = [('mean', np.mean), ('median', np.median), ('std', np.std),
                ('min', np.min), ('max', np.max)]
        return {name: float(f(vals)) for name, f in funcs}

    # Grouping
    by_agent, by_iter = defaultdict(list), defaultdict(list)
    for r in results:
        by_agent[r.agent_id].append(r)
        by_iter[r.iteration_id].append(r)

    # Collect metrics for group
    def collect(group):
        return {
            'accuracy': stats([r.metrics.accuracy for r in group]),
            'coverage': stats([r.metrics.coverage for r in group]),
            'compression': stats([r.representatives_count / r.train_data_size for r in group]),
            'build_time': stats([r.build_time for r in group]),
            'test_time': stats([r.test_time for r in group]),
            'n_reprs': stats([r.representatives_count for r in group]),
            'count': len(group)
        }

    return {
        'overall': collect(results),
        'by_agent': {k: collect(v) for k, v in by_agent.items()},
        'by_iteration': {k: collect(v) for k, v in by_iter.items()}
    }

def print_summary_statistics(analysis_results: AnalysisResults):
    """Print statistics for all agents and iterations"""

    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS STATISTICS")
    print("=" * 80)

    agg = analysis_results.aggregated_metrics
    n = len(analysis_results.all_results)

    print(f"\nOverall results ({n} successful analyses):\n" + "-" * 40)

    # Print metrics
    for key, name, fmt in [('accuracy', 'Accuracy', '.4f'),
                           ('coverage', 'Coverage', '.4f'),
                           ('compression', 'Compression', '.4f'),
                           ('build_time', 'Build time', '.3f'),
                           ('test_time', 'Test time', '.3f')]:
        if key in agg.get('overall', {}):
            m = agg['overall'][key]
            print(f"\n{name}: {m['mean']:{fmt}} ± {m['std']:{fmt}}"
                  + (f" [{m['min']:{fmt}}, {m['max']:{fmt}}]" if key!='build_time' and key!='test_time' else ""))

    # Top agents
    if by_agent := agg.get('by_agent', {}):
        print(f"\nBest agents:\n  " + "\n  ".join(
            f"Agent {aid}: {data['accuracy']['mean']:.4f}"
            for aid, data in sorted(by_agent.items(),
                                   key=lambda x: x[1]['accuracy']['mean'],
                                   reverse=True)[:5]
        ))

def plot_all_results(analysis_results, config):
    """
    Function to call all plot generation

    Args:
        analysis_results: analysis results
        config: experiment settings
    """
    try:
        # Attempt to import visualization module
        from visualization import (
            setup_plot_style, plot_heatmaps, plot_distributions,
            plot_time_analysis, plot_best_worst
        )

        # Basic setup
        setup_plot_style()

        # Prepare data
        results = analysis_results.all_results
        agents = sorted({r.agent_id for r in results})
        iterations = sorted({r.iteration_id for r in results})
        save_dir = config.output_dir

        # Generate plots
        if config.verbose:
            print(f"\nGenerating plots in directory: {save_dir}")
        if hasattr(config, 'plot_types'):
            # Selective plotting
            plot_types = config.plot_types
            if 'heatmaps' in plot_types:
                plot_heatmaps(results, save_dir, agents, iterations)
            if 'distributions' in plot_types:
                plot_distributions(results, save_dir)
            if 'time' in plot_types:
                plot_time_analysis(results, save_dir)
            if 'best_worst' in plot_types:
                plot_best_worst(results, save_dir)
        else:
            # Plot all
            plot_heatmaps(results, save_dir, agents, iterations)
            plot_distributions(results, save_dir)
            plot_time_analysis(results, save_dir)
            plot_best_worst(results, save_dir)

        if config.verbose:
            print("Plots generated successfully")

    except ImportError as e:
        print(f"Visualization module not found: {e}")
        print("Plots will not be generated")
    except Exception as e:
        print(f"Error generating plots: {e}")


# =================================================
# Save results
# =================================================

def _to_serializable(obj: Any) -> Any:
    """
    Recursive conversion of objects to JSON-compatible format
    """
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {_to_serializable(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, SimpleNamespace):
        return _to_serializable(vars(obj))
    if hasattr(obj, '__dict__'):
        return {k: _to_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return obj

def save_results_json(analysis_results: AnalysisResults, filepath: str) -> str:
    """
    Save results to JSON
    """
    data = {
        'results': [_to_serializable(r.__dict__) for r in analysis_results.all_results],
        'params': _to_serializable(analysis_results.parameters)
    }
    if analysis_results.parameters.mode == 'full':
        data['aggregated'] = _to_serializable(analysis_results.aggregated_metrics),

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath

def save_results_csv(analysis_results: AnalysisResults, filepath: str) -> str:
    """Save results to CSV"""
    import csv

    fieldnames = [
        'agent', 'iteration', 'accuracy', 'coverage', 'compression',
        'representatives', 'train_size', 'test_size', 'build_time', 'test_time'
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in analysis_results.all_results:
            writer.writerow({
                'agent': r.agent_id,
                'iteration': r.iteration_id,
                'accuracy': f"{r.metrics.accuracy:.4f}",
                'coverage': f"{r.metrics.coverage:.4f}",
                'compression': f"{r.metrics.compression_ratio:.4f}",
                'representatives': r.representatives_count,
                'train_size': r.train_data_size,
                'test_size': r.test_data_size,
                'build_time': f"{r.build_time:.3f}",
                'test_time': f"{r.test_time:.3f}"
            })

    return filepath

def save_analysis_results(
    analysis_results: AnalysisResults,
    save_dir: str = "analysis_results"
):
    """Save analysis results to files"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.join(save_dir, f"analysis_{timestamp}")

    json = save_results_json(analysis_results, f"{base_filename}.json")
    csv = save_results_csv(analysis_results, f"{base_filename}.csv")

    if analysis_results.parameters.verbose:
        print(f"JSON saved: {json}")
        print(f"CSV saved: {csv}")


def print_single_result_summary(result, args):
    """Print summary results for a single agent and iteration"""

    print(f"\n   Results:")
    print(f"   Training: {result.train_data_size} points")
    print(f"   Representatives: {result.representatives_count}")
    print(f"   Test: {result.test_data_size} points")
    print(f"   Accuracy: {result.metrics.accuracy:.4f}")
    print(f"   Coverage: {result.metrics.coverage:.4f}")
    print(f"   Compression ratio: {result.metrics.compression_ratio:.4f}")
    print(f"   Build time: {result.build_time:.3f}s")
    print(f"   Test time: {result.test_time:.3f}s")
    if args.verbose:
        print(f"   Representatives by class:")
        for class_id, count in sorted(result.class_distribution.items()):
            print(f"     Class {class_id}: {count} representatives")

# =================================================
# Run experiment
# =================================================


def run_analysis(data, data_test, args):
    """Run analysis for all agents and iterations"""

    all_results = []
    total = len(args.agents) * len(args.iterations)
    # Analyze each agent on each iteration
    for idx, (a, i) in enumerate(product(args.agents, args.iterations)):
        print(f"\n[{idx+1}/{total}]   Agent {a}, Iteration {i}:")

        result, preds_np, clf, update_durations = analyze_agent_iteration(data, data_test, a, i, args)
        if result:
            all_results.append(result)
            if hasattr(args, 'save_pred') and args.save_pred:
                np.savez(os.path.join(args.output_dir, f"pred_a{a:02}_i{i}"), **preds_np)
            if args.classifier == 'epsilon_net' and hasattr(args, 'save_reprs') and args.save_reprs:
                clf.save_representatives(os.path.join(args.output_dir, f"reprs_a{a:02}_i{i}"))
            if args.classifier == 'epsilon_net' and hasattr(args, 'update') and args.update:
                np.save(os.path.join(args.output_dir, f"updates_a{a:02}_i{i}"), update_durations)

    if len(all_results) == 0:
        print("No results to save")
        return

    analysis_results = AnalysisResults(
            all_results=all_results,
            aggregated_metrics=aggregate_results(all_results),
            parameters=args,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    save_analysis_results(analysis_results, save_dir=args.output_dir)

    if args.mode == 'single':
        print_single_result_summary(all_results[0], args)
    else:
        print_summary_statistics(analysis_results)

        # Generate plots
        plot_all_results(
            analysis_results=analysis_results,
            config=args
        )
