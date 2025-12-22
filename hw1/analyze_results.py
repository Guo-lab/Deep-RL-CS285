#!/usr/bin/env python3
"""Simple script to analyze experiment results and generate LaTeX tables."""

import os
import glob
import re
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: Install tensorboard with: pip install tensorboard")
    exit(1)


def get_metrics(event_file):
    """Get final metrics from event file."""
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        metrics = {}
        for tag in ea.Tags()['scalars']:
            if 'Eval_AverageReturn' in tag:
                metrics['eval_return'] = ea.Scalars(tag)[-1].value
            elif 'Eval_StdReturn' in tag:
                metrics['eval_std'] = ea.Scalars(tag)[-1].value
            elif 'Train_AverageReturn' in tag:
                metrics['train_return'] = ea.Scalars(tag)[-1].value

        return metrics
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return None


def parse_name(dirname):
    """Extract info from directory name."""
    lower = dirname.lower()

    # Environment
    if 'ant' in lower: env = 'Ant-v4'
    elif 'walker' in lower: env = 'Walker2d-v4'
    elif 'cheetah' in lower: env = 'HalfCheetah-v4'
    elif 'hopper' in lower: env = 'Hopper-v4'
    else: env = None

    # Parameters
    eval_bs = int(m.group(1)) if (m := re.search(r'evalbs(\d+)', lower)) else None
    train_steps = int(m.group(1)) if (m := re.search(r'trainsteps(\d+)', lower)) else None

    return {'env': env, 'eval_bs': eval_bs, 'train_steps': train_steps}


def collect_results(data_dir='data'):
    """Collect all experiment results."""
    results = []

    for exp_dir in glob.glob(os.path.join(data_dir, '*bc_*')):
        event_files = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
        if not event_files:
            continue

        info = parse_name(os.path.basename(exp_dir))
        metrics = get_metrics(event_files[0])

        if metrics and info['env']:
            results.append({**info, **metrics})

    return results


def print_latex_table(title, results, param_key, param_name):
    """Print a LaTeX table with rowspan for environments and train return."""
    print(f"\\subsection*{{\\small {title}}}")
    print("\\vspace{-0.5em}")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\begin{tabular}{llrrr}")
    print("\\toprule")
    print(f"Environment & {param_name} & Eval Return & Eval Std & Train Return \\\\")
    print("\\midrule")

    # Group by environment
    results.sort(key=lambda x: (x['env'], x[param_key] or 0))
    grouped = defaultdict(list)
    for r in results:
        grouped[r['env']].append(r)

    for env in sorted(grouped.keys()):
        rows = grouped[env]
        n_rows = len(rows)

        for i, r in enumerate(rows):
            if i == 0:
                # First row: print environment and train return with multirow
                print(f"\\multirow{{{n_rows}}}{{*}}{{{env}}} & {r[param_key]} & "
                      f"{r['eval_return']:.2f} & {r['eval_std']:.2f} & "
                      f"\\multirow{{{n_rows}}}{{*}}{{{r['train_return']:.2f}}} \\\\")
            else:
                # Subsequent rows: empty environment and train return columns
                print(f" & {r[param_key]} & "
                      f"{r['eval_return']:.2f} & {r['eval_std']:.2f} & \\\\")

        # Add a light rule between environments
        if env != sorted(grouped.keys())[-1]:
            print("\\cmidrule(lr){2-4}")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{{title}}}")
    print("\\end{table}")
    print("\\vspace{-1em}\n")


def main():
    results = collect_results()

    if not results:
        print("No results found!")
        return

    # LaTeX document header
    print("\\documentclass[9pt]{article}")
    print("\\usepackage{booktabs}")
    print("\\usepackage{multirow}")
    print("\\usepackage[margin=0.3in]{geometry}")
    print("\\usepackage[font=small,skip=2pt]{caption}")
    print("\\setlength{\\parindent}{0pt}")
    print("\\setlength{\\parskip}{0pt}")
    print("\\setlength{\\abovecaptionskip}{2pt}")
    print("\\setlength{\\belowcaptionskip}{2pt}")
    print("\\setlength{\\textfloatsep}{5pt}")
    print("\\setlength{\\intextsep}{5pt}")
    print()
    print("\\begin{document}")
    print()

    # Table 1: eval batch size experiments
    eval_bs_results = [r for r in results if r['eval_bs']]
    print_latex_table("Effect of Evaluation Batch Size", eval_bs_results, 'eval_bs', 'Eval Batch Size')

    # Table 2: train steps experiments
    train_results = [r for r in results if r['train_steps']]
    print_latex_table("Effect of Training Steps per Iteration", train_results, 'train_steps', 'Train Steps')

    # Best results table
    print("\\subsection*{\\small Best Performance by Environment}")
    print("\\vspace{-0.5em}")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{4pt}")
    print("\\begin{tabular}{lrl}")
    print("\\toprule")
    print("Environment & Best Eval Return & Configuration \\\\")
    print("\\midrule")

    best = defaultdict(lambda: {'return': -float('inf'), 'config': ''})
    for r in results:
        if r['eval_return'] > best[r['env']]['return']:
            best[r['env']]['return'] = r['eval_return']
            config = f"eval\\_bs={r['eval_bs']}" if r['eval_bs'] else f"train\\_steps={r['train_steps']}"
            best[r['env']]['config'] = config

    for env in sorted(best.keys()):
        print(f"{env} & {best[env]['return']:.2f} & {best[env]['config']} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Best Performance by Environment}")
    print("\\end{table}")
    print()
    print("\\end{document}")


if __name__ == '__main__':
    main()
