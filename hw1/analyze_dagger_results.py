#!/usr/bin/env python3
"""
Analyze DAgger experiment results and generate learning curves for Problem 2.

Creates plots showing:
- DAgger learning curves (iterations vs. mean return with error bars)
- Expert policy baseline (horizontal line)
- Behavioral cloning baseline (horizontal line)
"""

import os
import glob
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: Install tensorboard with: pip install tensorboard")
    exit(1)


def get_all_metrics(event_file):
    """Extract all Eval_AverageReturn and Eval_StdReturn values across iterations."""
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        eval_returns = []
        eval_stds = []

        if 'Eval_AverageReturn' in ea.Tags()['scalars']:
            eval_returns = [(s.step, s.value) for s in ea.Scalars('Eval_AverageReturn')]

        if 'Eval_StdReturn' in ea.Tags()['scalars']:
            eval_stds = [(s.step, s.value) for s in ea.Scalars('Eval_StdReturn')]

        return eval_returns, eval_stds
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return [], []


def parse_dagger_name(dirname):
    """Extract environment and seed from DAgger experiment directory name."""
    lower = dirname.lower()

    # Environment
    if 'ant' in lower:
        env = 'Ant-v4'
    elif 'walker' in lower:
        env = 'Walker2d-v4'
    elif 'cheetah' in lower:
        env = 'HalfCheetah-v4'
    elif 'hopper' in lower:
        env = 'Hopper-v4'
    else:
        env = None

    # Seed
    seed = int(m.group(1)) if (m := re.search(r'seed(\d+)', lower)) else None

    return {'env': env, 'seed': seed}


def collect_dagger_results(data_dir='data'):
    """Collect DAgger experiment results grouped by environment and seed."""
    results = defaultdict(lambda: defaultdict(list))

    for exp_dir in glob.glob(os.path.join(data_dir, '*dagger_*')):
        event_files = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
        if not event_files:
            continue

        info = parse_dagger_name(os.path.basename(exp_dir))
        if not info['env']:
            continue

        eval_returns, eval_stds = get_all_metrics(event_files[0])

        if eval_returns:
            results[info['env']][info['seed']] = {
                'returns': eval_returns,
                'stds': eval_stds
            }

    return results


def get_bc_baseline(data_dir='data'):
    """Get best BC performance for each environment from Problem 1."""
    bc_results = {}

    for exp_dir in glob.glob(os.path.join(data_dir, '*bc_*')):
        event_files = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
        if not event_files:
            continue

        dirname = os.path.basename(exp_dir).lower()

        # Environment
        if 'ant' in dirname:
            env = 'Ant-v4'
        elif 'walker' in dirname:
            env = 'Walker2d-v4'
        elif 'cheetah' in dirname:
            env = 'HalfCheetah-v4'
        elif 'hopper' in dirname:
            env = 'Hopper-v4'
        else:
            continue

        try:
            ea = EventAccumulator(event_files[0])
            ea.Reload()

            if 'Eval_AverageReturn' in ea.Tags()['scalars']:
                avg_return = ea.Scalars('Eval_AverageReturn')[-1].value

                # Keep best BC result
                if env not in bc_results or avg_return > bc_results[env]:
                    bc_results[env] = avg_return
        except Exception as e:
            print(f"Error reading BC results from {event_files[0]}: {e}")
            continue

    return bc_results


def get_expert_baseline(data_dir='data'):
    """
    Try to extract expert performance from 'Initial_DataCollection_AverageReturn'
    or 'Train_AverageReturn' in iteration 0 of any experiment.
    """
    expert_results = {}

    # Check both BC and DAgger experiments
    for exp_dir in glob.glob(os.path.join(data_dir, '*_*')):
        event_files = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
        if not event_files:
            continue

        dirname = os.path.basename(exp_dir).lower()

        # Environment
        if 'ant' in dirname:
            env = 'Ant-v4'
        elif 'walker' in dirname:
            env = 'Walker2d-v4'
        elif 'cheetah' in dirname:
            env = 'HalfCheetah-v4'
        elif 'hopper' in dirname:
            env = 'Hopper-v4'
        else:
            continue

        if env in expert_results:
            continue

        try:
            ea = EventAccumulator(event_files[0])
            ea.Reload()

            # Try to get expert performance from initial data collection
            if 'Initial_DataCollection_AverageReturn' in ea.Tags()['scalars']:
                expert_return = ea.Scalars('Initial_DataCollection_AverageReturn')[0].value
                expert_results[env] = expert_return
            elif 'Train_AverageReturn' in ea.Tags()['scalars']:
                # Use training data average as proxy for expert
                train_return = ea.Scalars('Train_AverageReturn')[0].value
                expert_results[env] = train_return
        except Exception:
            continue

    return expert_results


def plot_learning_curves(dagger_results, bc_baseline, expert_baseline, output_dir='plots'):
    """Generate learning curve plots for each environment. Returns list of generated files."""
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for env in sorted(dagger_results.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Aggregate results across seeds
        seeds_data = dagger_results[env]

        if not seeds_data:
            print(f"No data for {env}")
            continue

        # Find maximum number of iterations
        max_iters = max(len(data['returns']) for data in seeds_data.values())

        # Aggregate returns and stds across seeds
        mean_returns = []
        std_returns = []

        for iter_idx in range(max_iters):
            iter_returns = []
            iter_stds = []

            for seed, data in seeds_data.items():
                if iter_idx < len(data['returns']):
                    iter_returns.append(data['returns'][iter_idx][1])
                    if iter_idx < len(data['stds']):
                        iter_stds.append(data['stds'][iter_idx][1])

            if iter_returns:
                mean_returns.append(np.mean(iter_returns))
                # Standard error across seeds
                std_returns.append(np.std(iter_returns))

        iterations = list(range(len(mean_returns)))
        mean_returns = np.array(mean_returns)
        std_returns = np.array(std_returns)

        # Plot DAgger learning curve with error bars
        ax.plot(iterations, mean_returns, 'o-', label='DAgger', linewidth=2, markersize=6)
        ax.fill_between(iterations,
                        mean_returns - std_returns,
                        mean_returns + std_returns,
                        alpha=0.3)

        # Plot BC baseline
        if env in bc_baseline:
            ax.axhline(y=bc_baseline[env], color='orange', linestyle='--',
                      linewidth=2, label='Behavioral Cloning')

        # Plot Expert baseline
        if env in expert_baseline:
            ax.axhline(y=expert_baseline[env], color='green', linestyle='--',
                      linewidth=2, label='Expert Policy')

        ax.set_xlabel('DAgger Iteration', fontsize=12)
        ax.set_ylabel('Mean Return', fontsize=12)
        ax.set_title(f'Learning Curve: {env}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Save figure
        filename = f"dagger_learning_curve_{env.replace('-', '_').lower()}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

        # Also save as PNG
        png_filepath = filepath.replace('.pdf', '.png')
        plt.savefig(png_filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {png_filepath}")

        # Track generated file
        generated_files.append({'env': env, 'filename': filename})

        plt.close()

    return generated_files


def print_summary(dagger_results, bc_baseline, expert_baseline):
    """Print a text summary of results."""
    print("\n" + "="*60)
    print("DAGGER RESULTS SUMMARY")
    print("="*60)

    for env in sorted(dagger_results.keys()):
        print(f"\n{env}:")
        print("-" * 40)

        seeds_data = dagger_results[env]

        # Get final iteration performance
        final_returns = []
        for seed, data in seeds_data.items():
            if data['returns']:
                final_returns.append(data['returns'][-1][1])

        if final_returns:
            final_mean = np.mean(final_returns)
            final_std = np.std(final_returns)
            print(f"  DAgger (final):  {final_mean:.2f} ± {final_std:.2f}")

        if env in bc_baseline:
            print(f"  BC baseline:     {bc_baseline[env]:.2f}")

        if env in expert_baseline:
            print(f"  Expert baseline: {expert_baseline[env]:.2f}")

        # Calculate improvement
        if final_returns and env in bc_baseline:
            improvement = ((final_mean - bc_baseline[env]) / abs(bc_baseline[env])) * 100
            print(f"  Improvement over BC: {improvement:+.1f}%")

    print("\n" + "="*60)


def generate_latex_report(dagger_results, bc_baseline, expert_baseline, plot_files, output_file='dagger_report.tex'):
    """Generate LaTeX report with learning curves and results table."""

    with open(output_file, 'w') as f:
        # LaTeX document header
        f.write("\\documentclass[10pt]{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{subcaption}\n")
        f.write("\\usepackage[margin=0.75in]{geometry}\n")
        f.write("\\usepackage[font=small,skip=4pt]{caption}\n")
        f.write("\\setlength{\\parindent}{0pt}\n")
        f.write("\\setlength{\\parskip}{6pt}\n")
        f.write("\n")
        f.write("\\begin{document}\n")
        f.write("\n")
        f.write("\\section*{Problem 2: DAgger Results}\n")
        f.write("\n")

        # Add learning curves with captions
        f.write("\\subsection*{Learning Curves}\n")
        f.write("\n")

        for plot_info in plot_files:
            env = plot_info['env']
            filename = plot_info['filename']

            # Get final performance stats
            seeds_data = dagger_results[env]
            final_returns = []
            for data in seeds_data.values():
                if data['returns']:
                    final_returns.append(data['returns'][-1][1])

            final_mean = np.mean(final_returns) if final_returns else 0
            final_std = np.std(final_returns) if final_returns else 0

            bc_perf = bc_baseline.get(env, 'N/A')
            expert_perf = expert_baseline.get(env, 'N/A')

            f.write("\\begin{figure}[h]\n")
            f.write("\\centering\n")
            f.write(f"\\includegraphics[width=0.85\\textwidth]{{plots/{filename}}}\n")
            f.write(f"\\caption{{\\textbf{{DAgger Learning Curve for {env}.}} ")
            f.write(f"The plot shows mean return over {len(seeds_data)} seeds with error bars (standard deviation). ")
            f.write("Network architecture: 2-layer MLP with 64 hidden units per layer. ")
            f.write("Training: 1000 gradient steps per iteration, batch size 1000. ")
            f.write(f"Final DAgger performance: {final_mean:.2f} $\\pm$ {final_std:.2f}. ")

            if isinstance(bc_perf, float):
                f.write(f"BC baseline: {bc_perf:.2f}. ")
            if isinstance(expert_perf, float):
                f.write(f"Expert performance: {expert_perf:.2f}.")

            f.write("}\n")
            f.write(f"\\label{{fig:{env.lower().replace('-', '_')}}}\n")
            f.write("\\end{figure}\n")
            f.write("\n")

        # Add results table
        f.write("\\subsection*{Numerical Results Summary}\n")
        f.write("\\vspace{-0.5em}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Environment & Expert & BC & DAgger (Final) & Improvement \\\\\n")
        f.write("\\midrule\n")

        for env in sorted(dagger_results.keys()):
            seeds_data = dagger_results[env]

            # Get final DAgger performance
            final_returns = []
            for data in seeds_data.values():
                if data['returns']:
                    final_returns.append(data['returns'][-1][1])

            final_mean = np.mean(final_returns) if final_returns else 0
            final_std = np.std(final_returns) if final_returns else 0

            expert_perf = expert_baseline.get(env, None)
            bc_perf = bc_baseline.get(env, None)

            expert_str = f"{expert_perf:.2f}" if expert_perf else "N/A"
            bc_str = f"{bc_perf:.2f}" if bc_perf else "N/A"
            dagger_str = f"{final_mean:.2f} $\\pm$ {final_std:.2f}"

            # Calculate improvement over BC
            if bc_perf and final_returns:
                improvement = ((final_mean - bc_perf) / abs(bc_perf)) * 100
                improvement_str = f"{improvement:+.1f}\\%"
            else:
                improvement_str = "N/A"

            f.write(f"{env} & {expert_str} & {bc_str} & {dagger_str} & {improvement_str} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Summary of DAgger results compared to Expert and BC baselines. ")
        f.write("DAgger values show mean $\\pm$ std across seeds. ")
        f.write("Improvement shows percentage change from BC baseline.}\n")
        f.write("\\label{tab:dagger_results}\n")
        f.write("\\end{table}\n")
        f.write("\n")

        f.write("\\end{document}\n")

    print(f"\nLaTeX report saved to: {output_file}")


def main():
    print("Analyzing DAgger experiment results...")

    # Collect results
    dagger_results = collect_dagger_results()
    bc_baseline = get_bc_baseline()
    expert_baseline = get_expert_baseline()

    if not dagger_results:
        print("\nNo DAgger results found!")
        print("Make sure you've run: bash run_dagger_experiments.sh")
        return

    print(f"\nFound DAgger results for {len(dagger_results)} environment(s)")
    print(f"Found BC baseline for {len(bc_baseline)} environment(s)")
    print(f"Found Expert baseline for {len(expert_baseline)} environment(s)")

    # Generate plots
    plot_files = plot_learning_curves(dagger_results, bc_baseline, expert_baseline)

    # Generate LaTeX report
    generate_latex_report(dagger_results, bc_baseline, expert_baseline, plot_files)

    # Print summary
    print_summary(dagger_results, bc_baseline, expert_baseline)

    print("\nAnalysis complete!")
    print("- Plots saved in 'plots/' directory")
    print("- LaTeX report: dagger_report.tex")
    print("- Compile with: pdflatex dagger_report.tex")


if __name__ == '__main__':
    main()
