#!/usr/bin/env python3
"""
Baseline Comparison Script for Shimmer/LIRA

Compares:
1. LIRA Progressive (full curriculum: K=1→4, fixed→variable, +confidence)
2. LIRA Baseline (K=1 only, fixed masking, no confidence - like BERT)
3. Optionally: Phase-by-phase training (old approach)

Usage:
    python compare_baselines.py --num_samples 100000 --device cuda --gpu 0
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_training(name: str, args: list[str], log_file: str) -> dict:
    """Run a training job and capture results."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")
    print(f"Command: python train.py {' '.join(args)}")
    print(f"Log: {log_file}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', 'train.py'] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)        # Write to log

        process.wait()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    return {
        "name": name,
        "duration_seconds": duration,
        "return_code": process.returncode,
        "log_file": log_file,
    }


def parse_results_from_log(log_file: str) -> dict:
    """Parse training results from log file."""
    results = {
        "final_val_loss": None,
        "final_val_acc": None,
        "best_val_loss": None,
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Look for final epoch results
        lines = content.split('\n')
        for line in reversed(lines):
            if 'ValLoss' in line and 'ValAcc' in line:
                # Parse: "TrainLoss: X | ValLoss: Y | ValAcc: Z%"
                parts = line.split('|')
                for part in parts:
                    if 'ValLoss' in part:
                        try:
                            results["final_val_loss"] = float(part.split(':')[1].strip())
                        except:
                            pass
                    if 'ValAcc' in part:
                        try:
                            results["final_val_acc"] = float(part.split(':')[1].strip().replace('%', ''))
                        except:
                            pass
                break

        # Look for best val loss
        for line in lines:
            if 'Best val loss' in line:
                try:
                    results["best_val_loss"] = float(line.split(':')[-1].strip())
                except:
                    pass

    except Exception as e:
        print(f"Warning: Could not parse {log_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare LIRA training approaches")

    # Model config
    parser.add_argument("--hidden_size", type=int, default=288)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=256)

    # Data
    parser.add_argument("--dataset", type=str, default="tinystories")
    parser.add_argument("--num_samples", type=int, default=100000)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stage_epochs", type=int, default=3,
                        help="Epochs per stage for progressive (total = 4x)")
    parser.add_argument("--baseline_epochs", type=int, default=12,
                        help="Total epochs for baseline (should match progressive total)")
    parser.add_argument("--lr", type=float, default=3e-4)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=True)

    # What to run
    parser.add_argument("--skip_progressive", action="store_true",
                        help="Skip progressive training (if already done)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline training (if already done)")
    parser.add_argument("--run_phases", action="store_true",
                        help="Also run phase-by-phase training for comparison")

    # Output
    parser.add_argument("--output_dir", type=str, default="comparison_results")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Common args for all training runs
    common_args = [
        "--model", "lira",
        "--dataset", args.dataset,
        "--num_samples", str(args.num_samples),
        "--vocab_size", str(args.vocab_size),
        "--hidden_size", str(args.hidden_size),
        "--num_layers", str(args.num_layers),
        "--num_heads", str(args.num_heads),
        "--max_seq_len", str(args.max_seq_len),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
        "--gpu", str(args.gpu),
    ]

    if args.fp16:
        common_args.append("--fp16")

    results = {}

    # ========== 1. LIRA Progressive ==========
    if not args.skip_progressive:
        progressive_args = common_args + [
            "--progressive",
            "--stage_epochs", str(args.stage_epochs),
            "--checkpoint_name", f"compare_progressive_{timestamp}",
        ]

        result = run_training(
            "LIRA Progressive (K=1→4, curriculum)",
            progressive_args,
            f"{args.output_dir}/progressive_{timestamp}.log"
        )
        result.update(parse_results_from_log(result["log_file"]))
        results["progressive"] = result

    # ========== 2. LIRA Baseline (K=1, fixed masking) ==========
    if not args.skip_baseline:
        baseline_args = common_args + [
            "--phase", "1",  # K=1, fixed 30% masking
            "--epochs", str(args.baseline_epochs),
            "--checkpoint_name", f"compare_baseline_{timestamp}",
        ]

        result = run_training(
            "LIRA Baseline (K=1, fixed 30% masking)",
            baseline_args,
            f"{args.output_dir}/baseline_{timestamp}.log"
        )
        result.update(parse_results_from_log(result["log_file"]))
        results["baseline"] = result

    # ========== 3. Optional: Phase-by-Phase ==========
    if args.run_phases:
        phase_epochs = args.stage_epochs

        # Phase 1
        p1_args = common_args + [
            "--phase", "1",
            "--epochs", str(phase_epochs),
            "--checkpoint_name", f"compare_phase1_{timestamp}",
        ]
        result1 = run_training(
            "Phase 1: K=1, fixed masking",
            p1_args,
            f"{args.output_dir}/phase1_{timestamp}.log"
        )

        # Phase 2 (load from phase 1)
        p2_args = common_args + [
            "--phase", "2",
            "--epochs", str(phase_epochs),
            "--load_checkpoint", f"checkpoints/compare_phase1_{timestamp}_best.pt",
            "--checkpoint_name", f"compare_phase2_{timestamp}",
        ]
        result2 = run_training(
            "Phase 2: K=4, fixed masking",
            p2_args,
            f"{args.output_dir}/phase2_{timestamp}.log"
        )

        # Phase 3
        p3_args = common_args + [
            "--phase", "3",
            "--epochs", str(phase_epochs),
            "--load_checkpoint", f"checkpoints/compare_phase2_{timestamp}_best.pt",
            "--checkpoint_name", f"compare_phase3_{timestamp}",
        ]
        result3 = run_training(
            "Phase 3: K=4, variable masking",
            p3_args,
            f"{args.output_dir}/phase3_{timestamp}.log"
        )

        # Phase 4
        p4_args = common_args + [
            "--phase", "4",
            "--epochs", str(phase_epochs),
            "--load_checkpoint", f"checkpoints/compare_phase3_{timestamp}_best.pt",
            "--checkpoint_name", f"compare_phase4_{timestamp}",
        ]
        result4 = run_training(
            "Phase 4: K=4, variable + confidence",
            p4_args,
            f"{args.output_dir}/phase4_{timestamp}.log"
        )
        result4.update(parse_results_from_log(result4["log_file"]))

        # Total time for phases
        total_phase_time = sum([
            result1["duration_seconds"],
            result2["duration_seconds"],
            result3["duration_seconds"],
            result4["duration_seconds"],
        ])

        results["phases"] = {
            "name": "Phase-by-Phase (1→2→3→4)",
            "duration_seconds": total_phase_time,
            "final_val_loss": result4.get("final_val_loss"),
            "final_val_acc": result4.get("final_val_acc"),
            "best_val_loss": result4.get("best_val_loss"),
        }

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.hidden_size}h × {args.num_layers}L × {args.num_heads}H")
    print(f"Dataset: {args.dataset} ({args.num_samples} samples)")
    print(f"{'='*60}\n")

    print(f"{'Approach':<40} {'Val Loss':<12} {'Val Acc':<12} {'Time (min)':<12}")
    print("-" * 76)

    for key, result in results.items():
        name = result.get("name", key)[:38]
        val_loss = result.get("final_val_loss", "N/A")
        val_acc = result.get("final_val_acc", "N/A")
        duration = result.get("duration_seconds", 0) / 60

        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        val_acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, float) else str(val_acc)

        print(f"{name:<40} {val_loss_str:<12} {val_acc_str:<12} {duration:<12.1f}")

    print("-" * 76)

    # Determine winner
    if "progressive" in results and "baseline" in results:
        prog_loss = results["progressive"].get("final_val_loss")
        base_loss = results["baseline"].get("final_val_loss")
        prog_acc = results["progressive"].get("final_val_acc")
        base_acc = results["baseline"].get("final_val_acc")

        if prog_loss and base_loss:
            if prog_loss < base_loss:
                improvement = (base_loss - prog_loss) / base_loss * 100
                print(f"\n✓ Progressive wins on loss: {improvement:.1f}% lower than baseline")
            else:
                print(f"\n✗ Baseline has lower loss")

        if prog_acc and base_acc:
            if prog_acc > base_acc:
                improvement = prog_acc - base_acc
                print(f"✓ Progressive wins on accuracy: +{improvement:.1f}% over baseline")
            else:
                print(f"✗ Baseline has higher accuracy")

    # Save results
    results_file = f"{args.output_dir}/comparison_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
