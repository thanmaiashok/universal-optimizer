"""PredycatAI CLI Tool"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
import argparse

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def optimize_command(args):
    from predycat_ai.core.optimizer import PredycatOptimizer, OptimizationConfig
    
    config = OptimizationConfig(
        target_platform=args.target,
        preset=args.preset,
        max_accuracy_drop=args.max_drop,
        max_size_mb=args.max_size,
        max_ram_mb=args.max_ram,
        export_formats=args.formats.split(",") if args.formats else ["pt"],
        use_quantization=not args.no_quantize,
        use_pruning=not args.no_prune,
        use_distillation=not args.no_distill,
        enable_reasoning_check=not args.no_reasoning
    )
    
    optimizer = PredycatOptimizer(config=config, output_dir=args.output)
    
    print(f"Optimizing model: {args.model}")
    print(f"Target: {args.target}, Preset: {args.preset}")
    print(f"Max accuracy drop: {args.max_drop}%")
    
    result = optimizer.optimize(args.model)
    
    if result.success:
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE")
        print("="*50)
        print(f"Size: {result.original_size_mb:.1f}MB → {result.optimized_size_mb:.1f}MB")
        print(f"Compression: {result.compression_ratio:.1f}x")
        print(f"Latency: {result.original_latency_ms:.1f}ms → {result.optimized_latency_ms:.1f}ms ({result.latency_improvement:.1f}x faster)")
        print(f"Accuracy drop: {result.accuracy_drop:.1f}%")
        print(f"Pipeline: {' → '.join(result.pipeline_steps)}")
        print(f"Outputs: {result.output_paths}")
        print("="*50)
    else:
        print(f"FAILED: {result.error}")
        sys.exit(1)
    
    if args.report:
        with open(args.report, "w") as f:
            json.dump({
                "config": vars(args),
                "result": {
                    "original_size_mb": result.original_size_mb,
                    "optimized_size_mb": result.optimized_size_mb,
                    "compression_ratio": result.compression_ratio,
                    "latency_improvement": result.latency_improvement,
                    "accuracy_drop": result.accuracy_drop
                }
            }, f, indent=2)
        print(f"Report saved to {args.report}")


def profile_command(args):
    from predycat_ai.core.profiler import MultiProfiler, Platform
    
    profiler = MultiProfiler()
    
    print(f"Profiling model: {args.model}")
    
    results = profiler.profile_full(args.model)
    
    print("\nProfiling Results:")
    print("="*50)
    
    if results.laptop:
        print(f"Laptop: {results.laptop.latency_ms:.1f}ms, {results.laptop.vram_mb:.1f}MB VRAM")
    if results.mobile:
        print(f"Mobile: {results.mobile.latency_ms:.1f}ms, {results.mobile.ram_mb:.1f}MB RAM, {results.mobile.thermal_throttle_risk:.1f}% thermal risk")
    if results.edge:
        print(f"Edge: {results.edge.latency_ms:.1f}ms, {results.edge.ram_mb:.1f}MB RAM")
    
    print("="*50)


def evaluate_command(args):
    from predycat_ai.core.evaluator import ReasoningEvaluator, ReasoningTaskType
    
    print(f"Evaluating: {args.model}")
    
    evaluator = ReasoningEvaluator()
    score = evaluator.evaluate_reasoning(args.model)
    
    print("\nReasoning Evaluation:")
    print("="*50)
    print(f"Coherence: {score.coherence:.2f}")
    print(f"Logical Correctness: {score.logical_correctness:.2f}")
    print(f"Hallucination Rate: {score.hallucination_rate:.2f}")
    print(f"Overall Score: {score.overall_score:.2f}")
    print("="*50)


def batch_command(args):
    import glob
    
    model_files = glob.glob(args.pattern)
    
    print(f"Found {len(model_files)} models")
    
    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Processing {model_path}")
        
        try:
            result = optimize_command(args)
            args.model = model_path
        except Exception as e:
            print(f"FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PredycatAI Universal Optimizer - Production-grade AI optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    optimize_parser = subparsers.add_parser("optimize", help="Optimize a model")
    optimize_parser.add_argument("model", help="Model path or HuggingFace ID")
    optimize_parser.add_argument("--target", default="mobile", choices=["laptop", "cloud", "mobile", "edge"])
    optimize_parser.add_argument("--preset", default="balanced", choices=["balanced", "ultra_compression", "max_accuracy", "mobile_safe"])
    optimize_parser.add_argument("--max-drop", type=float, default=2.0, help="Max accuracy drop percentage")
    optimize_parser.add_argument("--max-size", type=float, help="Max model size in MB")
    optimize_parser.add_argument("--max-ram", type=float, help="Max RAM usage in MB")
    optimize_parser.add_argument("--formats", default="pt", help="Export formats (comma-separated)")
    optimize_parser.add_argument("--no-quantize", action="store_true", help="Disable quantization")
    optimize_parser.add_argument("--no-prune", action="store_true", help="Disable pruning")
    optimize_parser.add_argument("--no-distill", action="store_true", help="Disable distillation")
    optimize_parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning check")
    optimize_parser.add_argument("--output", default="./outputs", help="Output directory")
    optimize_parser.add_argument("--report", help="Save report to file")
    optimize_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    profile_parser = subparsers.add_parser("profile", help="Profile a model")
    profile_parser.add_argument("model", help="Model path")
    profile_parser.add_argument("-v", "--verbose", action="store_true")
    
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate reasoning")
    evaluate_parser.add_argument("model", help="Model path")
    evaluate_parser.add_argument("-v", "--verbose", action="store_true")
    
    batch_parser = subparsers.add_parser("batch", help="Batch optimize")
    batch_parser.add_argument("pattern", help="Glob pattern")
    batch_parser.add_argument("--target", default="mobile")
    batch_parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose if hasattr(args, "verbose") else False)
    
    if args.command == "optimize":
        optimize_command(args)
    elif args.command == "profile":
        profile_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "batch":
        batch_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()