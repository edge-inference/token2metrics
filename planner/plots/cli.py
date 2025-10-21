#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""CLI for plotting Natural-Plan evaluation results."""
import argparse
from pathlib import Path
from ..analysis import ResultsAnalyzer
from .accuracy import plot_accuracy
from ..config import TASKS, MODELS, DEFAULT_ARGS


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Natural-Plan plots")
    p.add_argument("--results-dir", default=DEFAULT_ARGS["results_dir"], help="Results root directory")
    p.add_argument("--output-dir", default=DEFAULT_ARGS["output_dir"] + "plots", help="Plots output directory")
    p.add_argument("--task", choices=TASKS + ["all"], default="all",
                   help="Task to plot; 'all' generates combined plot too")
    p.add_argument("--models", default=DEFAULT_ARGS["models"], help="Comma-separated models to include")
    args = p.parse_args()

    analyzer = ResultsAnalyzer(Path(args.results_dir))
    results = analyzer.scan_all_results()
    if not results:
        print("No results found; nothing to plot")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models.split(",") if args.models else MODELS
    if args.task == "all":
        plot_accuracy(results, task=None, models=models, output_dir=output_dir)
        for task in TASKS:
            plot_accuracy(results, task=task, models=models, output_dir=output_dir)
    else:
        plot_accuracy(results, task=args.task, models=models, output_dir=output_dir)

    print(f"Plots written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


