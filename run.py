#!/usr/bin/env python3
"""
Interactive experiment launcher for Bat Face Recognition project.

Prompts for experiment parameters and launches the appropriate command.
Always shows the underlying command so you can copy/paste it later.

Usage:
    python run.py

Training prompts for species, data source, augmentation, background, and train/test split mode
(image_split vs bat_split), then launches train_siamese with matching flags.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))


def prompt_choice(prompt, options, default=None):
    """Prompt user to pick from a numbered list. Returns the chosen value."""
    print(prompt)
    for i, (label, value) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"> ").strip()
        if not raw and default is not None:
            return default
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][1]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}")


def prompt_yn(prompt, default=True):
    """Prompt yes/no. Returns bool."""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw.startswith('y')


def run_training():
    """Interactive training flow."""
    print("\n--- Training Configuration ---\n")

    species_map = {"m": "mauritius", "r": "rousettus"}
    bat_type = prompt_choice("Bat species:", [
        ("Mauritius (m)", "m"),
        ("Rousettus (r)", "r"),
    ])
    species = species_map[bat_type]

    data_source = prompt_choice("\nData source:", [
        ("Video frames", "video"),
        ("Still images", "still"),
    ])

    augmented = prompt_yn("\nUse augmented data?", default=False)

    bg_type = prompt_choice("\nBackground photos:", [
        ("Green", "green"),
        ("Random", "random"),
        ("Original", "original"),
    ])

    split_mode = prompt_choice(
        "\nTrain/test split:",
        [
            ("Per-image within each class (image_split) — same identities in train and test", "image_split"),
            ("By bat identity (bat_split) — disjoint classes, no identity overlap", "bat_split"),
        ],
        default="image_split",
    )

    run_in_bg = prompt_yn("\nRun in background (nohup)?", default=True)

    # Build command
    cmd_parts = [
        sys.executable, "-m", "app.siamese_training.train_siamese",
        "--bat-type", bat_type,
        "--data-source", data_source,
        "--background", bg_type,
        "--split-mode", split_mode,
    ]
    if augmented:
        cmd_parts.append("--augmented-data")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Species:    {species}")
    print(f"  Source:     {data_source}")
    print(f"  Augmented:  {'Yes' if augmented else 'No'}")
    print(f"  Background: {bg_type}")
    print(f"  Split mode: {split_mode}")
    print(f"  Nohup:      {'Yes' if run_in_bg else 'No'}")

    if not prompt_yn("\nProceed?", default=True):
        print("Aborted.")
        return

    if run_in_bg:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "training")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{timestamp}.log")

        cmd_str = " ".join(cmd_parts)
        print(f"\nLaunching training...")
        print(f"   {cmd_str}")
        print(f"   Log: {log_file}")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd_parts,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                start_new_session=True,
            )
        print(f"   PID: {proc.pid}")
    else:
        cmd_str = " ".join(cmd_parts)
        print(f"\nRunning: {cmd_str}\n")
        sys.exit(subprocess.call(cmd_parts, cwd=PROJECT_ROOT))


def _parse_experiment_params(dirname):
    """Extract training params from the experiment directory name.

    Expected format (from trainer._build_experiment_name + _create_output_directory):
      {YYYYMMDD}_siamese_{species}_{data_source}_{aug_str}_{background}_bg_{run_id}

    Returns dict with bat_type, data_source, augmented, background or None on failure.
    """
    m = re.match(
        r"\d{8}_siamese_(?P<species>rousettus|mauritius)"
        r"_(?P<source>video|still)"
        r"_(?P<aug>augmented|no_aug)"
        r"_(?P<bg>\w+)_bg_",
        dirname,
    )
    if not m:
        return None
    species = m.group("species")
    return {
        "bat_type": "m" if species == "mauritius" else "r",
        "species": species,
        "data_source": m.group("source"),
        "augmented": m.group("aug") == "augmented",
        "background": m.group("bg"),
    }


def _choose_experiment(species):
    """List experiment dirs for a species and let the user pick one.

    Returns (experiment_path, experiment_dirname) or (None, None).
    """
    cfg = _load_config()
    sn_train = cfg.siamese_network.training
    output_dirs = sn_train.get("output_dir")
    experiments_base = output_dirs[species]
    if not os.path.isdir(experiments_base):
        print(f"  No experiments directory found: {experiments_base}")
        return None, None

    experiment_dirs = sorted(
        [d for d in os.listdir(experiments_base)
         if os.path.isdir(os.path.join(experiments_base, d))],
        reverse=True,
    )
    if not experiment_dirs:
        print(f"  No experiments found in {experiments_base}")
        return None, None

    experiment_options = [(d, d) for d in experiment_dirs]
    experiment = prompt_choice("\nChoose experiment:", experiment_options)
    return os.path.join(experiments_base, experiment), experiment


def _choose_model_checkpoint(experiment_path):
    """List best_model_* dirs and let user pick. Returns (model_path, model_choice, epoch)."""
    model_subdirs = sorted([
        d for d in os.listdir(experiment_path)
        if d.startswith("best_model_") and os.path.isdir(os.path.join(experiment_path, d))
    ])
    if not model_subdirs:
        print(f"  No best_model_* directories found in {experiment_path}")
        return None, None, 0

    model_options = [(d, d) for d in model_subdirs]
    model_options.append(("Enter custom model path", "__custom__"))
    model_choice = prompt_choice("\nChoose model checkpoint:", model_options)

    if model_choice == "__custom__":
        model_path = input("  Full model path: ").strip()
    else:
        model_path = os.path.join(experiment_path, model_choice)

    # Auto-resolve epoch from training_summary.json
    model_version = 0
    summary_path = os.path.join(experiment_path, "training_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            metric_key = model_choice.replace("best_model_", "best_")
            if metric_key in summary:
                model_version = summary[metric_key]["epoch"]
        except Exception:
            pass
    if model_version == 0:
        raw = input("  Model version/epoch number [0]: ").strip()
        model_version = int(raw) if raw else 0

    return model_path, model_choice, model_version


def run_permutation_test():
    """Interactive permutation test flow -- experiment-centric.

    All training parameters (bat_type, data_source, background, etc.) are
    auto-loaded from the selected experiment directory so the permutation
    test exactly matches the original training conditions.
    """
    print("\n--- Permutation Test Configuration ---\n")

    # Step 1: Species
    species_map = {"m": "mauritius", "r": "rousettus"}
    bat_type = prompt_choice("Bat species:", [
        ("Mauritius (m)", "m"),
        ("Rousettus (r)", "r"),
    ])
    species = species_map[bat_type]

    # Step 2: Choose experiment
    experiment_path, experiment_dirname = _choose_experiment(species)
    if experiment_path is None:
        return

    # Step 3: Auto-extract training params from directory name
    exp_params = _parse_experiment_params(experiment_dirname)
    if exp_params is None:
        print(f"  Warning: Could not parse training params from directory name: {experiment_dirname}")
        print(f"  This experiment may use a legacy naming scheme.")
        print(f"  Aborting -- please use the CLI directly for legacy experiments.")
        return

    print(f"\n--- Training config (from experiment) ---")
    print(f"  Species:    {exp_params['species']}")
    print(f"  Source:     {exp_params['data_source']}")
    print(f"  Augmented:  {'Yes' if exp_params['augmented'] else 'No'}")
    print(f"  Background: {exp_params['background']}")

    # Step 4: Choose model checkpoint
    model_path, model_choice, model_version = _choose_model_checkpoint(experiment_path)
    if model_path is None:
        return

    # Step 5: Choose mode
    mode = prompt_choice("\nPermutation test mode:", [
        ("Inference-based (fast -- evaluate trained model on permuted test labels)", "inference"),
        ("Retrain from scratch (classical -- retrain with permuted labels)", "retrain"),
    ])

    # Step 6: Number of permutations
    default_n = "1000" if mode == "inference" else "100"
    n_perms = input(f"\nNumber of permutations [{default_n}]: ").strip() or default_n

    # Step 7: Degradation curve (inference mode only)
    include_degradation = False
    if mode == "inference":
        include_degradation = prompt_yn("Include degradation curve (metric vs permutation fraction)?", default=True)

    # Step 8: Background execution
    run_in_bg = prompt_yn("\nRun in background?", default=True)

    # Build command
    cmd_parts = [
        sys.executable, "-m", "app.statistical_tests.run_permutation_test",
        "--experiment-dir", experiment_path,
        "--model-path", model_path,
        "--mode", mode,
        "--n-permutations", n_perms,
    ]
    if include_degradation:
        cmd_parts.append("--degradation-curve")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Experiment:  {experiment_dirname}")
    print(f"  Model:       {model_choice} (epoch {model_version})")
    print(f"  Mode:        {mode}")
    print(f"  Permutations: {n_perms}")
    if mode == "inference":
        print(f"  Degradation curve: {'Yes' if include_degradation else 'No'}")
    print(f"  Nohup:       {'Yes' if run_in_bg else 'No'}")

    cmd_str = " ".join(cmd_parts)
    print(f"\n--- Command ---")
    print(f"  {cmd_str}")

    if not prompt_yn("\nProceed?", default=True):
        print("Aborted.")
        return

    if run_in_bg:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "permutation_tests")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{timestamp}.log")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd_parts,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                start_new_session=True,
            )
        print(f"  Log: {log_file}")
        print(f"  PID: {proc.pid}")
    else:
        sys.exit(subprocess.call(cmd_parts, cwd=PROJECT_ROOT))


def run_saliency():
    """Interactive saliency map generation."""
    print("\n--- Saliency Map Configuration ---\n")

    model_dir = input("Model directory (e.g., models/siamese/rousettus/<run_dir>/best_model_f1): ").strip()
    input_dir = input("Input directory (images organized by class): ").strip()

    cmd_parts = [
        sys.executable, "-m", "app.visualization.run_saliency_maps",
        "--model_path", model_dir,
        "--input_dir", input_dir,
    ]

    cmd_str = " ".join(cmd_parts)
    print(f"\n  {cmd_str}")

    if prompt_yn("\nProceed?", default=True):
        sys.exit(subprocess.call(cmd_parts, cwd=PROJECT_ROOT))


def _load_config():
    """Load project config.yml."""
    from config.loader import load_config
    return load_config()


def run_evaluate():
    """Interactive evaluation of a trained experiment."""
    print("\n--- Evaluate Trained Experiment ---\n")

    cfg = _load_config()

    # Step 1: Species
    species_map = {"m": "mauritius", "r": "rousettus"}
    bat_type = prompt_choice("Bat species:", [
        ("Mauritius (m)", "m"),
        ("Rousettus (r)", "r"),
    ])
    species = species_map[bat_type]

    # Step 2: Choose experiment
    experiment_path, experiment = _choose_experiment(species)
    if experiment_path is None:
        return

    # Step 3: Choose model (epoch)
    model_path, model_choice, model_version = _choose_model_checkpoint(experiment_path)
    if model_path is None:
        return

    # Step 4: Evaluation background
    bg_type = prompt_choice("\nEvaluation background:", [
        ("Green", "green"),
        ("Random", "random"),
        ("Original", "original"),
    ])

    # Step 5: Data source
    data_source = prompt_choice("\nData source:", [
        ("Video frames", "video"),
        ("Still images", "still"),
    ])

    # Step 6: Optional MLflow run id for AUC logging
    mlflow_run_id = input(
        "\nLog ROC-AUC to an existing MLflow run? (enter run id or leave blank to skip): "
    ).strip() or None

    # Step 7: Background execution
    run_in_bg = prompt_yn("\nRun in background (nohup)?", default=True)

    # Build command
    cmd_parts = [
        sys.executable, "-m", "app.generate_predictions",
        "--model", model_path,
        "--input", _resolve_input_dir(cfg, species, bg_type),
        "--model-version", str(model_version),
        "--bat-type", bat_type,
        "--source", data_source,
        "--background", bg_type,
        "--output", experiment_path,
    ]
    if mlflow_run_id:
        cmd_parts.extend(["--mlflow-run-id", mlflow_run_id])
    else:
        cmd_parts.append("--no-mlflow-log")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Species:      {species}")
    print(f"  Experiment:   {experiment}")
    print(f"  Model:        {model_choice}")
    print(f"  Version:      {model_version}")
    print(f"  Background:   {bg_type}")
    print(f"  Data source:  {data_source}")
    print(f"  Output dir:   {experiment_path}")
    print(f"  AUC -> MLflow:{' run ' + mlflow_run_id if mlflow_run_id else ' disabled'}")
    print(f"  Nohup:        {'Yes' if run_in_bg else 'No'}")

    if not prompt_yn("\nProceed?", default=True):
        print("Aborted.")
        return

    cmd_str = " ".join(cmd_parts)

    if run_in_bg:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "evaluation")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{timestamp}.log")

        print(f"\nLaunching evaluation...")
        print(f"   {cmd_str}")
        print(f"   Log: {log_file}")

        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd_parts,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
                start_new_session=True,
            )
        print(f"   PID: {proc.pid}")
    else:
        print(f"\nRunning: {cmd_str}\n")
        sys.exit(subprocess.call(cmd_parts, cwd=PROJECT_ROOT))


def _resolve_input_dir(cfg, species, bg_type):
    """Resolve the input directory for a given species and background type."""
    input_paths = cfg.siamese_network.input_paths[species]
    bg_key_map = {"green": "green_bg_input", "random": "random_bg_input", "original": "original_bg_input"}
    return input_paths.get(bg_key_map[bg_type], "")


def run_cleanup():
    """Interactive checkpoint cleanup."""
    cmd_parts = [sys.executable, "scripts/cleanup_checkpoints.py", "--dry-run"]
    print("\nRunning dry-run first...\n")
    subprocess.call(cmd_parts, cwd=PROJECT_ROOT)

    if prompt_yn("\nProceed with actual deletion?", default=False):
        cmd_parts = [sys.executable, "scripts/cleanup_checkpoints.py", "--confirm"]
        subprocess.call(cmd_parts, cwd=PROJECT_ROOT)


def main():
    print()
    print("+" + "=" * 42 + "+")
    print("|   Bat Face Recognition - Run Menu      |")
    print("+" + "=" * 42 + "+")
    print()

    action = prompt_choice("What would you like to do?", [
        ("Train a Siamese model", "train"),
        ("Run permutation test (null hypothesis)", "permutation"),
        ("Generate saliency maps", "saliency"),
        ("Evaluate a trained model", "evaluate"),
        ("Clean up old checkpoints", "cleanup"),
    ])

    if action == "train":
        run_training()
    elif action == "permutation":
        run_permutation_test()
    elif action == "saliency":
        run_saliency()
    elif action == "evaluate":
        run_evaluate()
    elif action == "cleanup":
        run_cleanup()


if __name__ == "__main__":
    main()
