"""YAML sweep runner: reads a YAML config, expands the parameter grid, and
launches training jobs across GPUs.

Usage:
    uv run sweeps/run_yaml_sweep.py <yaml_path> [--dry-run]
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import tyro
import yaml


def expand_grid(params: dict[str, list[str]]) -> list[dict[str, str]]:
    """Cartesian product of parameter lists."""
    if not params:
        return [{}]
    keys = list(params.keys())
    values = [params[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def substitute_command(template: str, params: dict[str, str]) -> str:
    """Replace ~VAR~ placeholders in the command template."""
    result = template
    for key, value in params.items():
        result = result.replace(f"~{key}~", str(value))
    return result


def check_no_unsubstituted(command: str) -> None:
    """Assert no ~VAR~ placeholders remain in the command."""
    remaining = re.findall(r"~[A-Za-z_][A-Za-z0-9_]*~", command)
    assert not remaining, (
        f"Unsubstituted placeholders: {remaining} in command: {command}"
    )


def run_device_queue(
    device_id: str,
    commands: list[str],
    dry_run: bool,
    concurrent_per_device: int = 1,
) -> tuple[int, int]:
    """Run a queue of commands for one device, up to N concurrently.

    With concurrent_per_device=1 (default) the queue is strictly sequential;
    with >1 up to that many commands run in parallel sharing the same GPU
    via CUDA time-slicing.

    Returns (success_count, failure_count).
    """
    if dry_run:
        for i, cmd in enumerate(commands):
            label = f"[GPU {device_id}] ({i + 1}/{len(commands)})"
            print(f"{label} {cmd}")
        return len(commands), 0

    def _run_one(indexed: tuple[int, str]) -> bool:
        i, cmd = indexed
        label = f"[GPU {device_id}] ({i + 1}/{len(commands)})"
        print(f"{label} Starting: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print(f"{label} Completed successfully")
            return True
        print(f"{label} Failed with return code {result.returncode}")
        return False

    workers = max(1, min(concurrent_per_device, len(commands)))
    if workers == 1:
        outcomes = [_run_one(item) for item in enumerate(commands)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_run_one, enumerate(commands)))

    successes = sum(1 for ok in outcomes if ok)
    failures = len(outcomes) - successes
    return successes, failures


def main(yaml_path: tyro.conf.Positional[str], dry_run: bool = False) -> None:
    config_path = Path(yaml_path)
    assert config_path.exists(), f"YAML config not found: {config_path}"

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    command_parts: list[str] = config["commands"]
    command_template = " && ".join(command_parts)

    concurrent_per_device = int(config.get("concurrent_per_device", 1))
    assert concurrent_per_device >= 1, "concurrent_per_device must be >= 1"

    params = config["parameters"]
    all_params = dict(params.get("all", {}))

    # extract device list from all.devices
    devices: list[str] = all_params.pop("devices")
    assert len(devices) > 0, "Must specify at least one device in all.devices"

    # build per-device command queues
    device_queues: dict[str, list[str]] = {d: [] for d in devices}

    for i, device_id in enumerate(devices):
        gpu_key = f"gpu{i}"
        gpu_params = params.get(gpu_key, {})

        # cartesian product of shared params x device-specific params
        shared_combos = expand_grid(all_params)
        gpu_combos = expand_grid(gpu_params)

        for shared, gpu_specific in product(shared_combos, gpu_combos):
            merged = {**shared, **gpu_specific, "DEVICE": device_id}
            cmd = substitute_command(command_template, merged)
            check_no_unsubstituted(cmd)
            device_queues[device_id].append(cmd)

    total = sum(len(q) for q in device_queues.values())
    print(
        f"{'DRY RUN: ' if dry_run else ''}Launching {total} jobs across "
        f"{len(devices)} devices (concurrent_per_device={concurrent_per_device})"
    )
    for device_id, queue in device_queues.items():
        print(f"  GPU {device_id}: {len(queue)} jobs")
    print()

    results: dict[str, tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            device_id: executor.submit(
                run_device_queue,
                device_id,
                queue,
                dry_run,
                concurrent_per_device,
            )
            for device_id, queue in device_queues.items()
        }
        for device_id, future in futures.items():
            results[device_id] = future.result()

    print("\n--- Summary ---")
    total_success = 0
    total_fail = 0
    for device_id in devices:
        s, f = results[device_id]
        total_success += s
        total_fail += f
        status = "all passed" if f == 0 else f"{f} FAILED"
        print(f"  GPU {device_id}: {s} succeeded, {status}")
    print(f"  Total: {total_success} succeeded, {total_fail} failed")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
