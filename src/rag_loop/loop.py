import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .evals import load_dataset, run_evaluation
from .report import Verdict, decide
from .results import save_result, to_result_record

SOURCE_PATHS = ["src"]
RECORD_PATHS = ["evals/experiments.json", "evals/results"]


def load_experiments(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def run_experiment(
    experiment: dict[str, Any],
    settings: Settings | None = None,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    experiment_settings = replace(
        settings,
        chat_model=experiment.get("chat_model", settings.chat_model),
        retrieval_k=experiment.get("retrieval_k", settings.retrieval_k),
        trace_tag=experiment["name"],
    )
    evaluation = run_evaluation(
        experiment_settings,
        experiment_name=experiment["name"],
        dataset_version=dataset_version,
    )
    return {
        "name": experiment["name"],
        "change": experiment["change"],
        "acceptance": experiment.get("acceptance", {}),
        "evaluation": evaluation,
    }


def run_experiment_loop(
    experiments_path: Path = Path("evals/experiments.json"),
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    dataset_version = load_dataset().get_current_version_name()
    return [
        run_experiment(experiment, settings, dataset_version)
        for experiment in load_experiments(experiments_path)
    ]


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _commit(paths: list[str], message: str) -> None:
    _git("add", *paths)
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    _git("commit", "-m", message)


def _revert(paths: list[str]) -> None:
    _git("restore", "--worktree", "--staged", *paths)


def run_iteration(
    name: str,
    experiments_path: Path = Path("evals/experiments.json"),
    settings: Settings | None = None,
) -> Verdict:
    experiments = load_experiments(experiments_path)
    entry = next((e for e in experiments if e["name"] == name), None)
    if entry is None:
        raise ValueError(f"No entry named {name!r} in {experiments_path}")

    dataset_version = load_dataset().get_current_version_name()
    result = run_experiment(entry, settings, dataset_version)
    save_result(to_result_record(result))

    verdict = decide(name, experiments)
    entry["result"] = {
        "decision": verdict.decision,
        "reason": verdict.reason,
        "plateaued": verdict.plateaued,
        "flat": bool(verdict.comparisons)
        and all(c.delta is not None and c.delta == 0 for c in verdict.comparisons),
        "comparisons": [asdict(c) for c in verdict.comparisons],
    }
    experiments_path.write_text(json.dumps(experiments, indent=2))

    if verdict.decision == "reject":
        _revert(SOURCE_PATHS)
        _commit(RECORD_PATHS, f"{name} (rejected): {entry['change']}")
    else:
        _commit(SOURCE_PATHS + RECORD_PATHS, f"{name}: {entry['change']}")

    return verdict
