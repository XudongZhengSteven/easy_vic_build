from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class EVBUnavailableError(RuntimeError):
    """Raised when easy_vic_build cannot be imported in the UI runtime."""


@dataclass(frozen=True)
class CaseContext:
    cases_home: str
    case_name: str


def _ensure_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


def _load_evb_symbols():
    _ensure_src_path()
    try:
        from easy_vic_build.Evb_dir_class import Evb_dir
        from easy_vic_build.build_GlobalParam import buildGlobalParam
        from easy_vic_build.warmup import warmup_VIC
    except Exception as exc:
        raise EVBUnavailableError(
            "Failed to import easy_vic_build. Install project dependencies first."
        ) from exc

    return Evb_dir, buildGlobalParam, warmup_VIC


def _build_evb_dir(context: CaseContext):
    Evb_dir, _, _ = _load_evb_symbols()
    evb_dir = Evb_dir(cases_home=context.cases_home)
    evb_dir.builddir(context.case_name)
    return evb_dir


def init_case(context: CaseContext) -> dict[str, str]:
    evb_dir = _build_evb_dir(context)
    return {
        "case_dir": str(Path(context.cases_home) / context.case_name),
        "domain_file": evb_dir.domainFile_path,
        "params_level0": evb_dir.params_dataset_level0_path,
        "params_level1": evb_dir.params_dataset_level1_path,
        "forcing_dir": evb_dir.MeteForcing_dir,
        "global_param": evb_dir.globalParam_path,
        "rvic_dir": evb_dir.RVIC_dir,
        "vic_log_dir": evb_dir.VICLog_dir,
        "vic_results_dir": evb_dir.VICResults_dir,
        "vic_states_dir": evb_dir.VICStates_dir,
    }


def build_global_param_file(
    context: CaseContext, global_param_dict: dict
) -> str:
    evb_dir = _build_evb_dir(context)
    _, buildGlobalParam, _ = _load_evb_symbols()
    buildGlobalParam(evb_dir, global_param_dict)
    return evb_dir.globalParam_path


def run_warmup(
    context: CaseContext, vic_exe_path: str, warmup_start: str, warmup_end: str
) -> str:
    evb_dir = _build_evb_dir(context)
    _, _, warmup_VIC = _load_evb_symbols()
    evb_dir.vic_exe_path = vic_exe_path
    warmup_VIC(evb_dir, [warmup_start, warmup_end])
    return evb_dir.VICStates_dir


def run_shell_command(
    command: str,
    working_dir: str,
    on_output: Callable[[str], None] | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=working_dir,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        if on_output is not None:
            on_output(line.rstrip("\n"))

    return process.wait()
