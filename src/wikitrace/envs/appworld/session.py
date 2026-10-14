"""One AppWorld episode: its server in a container that holds no gold, graded on the host after."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from wikitrace.envs.appworld.benchmarks.appworld import MAX_STEPS
from wikitrace.envs.appworld.envs.appworld import EXPERIMENT_NAME
from wikitrace.harness.environment.client import (
    CONTAINER_SERVER, SERVER_PATH, Container, EnvironmentSession,
)
from wikitrace.harness.environment.protocol import EVALUATION_FILE

ENV_MODULE = Path(__file__).with_name("envs") / "appworld.py"
_REPO = Path(__file__).resolve().parents[4]
BUILD_SCRIPT = _REPO / "scripts" / "setup" / "appworld_env_image.sh"
IMAGE_REPO = "wikitrace-appworld-env"
# The image tag carries a digest of what it is built from, so an image built from other code is absent.
IMAGE_SOURCES = (
    _REPO / "docker" / "appworld-env" / "Dockerfile", BUILD_SCRIPT,
    SERVER_PATH, SERVER_PATH.with_name("protocol.py"), ENV_MODULE,
)
CONTAINER_ROOT = "/appworld"


def appworld_version(install: Path) -> str:
    """The AppWorld version installed in ``install``'s venv."""
    dist = next(Path(install).glob(".venv/lib/python*/site-packages/appworld-*.dist-info"))
    return dist.name.removeprefix("appworld-").removesuffix(".dist-info")


def image_tag(install: Path) -> str:
    """The episode image for this install: its AppWorld version and our server code's digest."""
    digest = hashlib.sha256()
    for path in IMAGE_SOURCES:
        digest.update(path.read_bytes())
    return f"{IMAGE_REPO}:{appworld_version(install)}-{digest.hexdigest()[:12]}"


@functools.cache
def require_image(image: str) -> None:
    """Raise, naming the build script, when the episode image is not on this host."""
    try:
        proc = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(f"AppWorld episodes run in docker, which is not on PATH: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"the AppWorld environment image {image!r} is not on this host; build it with "
            f"{BUILD_SCRIPT.relative_to(_REPO)}")


def _reclaim(_func: Any, path: str, _exc: Any) -> None:
    """rmtree's error handler: give the owner rwx on ``path`` and its parent, then remove ``path``."""
    if not os.path.lexists(path):
        return
    for p in (os.path.dirname(path), path):
        # chmod follows a link, which may point anywhere on the host.
        if not os.path.islink(p):
            os.chmod(p, stat.S_IRWXU)
    if os.path.isdir(path) and not os.path.islink(path):
        remove_tree(path)
    else:
        os.unlink(path)


def remove_tree(path: str | Path) -> None:
    """Remove a dir the episode's code wrote into, whatever permissions it left; absent is fine."""
    if os.path.lexists(path):
        shutil.rmtree(path, **{"onexc" if sys.version_info >= (3, 12) else "onerror": _reclaim})


class AppWorldSession(EnvironmentSession):
    """An AppWorld episode whose container sees one task without its gold."""

    def __init__(self, install: Path, task_id: str, out_dir: Path) -> None:
        self.install = Path(install).resolve()
        self.task_id = task_id
        # The container writes its state here; the host grades it with the install's data linked in.
        self.episode_root = Path(out_dir) / "appworld"
        self._graded = False
        data = self.install / "data"
        task = f"data/tasks/{task_id}"
        mounts = (
            (str(self.episode_root / "experiments" / "outputs"),
             f"{CONTAINER_ROOT}/experiments/outputs", False),
            (str(data / "api_docs"), f"{CONTAINER_ROOT}/data/api_docs", True),
            (str(data / "base_dbs"), f"{CONTAINER_ROOT}/data/base_dbs", True),
            (str(self.install / task / "specs.json"), f"{CONTAINER_ROOT}/{task}/specs.json", True),
            (str(self.install / task / "dbs"), f"{CONTAINER_ROOT}/{task}/dbs", True),
        )
        super().__init__(
            factory=f"{Path(CONTAINER_SERVER).parent / 'envs' / 'appworld.py'}:make_env",
            spec={"appworld_root": CONTAINER_ROOT, "task_id": task_id},
            out_dir=Path(out_dir), step_limit=MAX_STEPS,
            container=Container(image=image_tag(self.install), mounts=mounts,
                                env=(("APPWORLD_ROOT", CONTAINER_ROOT),)),
        )

    def __enter__(self) -> "AppWorldSession":
        assert self.container is not None
        require_image(self.container.image)
        remove_tree(self.episode_root)
        (self.episode_root / "experiments" / "outputs").mkdir(parents=True)
        try:
            super().__enter__()
        except BaseException:
            remove_tree(self.episode_root)
            raise
        return self

    def finish(self, reason: str = "harness_stopped") -> dict[str, Any]:
        """Close the episode, then grade its saved state on the host into ``evaluation``."""
        reply = super().finish(reason)
        if reply and not self._graded:
            self._graded = True
            reply["evaluation"] = self._grade(dict(reply.get("evaluation") or {}))
            (self.out_dir / EVALUATION_FILE).write_text(
                json.dumps(reply["evaluation"], ensure_ascii=False, indent=1), encoding="utf-8")
        return reply

    def _grade(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        # No saved state means the task never loaded: no report, as a crashed episode has none.
        """The server's evaluation head with AppWorld's own test report merged in."""
        saved = self.episode_root / "experiments" / "outputs" / EXPERIMENT_NAME / "tasks" / self.task_id
        if not (saved / "dbs").is_dir():
            return evaluation
        (self.episode_root / "data").symlink_to(self.install / "data")
        python = self.install / ".venv" / "bin" / "python"
        # `-P` keeps the module dir off sys.path, where it would shadow `appworld`; root is "." because AppWorld
        # refuses a change-log path holding "memory", as a `no-memory` run dir does.
        try:
            proc = subprocess.run(
                [str(python), "-P", str(ENV_MODULE), ".", self.task_id], cwd=self.episode_root,
                capture_output=True, text=True, timeout=self.request_timeout,
            )
            block = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return self._grading_failed(evaluation, f"timed out after {self.request_timeout} s")
        except (IndexError, json.JSONDecodeError):
            return self._grading_failed(evaluation, f"code {proc.returncode}\n{proc.stderr}")
        evaluation["success"] = bool(block.pop("success"))
        evaluation.update(block)
        return evaluation

    def _grading_failed(self, evaluation: dict[str, Any], why: str) -> dict[str, Any]:
        """Log why the host evaluation gave no report; the head alone is then the record."""
        with (self.out_dir / "env_server.log").open("a", encoding="utf-8") as log:
            log.write(f"host evaluation failed: {why}\n")
        return evaluation

    def __exit__(self, *exc: Any) -> None:
        try:
            super().__exit__(*exc)
        finally:
            try:
                remove_tree(self.episode_root)
            except OSError:
                pass
