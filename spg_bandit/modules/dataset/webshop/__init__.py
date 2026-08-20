"""WebShop HTTP dataset adapter.

The adapter follows ExpeL's vendored WebShop environment protocol while keeping
the web server as an external service.  Start Princeton WebShop separately and
point ``server_url`` at it; importing this module never starts a server.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from spg_bandit.modules.dataset.base import BaseDataset, EnvironmentState, EnvironmentStep, TaskPool
from spg_bandit.modules.dataset.embedding_cache import EmbeddingCache


_SUBPAGES = {"Description", "Features", "Reviews", "Attributes"}
_BRACKETED = re.compile(r"\[([^\[\]]+)\]")


def _embedding(text: str, model: str, api_url: str, api_type: str) -> list[float]:
    # Reuse the ALFWorld embedding implementation so task vectors have the
    # exact same cache/provider semantics across benchmarks.
    from spg_bandit.modules.dataset.alfworld import _get_embedding
    return _get_embedding(text, model, api_url, api_type)


@dataclass
class _Session:
    session: str
    page_type: str = "init"
    query_string: str = ""
    page_num: int = 1
    asin: str = ""
    options: dict[str, str] = field(default_factory=dict)
    subpage: str = ""
    max_products: int = 0
    steps: int = 0
    done: bool = False
    last_action: str | None = None


class WebShopDataset(BaseDataset):
    """Fixed-task WebShop adapter compatible with SPG and skill evolution.

    It uses the same server routes and text observation representation as the
    WebShop environment vendored under ``docs/ExpeL/envs/webshop``.  The
    server must be started separately (normally on port 3000).
    """

    name = "webshop"

    def __init__(self, config: dict):
        self._server_url = str(config.get("server_url", "http://127.0.0.1:3000")).rstrip("/")
        default_file = Path(__file__).resolve().parents[4] / "docs" / "ExpeL" / "data" / "webshop" / "webshop.fixed100.json"
        self._split = str(config.get("split", "test")).lower()
        fixed_ranges = config.get("fixed_session_ranges", {}) or {}
        if not isinstance(fixed_ranges, dict):
            raise ValueError("WebShop fixed_session_ranges must be a mapping")
        split_range = fixed_ranges.get(self._split)
        if split_range is not None and not isinstance(split_range, dict):
            raise ValueError(f"WebShop range for split {self._split!r} must be a mapping")
        self._fixed_session_range = dict(split_range) if split_range is not None else None
        self._task_file = Path(config.get("task_file", default_file)).expanduser()
        self.max_turns = int(config.get("max_turns", config.get("max_steps", 15)))
        self._n_tasks = config.get("n_tasks", config.get("num_tasks", "all"))
        self._embedding_model = str(config.get("embedding_model", "all-MiniLM-L6-v2"))
        self._embedding_type = str(config.get("embedding_type", "local"))
        self._embedding_url = str(config.get("embedding_url", ""))
        self._embedding_cache_enabled = bool(config.get("embedding_cache", True))
        self._embedding_cache_dir = config.get("embedding_cache_dir")
        self._embedding_cache_save_interval = max(1, int(config.get("embedding_cache_save_interval", 100)))
        self._request_timeout = float(config.get("request_timeout", 30))
        self._pool: TaskPool | None = None
        self._tasks: list[dict[str, Any]] = []

    @property
    def task_pool(self) -> TaskPool:
        if self._pool is None:
            self.load()
        return self._pool

    def get_task_goal(self, task_id: int) -> str:
        return str(self._tasks[task_id]["goal"])

    def get_skill_task_type(self, task_id: int) -> str:
        del task_id
        return "webshop_purchase"

    def _get(self, path: str) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - import error is actionable
            raise ImportError("WebShop requires the optional 'requests' and 'beautifulsoup4' packages") from exc
        response = requests.get(f"{self._server_url}{path}", timeout=self._request_timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse_html(html: str) -> tuple[str, dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup, Comment
        except ImportError as exc:  # pragma: no cover - import error is actionable
            raise ImportError("WebShop requires the optional 'requests' and 'beautifulsoup4' packages") from exc
        soup = BeautifulSoup(html, "html.parser")
        visible = []
        for text in soup.find_all(string=True):
            if isinstance(text, Comment) or text.parent.name in {"style", "script", "head", "title", "meta", "[document]"}:
                continue
            value = str(text).strip()
            if value:
                visible.append((text, value))

        observation, option_type, options, asins = "", "", {}, []
        for text, value in visible:
            parent = text.parent
            if parent.name == "button":
                observation += f"\n[{value}] "
            elif parent.name == "label":
                selected = f"'{value}'" in html
                observation += f"[[{value}]]" if selected else f"[{value}]"
                options[value] = option_type
            elif parent.get("class") == ["product-link"]:
                observation += f"\n[{value}] "
                asins.append(value)
            else:
                observation += f"\n{value} "
                option_type = value
        info: dict[str, Any] = {}
        if options:
            info["option_types"] = options
        if asins:
            info["asins"] = asins
        values = [value for _, value in visible]
        marker = "Your score (min 0.0, max 1.0)"
        if marker in values:
            index = values.index(marker)
            if index + 1 < len(values):
                try:
                    info["reward"] = float(values[index + 1])
                    observation = f"{marker}: {values[index + 1]}"
                except ValueError:
                    pass
        observation = observation.replace("\nWebShop ", "").replace("\nInstruction: ", "").replace("[Search]\n", "[Search]")
        return observation.strip(), info

    def _observe(self, state: _Session) -> tuple[str, dict[str, Any]]:
        # WebShop's Flask routes use a single path segment for ``keywords``.
        # A literal slash (including a once-percent-encoded ``%2F``) is decoded
        # before route matching and produces a 404.  In free-form searches it
        # acts as a word separator, so normalize it before serializing the URL.
        query = quote(state.query_string.replace("/", " "), safe="")
        if state.page_type == "init":
            path = f"/{quote(state.session, safe='')}"
        elif state.page_type == "search":
            path = f"/search_results/{quote(state.session, safe='')}/{query}/{state.page_num}"
        elif state.page_type == "item":
            path = f"/item_page/{quote(state.session, safe='')}/{quote(state.asin, safe='')}/{query}/{state.page_num}/{quote(str(state.options), safe='')}"
        elif state.page_type == "item_sub":
            path = f"/item_sub_page/{quote(state.session, safe='')}/{quote(state.asin, safe='')}/{query}/{state.page_num}/{quote(state.subpage, safe='')}/{quote(str(state.options), safe='')}"
        elif state.page_type == "end":
            path = f"/done/{quote(state.session, safe='')}/{quote(state.asin, safe='')}/{quote(str(state.options), safe='')}"
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"Unknown WebShop page type: {state.page_type}")
        return self._parse_html(self._get(path))

    @staticmethod
    def _admissible(observation: str) -> list[str]:
        actions = []
        for label in _BRACKETED.findall(observation):
            if label == "Search":
                actions.append("search[query]")
            elif label and not label.startswith("["):
                actions.append(f"click[{label}]")
        return list(dict.fromkeys(actions))

    def create_env(self, task_id: int):
        return _Session(session=str(self._tasks[task_id]["session_idx"]))

    @staticmethod
    def _goal_from_fixed_observation(observation: str, session: str) -> str:
        """Extract a WebShop instruction from its fixed-session landing page."""
        match = re.search(
            r"(?:(?:WebShop\s+)?Instruction:\s*)?(.*?)\s*\[Search\]\s*$",
            observation,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(
                f"Could not parse the instruction returned for WebShop session {session!r}"
            )
        return " ".join(match.group(1).split())

    def _load_fixed_session_rows(self) -> list[dict[str, Any]]:
        """Materialize a deterministic non-overlapping range of fixed sessions.

        WebShop maps ``fixed_<index>`` to a seeded goal, enabling an explicit
        train/test partition without reusing the upstream fixed-100 benchmark.
        The resolved instructions are cached so later runs do not make the 100
        setup requests again.
        """
        assert self._fixed_session_range is not None
        try:
            start = int(self._fixed_session_range["start"])
            count = int(self._fixed_session_range["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"WebShop split {self._split!r} requires integer 'start' and 'count'"
            ) from exc
        if start < 0 or count < 1:
            raise ValueError("WebShop fixed-session range must have start >= 0 and count >= 1")

        project_root = Path(__file__).resolve().parents[4]
        manifest_dir = Path(
            self._fixed_session_range.get(
                "cache_dir", project_root / "cache" / "webshop" / "task_manifests"
            )
        ).expanduser()
        manifest_path = manifest_dir / f"{self._split}_fixed_{start}_{count}.json"
        if manifest_path.is_file():
            rows = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(rows, list) and len(rows) == count:
                return rows
            raise ValueError(f"Invalid cached WebShop manifest: {manifest_path}")

        rows = []
        for index in range(start, start + count):
            session = f"fixed_{index}"
            observation, _ = self._parse_html(self._get(f"/{session}"))
            goal = self._goal_from_fixed_observation(observation, session)
            rows.append({
                "task": f"Instruction: {goal}\n[Search]\n",
                "session_idx": session,
                "key": {},
            })

        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"WebShop ({self._split}): cached fixed-session manifest -> {manifest_path}")
        return rows

    def reset_env(self, env_handle: _Session) -> EnvironmentState:
        env_handle.__dict__.update(_Session(session=env_handle.session).__dict__)
        observation, info = self._observe(env_handle)
        env_handle.__dict__.update(info)
        return EnvironmentState(observation, self._admissible(observation), info)

    def step_env(self, env_handle: _Session, action: str) -> EnvironmentStep:
        state = env_handle
        action = str(action).strip()
        invalid = False
        observation_override = ""
        try:
            if action.startswith("think[") and action.endswith("]"):
                observation_override = "OK."
            elif action.startswith("search[") and action.endswith("]") and state.page_type == "init":
                state.query_string, state.page_num, state.page_type = action[7:-1], 1, "search"
            elif action.startswith("click[") and action.endswith("]"):
                button = action[6:-1]
                if button == "Buy Now" and state.page_type == "item":
                    state.page_type, state.done = "end", True
                elif button == "Back to Search" and state.page_type in {"search", "item", "item_sub"}:
                    state.__dict__.update(_Session(session=state.session).__dict__)
                elif button == "Next >" and state.page_type == "search" and state.page_num < max(1, math.ceil(state.max_products / 10)):
                    state.page_num += 1
                elif button == "< Prev" and state.page_type == "search" and state.page_num > 1:
                    state.page_num -= 1
                elif button == "< Prev" and state.page_type == "item_sub":
                    state.page_type = "item"
                elif button == "< Prev" and state.page_type == "item":
                    state.page_type, state.options = "search", {}
                elif button in _SUBPAGES and state.page_type == "item":
                    state.page_type, state.subpage = "item_sub", button
                elif state.page_type == "search" and button in getattr(state, "asins", []):
                    state.page_type, state.asin = "item", button
                elif state.page_type == "item" and button in getattr(state, "option_types", {}):
                    state.options[state.option_types[button]] = button
                    observation_override = f"You have clicked {button}."
                else:
                    invalid = True
            else:
                invalid = True
        except (AssertionError, KeyError):
            invalid = True

        if invalid:
            observation_override = "Invalid action!"
            if state.last_action == action and action.lower().startswith(("search[", "think[")):
                state.done, observation_override = True, "Repeated action!"
        observation, info = self._observe(state)
        if action.startswith("search[") and not invalid:
            match = re.search(r"\(Total results: (\d+)\)", observation)
            if match:
                state.max_products = int(match.group(1))
        if observation_override:
            observation = observation_override
        state.__dict__.update(info)
        state.steps += 1
        reward = float(info.get("reward", 0.0))
        success = reward >= 1.0
        if state.steps >= self.max_turns and not state.done:
            state.done = True
            observation += "\n\nRan out of steps! TASK FAILED"
        state.last_action = action
        return EnvironmentStep(observation, self._admissible(observation), info, reward, state.done or success, success)

    def load(self):
        if self._fixed_session_range is not None:
            rows = self._load_fixed_session_rows()
            source = f"fixed sessions ({self._split})"
        else:
            if not self._task_file.is_file():
                raise FileNotFoundError(f"WebShop task file not found: {self._task_file}")
            rows = json.loads(self._task_file.read_text(encoding="utf-8"))
            source = str(self._task_file)
        if not isinstance(rows, list):
            raise ValueError("WebShop task file must contain a JSON array")
        if isinstance(self._n_tasks, int) and self._n_tasks > 0:
            if self._n_tasks > len(rows):
                raise ValueError(
                    f"WebShop {self._split} split has only {len(rows)} tasks, "
                    f"but n_tasks={self._n_tasks} was requested"
                )
            rows = rows[:self._n_tasks]
        self._tasks = []
        for index, row in enumerate(rows):
            task = str(row.get("task", "")).strip()
            goal = re.sub(r"^Instruction:\s*", "", task).replace("\n[Search]", "").strip()
            if not goal or "session_idx" not in row:
                raise ValueError(f"Invalid WebShop task at index {index}")
            self._tasks.append({"id": index, "goal": goal, "task_type": "webshop_purchase", "session_idx": row["session_idx"], "key": row.get("key", {})})
        print(f"WebShop ({self._split}): {len(self._tasks)} tasks loaded from {source}")
        cache = EmbeddingCache(self._embedding_cache_dir, self.name, {"embedding_type": self._embedding_type, "embedding_model": self._embedding_model, "embedding_url": self._embedding_url}, self._embedding_cache_enabled)
        embeddings, hits, misses = [], 0, 0
        try:
            for task in self._tasks:
                vector = cache.get(task["goal"])
                if vector is None:
                    misses += 1
                    vector = _embedding(task["goal"], self._embedding_model, self._embedding_url, self._embedding_type)
                    cache.put(task["goal"], vector)
                    if misses % self._embedding_cache_save_interval == 0:
                        cache.save()
                else:
                    hits += 1
                embeddings.append(vector)
        finally:
            cache.save()
        if self._embedding_cache_enabled:
            print(f"  Embedding cache: {hits} hits, {misses} misses ({cache.size} entries) -> {cache.path}", flush=True)
        self._pool = TaskPool(np.asarray(embeddings), self._tasks)
        print(f"  TaskPool: {self._pool.M} tasks, {self._pool.d_c} dims")
