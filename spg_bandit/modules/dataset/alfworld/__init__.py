"""ALFWorld dataset implementation."""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from spg_bandit.modules.dataset.base import BaseDataset, TaskPool


TASK_TYPES = [
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
]

TYPE_TO_DIM = {t: i for i, t in enumerate(TASK_TYPES)}
K = len(TASK_TYPES)


_embedder = None

def _get_embedding(text: str, model: str = "all-MiniLM-L6-v2",
                   api_url: str = "", api_type: str = "local") -> list[float]:
    """Get embedding: local (sentence-transformers), OpenAI-compatible, or Ollama."""
    if api_type == "local":
        global _embedder
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(model, trust_remote_code=True)
        return _embedder.encode(text).tolist()

    data = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        api_url or "http://localhost:11434/api/embed", data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    if api_type == "openai":
        return result["data"][0]["embedding"]
    return result["embeddings"][0]


class ALFWorldDataset(BaseDataset):
    """ALFWorld dataset with configurable task filtering."""

    def __init__(self, config: dict):
        self.max_turns = config.get("max_turns", 30)
        self._task_types = config.get("task_types", TASK_TYPES)  # list or "all"
        self._tasks_per_type = config.get("tasks_per_type", 0)    # 0 = all
        self._split = config.get("split", "valid_seen")           # valid_seen/valid_unseen/train
        self._embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        self._embedding_type = config.get("embedding_type", "local")  # local / ollama / openai
        self._embedding_url = config.get("embedding_url", "")
        self._pool: TaskPool | None = None
        self._task_list: list[dict] = []

    @property
    def task_pool(self) -> TaskPool:
        if self._pool is None:
            self.load()
        return self._pool

    def get_task_goal(self, task_id: int) -> str:
        return self._task_list[task_id]["goal"]

    def create_env(self, task_id: int):
        task = self._task_list[task_id]
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        req = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [task["game_file"]], req, batch_size=1,
            asynchronous=True, max_episode_steps=self.max_turns,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id), env_id

    @staticmethod
    def close_env(env, env_id: str):
        env.close()
        try:
            import gym
            reg = gym.envs.registration.registry
            if isinstance(reg, dict) and env_id in reg:
                del reg[env_id]
            elif hasattr(reg, "env_specs") and env_id in reg.env_specs:
                del reg.env_specs[env_id]
        except Exception:
            pass

    def load(self):
        cache = Path.home() / ".cache" / "alfworld"

        types_to_include = TASK_TYPES if self._task_types == "all" else [t for t in self._task_types if t in TYPE_TO_DIM]
        type_count = {t: 0 for t in types_to_include}

        data_dir = cache / "json_2.1.1" / self._split
        if not data_dir.exists():
            print(f"  Split '{self._split}' not found, using valid_seen")
            data_dir = cache / "json_2.1.1" / "valid_seen"

        task_list = []
        for root, dirs, files in os.walk(data_dir):
            if "traj_data.json" not in files:
                continue
            game_file = os.path.join(root, "game.tw-pddl")
            if not os.path.exists(game_file):
                continue
            with open(os.path.join(root, "traj_data.json")) as f:
                data = json.load(f)
            tt = data["task_type"]
            if tt not in types_to_include:
                continue
            if self._tasks_per_type > 0 and type_count[tt] >= self._tasks_per_type:
                continue
            type_count[tt] = type_count[tt] + 1
            goal = data["turk_annotations"]["anns"][0]["task_desc"]
            task_list.append({
                "id": len(task_list),
                "game_file": game_file,
                "task_type": tt,
                "dim": TYPE_TO_DIM[tt],
                "goal": goal,
            })

        self._task_list = task_list
        print(f"ALFWorld ({self._split}): {len(task_list)} tasks loaded")
        print(f"  Generating embeddings ({self._embedding_type}: {self._embedding_model})...")
        embeddings = []
        for t in task_list:
            emb = _get_embedding(t["goal"], self._embedding_model, self._embedding_url, self._embedding_type)
            embeddings.append(emb)
        self._pool = TaskPool(
            embeddings=np.array(embeddings),
            metadata=task_list,
        )
        print(f"  TaskPool: {self._pool.M} tasks, {self._pool.d_c} dims")
