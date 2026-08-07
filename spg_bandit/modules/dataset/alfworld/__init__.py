"""ALFWorld dataset implementation."""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from spg_bandit.modules.dataset.base import (
    BaseDataset, EnvironmentState, EnvironmentStep, TaskPool,
)


TASK_TYPES = [
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
]

TYPE_TO_DIM = {t: i for i, t in enumerate(TASK_TYPES)}
K = len(TASK_TYPES)


_TEMPLATE_NO_HISTORY = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

_TEMPLATE_WITH_MEMORY = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_goal}

## Retrieved Relevant Experience

{skill_section}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

_REFLECT_PROMPT = """Analyze the trajectory below.

OUTCOME: {outcome}
TASK: {task}
TASK TYPE: {task_type}
TRAJECTORY (last steps):
{trajectory}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

If SUCCESS → extract a planning_pattern (generalized execution template):
- Abstract the trajectory into a high-level logical chain using " -> ".
- NEVER use specific object names. Replace with [Object_1], [Object_2], [Location], [Target_Location].
- Return JSON: {{"planning_pattern": "Search [Location] -> Acquire [Object] -> Use [Appliance] -> Place [Target]", "title": "3-5 word title", "principle": "1-2 sentence explanation"}}

If FAILED → extract mistakes_to_avoid:
- Use abstract terms only: [Target_Object], [Container], [Location].
- Return JSON: {{"mistakes_to_avoid": [{{"trigger_condition": "abstract context", "bad_action": "abstract incorrect action"}}]}}

Return ONLY the JSON object, no other text."""


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

    name = "alfworld"

    def __init__(self, config: dict):
        self.max_turns = config.get("max_turns", 30)
        self._task_types = config.get("task_types", TASK_TYPES)  # list or "all"
        self._n_tasks = config.get("n_tasks", "all")              # "all" = all
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

    def get_skill_task_type(self, task_id: int) -> str:
        """Map ALFWorld's six raw types to SkillRL's skill-bank categories."""
        raw_type = self._task_list[task_id]["task_type"]
        return {
            "pick_and_place_simple": "pick_and_place",
            "look_at_obj_in_light": "look_at_obj_in_light",
            "pick_clean_then_place_in_recep": "clean",
            "pick_heat_then_place_in_recep": "heat",
            "pick_cool_then_place_in_recep": "cool",
            "pick_two_obj_and_place": "pick_two_obj_and_place",
        }.get(raw_type, raw_type)

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
    def _split_handle(env_handle, env_id: str | None = None):
        if env_id is not None:
            return env_handle, env_id
        if isinstance(env_handle, tuple) and len(env_handle) == 2:
            return env_handle
        return env_handle, None

    def reset_env(self, env_handle) -> EnvironmentState:
        env, _ = self._split_handle(env_handle)
        obs_tuple, info = env.reset()
        observation = obs_tuple[0] if isinstance(obs_tuple, (list, tuple)) else obs_tuple
        info = info if isinstance(info, dict) else {}
        return EnvironmentState(
            observation=observation,
            admissible_actions=self._actions_from_info(info),
            info=info,
        )

    def step_env(self, env_handle, action: str) -> EnvironmentStep:
        env, _ = self._split_handle(env_handle)
        obs, reward, done, info = env.step([action])
        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        info = info if isinstance(info, dict) else {}
        won = self._as_bool(info.get("won", False))
        success = won or "You win!" in str(observation)
        return EnvironmentStep(
            observation=observation,
            admissible_actions=self._actions_from_info(info),
            info=info,
            reward=float(reward) if isinstance(reward, (int, float, np.number)) else 0.0,
            # TextWorld's batch-size-one API may return ``[False]`` or a
            # one-element ndarray.  ``bool([False])`` is True because the
            # container is non-empty, which made every episode terminate
            # after exactly one action.  Normalize the batch value before
            # deciding whether the rollout should continue.
            done=self._as_bool(done) or success,
            success=success,
        )

    @staticmethod
    def close_env(env_handle, env_id: str | None = None):
        env, env_id = ALFWorldDataset._split_handle(env_handle, env_id)
        env.close()
        if env_id is not None:
            try:
                import gym
                reg = gym.envs.registration.registry
                if isinstance(reg, dict) and env_id in reg:
                    del reg[env_id]
                elif hasattr(reg, "env_specs") and env_id in reg.env_specs:
                    del reg.env_specs[env_id]
            except Exception:
                pass

    def build_action_prompt(
        self, *, task_goal: str, skill_section: str, observation,
        admissible_actions: list[str], step: int,
        recent: list[tuple[str, str]], history_window: int,
    ) -> str:
        admissible = "; ".join(admissible_actions)
        if step == 0 and not skill_section and not recent:
            return _TEMPLATE_NO_HISTORY.format(
                obs=observation, admissible=admissible,
            )
        action_history = "\n".join(
            f"[Observation {i}: '{obs}', Action {i}: '{action}']"
            for i, (obs, action) in enumerate(recent[-history_window:], 1)
        ) or "(none)"
        return _TEMPLATE_WITH_MEMORY.format(
            task_goal=task_goal,
            skill_section=skill_section or "(none)",
            step_count=step,
            history_length=min(len(recent), history_window),
            action_history=action_history,
            current_step=step + 1,
            obs=observation,
            admissible=admissible,
        )

    def build_reflection_prompt(
        self, *, outcome: str, task_goal: str, task_type: str,
        trajectory: str, existing_titles: list[str],
    ) -> str:
        return _REFLECT_PROMPT.format(
            outcome=outcome,
            task=task_goal,
            task_type=task_type,
            trajectory=trajectory,
            existing_titles=json.dumps(existing_titles),
        )

    def load(self):
        cache = Path.home() / ".cache" / "alfworld"

        types_to_include = TASK_TYPES if self._task_types == "all" else [t for t in self._task_types if t in TYPE_TO_DIM]

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
            if isinstance(self._n_tasks, int) and self._n_tasks > 0 and len(task_list) >= self._n_tasks:
                break
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
