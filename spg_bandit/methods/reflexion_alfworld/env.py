"""ALFWorld environment wrapper for the SPG-Bandit framework."""

import os
import json
import random
import textworld
import textworld.gym
import numpy as np
from pathlib import Path
from typing import Optional
from spg_bandit.core.interfaces import SkillEvolvingMethod


class ALFWorldMethod(SkillEvolvingMethod):
    """SkillEvolvingMethod using Reflexion on ALFWorld with Ollama Gemma."""

    def __init__(
        self,
        seed: int = 42,
        max_turns: int = 30,
        model: str = "gemma4-26b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.max_turns = max_turns
        self.rng = random.Random(seed)

        self._task_type_to_dim = {
            "pick_and_place_simple": 0,
            "look_at_obj_in_light": 1,
            "pick_clean_then_place_in_recep": 2,
            "pick_heat_then_place_in_recep": 3,
            "pick_cool_then_place_in_recep": 4,
            "pick_two_obj_and_place": 5,
        }
        self.K = len(self._task_type_to_dim)

        # Load task list from ALFWorld data
        self._task_list = self._build_task_list()

        self._agent = None
        self._model = model
        self._ollama_base_url = ollama_base_url

        # Skills directory for this method
        self._skills_dir = Path(__file__).parent / "skills"
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        # Sliding window profile
        self._success_buffer: list[list[bool]] = [[] for _ in range(self.K)]
        self._window_size = 20
        self._total_api_calls = 0

    def _build_task_list(self):
        """Build task index from ALFWorld JSON data."""
        cache = Path.home() / ".cache" / "alfworld"
        data_dir = cache / "json_2.1.1" / "valid_seen"
        task_list = []

        for root, dirs, files in os.walk(data_dir):
            if "traj_data.json" in files:
                json_path = os.path.join(root, "traj_data.json")
                game_file = os.path.join(root, "game.tw-pddl")
                if not os.path.exists(game_file):
                    continue
                with open(json_path) as f:
                    data = json.load(f)
                    task_type = data["task_type"]
                    if task_type not in self._task_type_to_dim:
                        continue
                    ann = data["turk_annotations"]["anns"][0]
                    goal = ann.get("task_desc", "")
                    task_list.append({
                        "id": len(task_list),
                        "game_file": game_file,
                        "task_type": task_type,
                        "goal": goal,
                        "dim": self._task_type_to_dim[task_type],
                    })

        print(f"ALFWorld: {len(task_list)} tasks loaded")
        return task_list

    @property
    def num_tasks(self) -> int:
        return len(self._task_list)

    def _create_agent(self):
        """Lazy create Reflexion agent."""
        from .agent import ReflexionAgent
        self._agent = ReflexionAgent(
            model=self._model,
            base_url=self._ollama_base_url,
            max_turns=self.max_turns,
        )

    def execute(self, task_id: int) -> bool:
        """Run Reflexion agent on one ALFWorld task."""
        if self._agent is None:
            self._create_agent()

        task = self._task_list[task_id]
        game_file = task["game_file"]
        goal = task["goal"]

        # Create a gym env for this specific game
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler, AlfredInfos,
        )

        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )

        env_id = textworld.gym.register_games(
            [game_file], request_infos, batch_size=1,
            asynchronous=True, max_episode_steps=self.max_turns,
            wrappers=wrappers,
        )
        gym_env = textworld.gym.make(env_id)

        success, traj, _ = self._agent.run(gym_env, goal)
        self._total_api_calls += self._agent.get_usage()["api_calls"]

        # Cleanup
        gym_env.close()
        import gym
        gym.envs.registration.registry.env_specs.pop(env_id, None)

        return success

    def update_profile(self, task_id: int, success: bool) -> None:
        task = self._task_list[task_id]
        dim = task["dim"]
        buf = self._success_buffer[dim]
        buf.append(success)
        if len(buf) > self._window_size:
            buf.pop(0)

    def get_profile(self) -> np.ndarray:
        profile = np.zeros(self.K)
        for d in range(self.K):
            buf = self._success_buffer[d]
            if buf:
                profile[d] = sum(buf) / len(buf)
        return profile

    def get_K(self) -> int:
        return self.K

    def get_usage(self) -> dict:
        return {
            "api_calls": self._total_api_calls,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
