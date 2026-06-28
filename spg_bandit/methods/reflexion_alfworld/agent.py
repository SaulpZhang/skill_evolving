"""Reflexion agent for ALFWorld using Ollama + OpenAI library."""

import sys
from typing import Optional
from openai import OpenAI


REACT_SYSTEM_PROMPT = """You are an AI agent in a text-based household. Complete the task by issuing commands.

Valid commands: go to X, open X, close X, take X from Y, put X in/on Y, clean X with Y, heat X with Y, cool X with Y, look, inventory.

Rules:
- Output ONLY the command. No explanations, no prefixes.
- Only use objects and receptacles that appear in the observation.
- Use the EXACT object names as shown in the observation."""

REFLECTION_PROMPT = """You attempted a task and {outcome}.

Trajectory:
{trajectory}

Write a concise reflection (max 80 words) on what went wrong and what to do differently next time."""


class ReflexionAgent:
    def __init__(self, model="gemma4-26b", base_url="http://localhost:11434/v1",
                 max_turns=30, temperature=0.3, verbose=True):
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.verbose = verbose
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=120.0)
        self.reflections: list[str] = []
        self._api_calls = 0

    def get_usage(self) -> dict:
        return {"api_calls": self._api_calls}

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Reflexion] {msg}", flush=True)

    def _chat(self, messages: list[dict], max_tokens: int = 128000) -> str:
        self._api_calls += 1
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def run(self, env, task_goal: str) -> tuple[bool, str, list]:
        self._log(f"Starting task: {task_goal}")
        obs_tuple, info = env.reset()
        obs_text = obs_tuple[0]
        self._log(f"Initial: {obs_text[:]}...")
        trajectory = []

        for step in range(self.max_turns):
            messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Task: {task_goal}\n\n{obs_text}"}]

            for r in self.reflections[-3:]:
                messages.append({"role": "user", "content": f"Previous lesson: {r}"})

            messages.append({"role": "user", "content": "What is your next command?"})

            action = self._chat(messages)
            action = action.split("\n")[0]  # take only first line
            trajectory.append(f"Agent: {action}")
            self._log(f"Step {step}: {action}")

            if not action:
                trajectory.append("Obs: empty command")
                continue

            try:
                obs2, r, done, info2 = env.step([action])
                obs_text = obs2[0]
            except Exception as e:
                self._log(f"Error: {e}")
                trajectory.append(f"Obs: Error - {e}")
                continue

            trajectory.append(f"Obs: {obs_text[:]}")
            self._log(f"Obs: {obs_text[:100]}...")

            done_flag = isinstance(done, (tuple, list)) and done[0]
            won_flag = info2.get("won", [False])
            if isinstance(won_flag, (list, tuple)):
                won_flag = won_flag[0] if won_flag else False

            if done_flag or won_flag or "You win!" in obs_text:
                self._log("SUCCESS! Generating reflection...")
                ref = self._reflect(trajectory, "succeeded")
                self.reflections.append(ref)
                self._log(f"Reflection: {ref}")
                return True, "\n".join(trajectory), trajectory

        self._log(f"FAIL after {self.max_turns} turns. Generating reflection...")
        ref = self._reflect(trajectory, f"failed after {self.max_turns} turns")
        self.reflections.append(ref)
        self._log(f"Reflection: {ref}")
        return False, "\n".join(trajectory), trajectory

    def _reflect(self, trajectory, outcome):
        traj_text = "\n".join(trajectory[-20:])[:1500]
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": REFLECTION_PROMPT.format(outcome=outcome, trajectory=traj_text)},
        ]
        self._api_calls += 1
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3, max_tokens=400,
        )
        return resp.choices[0].message.content.strip()

    def reset_memory(self):
        self.reflections = []
        self._api_calls = 0
