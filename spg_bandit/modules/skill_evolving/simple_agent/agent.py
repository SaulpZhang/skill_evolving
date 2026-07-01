"""Simple skill evolving agent for ALFWorld — SkillRL-inspired design.

- Structured prompt: system with Retrieved Relevant Experience, per-turn Current Progress
- Model outputs <think>reasoning</think><action>command</action>
- Skills stored in skills.json (general, task_specific, common_mistakes)
- Reflection: on failure, generate new skills via LLM
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from spg_bandit.modules.dataset.alfworld import ALFWorldDataset
from spg_bandit.modules.skill_evolving.base import BaseSkillEvolving
from spg_bandit.modules.skill_evolving.simple_agent.skill_manager import SkillManager

load_dotenv(Path(__file__).parents[4] / ".env")

SYSTEM_PROMPT = """You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_goal}

## Retrieved Relevant Experience
{skill_section}"""

USER_PROMPT = """## Current Progress

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent observations and the corresponding actions you took:
{history}

You are now at step {current_step} and your current observation is:
{obs}

Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

REFLECT_PROMPT = """Analyze the failed agent trajectory below and suggest NEW skills to add to the skill bank.

TASK: {task}
TASK TYPE: {task_type}
TRAJECTORY (last 5 steps):
{trajectory}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

Generate 1-3 NEW actionable skills that would help avoid this failure in the future.
Each skill must have: skill_id (use "dyn_001", "dyn_002" etc.), title (3-5 words), principle (1-2 sentences), when_to_apply (when to use this skill).

Return ONLY a JSON array of skills, no other text.
Example: ```json
[{{"skill_id": "dyn_001", "title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
```
"""


class SimpleAgent(BaseSkillEvolving):
    """ALFWorld agent with SkillRL-style prompt and skill evolution."""

    def __init__(self, dataset: ALFWorldDataset, max_turns: int = 30,
                 records_dir: str = None):
        self._dataset = dataset
        self.max_turns = max_turns
        self._records_dir = Path(records_dir) if records_dir else None
        if self._records_dir:
            self._records_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            timeout=120,
        )
        self._model = os.getenv("LLM_MODEL")
        self._total_calls = 0
        self._skills_dir = None
        self._skill_mgr: SkillManager | None = None
        self._loaded_skill = None

    def load_skills(self, skills_dir: str):
        """Load skills from a directory (skills.json)."""
        self._skills_dir = Path(skills_dir)
        self._skill_mgr = SkillManager(skills_dir)
        c = self._skill_mgr.count
        total = c["general"] + c["task_specific"] + c["common_mistakes"]
        if total:
            print(f"  >>> Loaded {total} skills from {skills_dir}", flush=True)

    def get_usage(self) -> dict:
        return {"api_calls": self._total_calls}

    def reset(self):
        self._total_calls = 0
        self._loaded_skill = None

    def _chat(self, messages, max_tokens=512):
        """Chat. Saves messages before API call."""
        self._total_calls += 1
        return self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens,
            temperature=0.0,
        ).choices[0].message.content.strip()

    def _save_triplets(self, task_id: int, triplets: list, result: dict):
        """Save all N triplets (system, user, assistant) + result to one JSON file."""
        if not self._records_dir:
            return
        self._records_dir.mkdir(parents=True, exist_ok=True)
        path = self._records_dir / f"task_{task_id}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump({
                "task_id": task_id,
                "success": result.get("success"),
                "api_calls": result.get("api_calls", len(triplets)),
                "result": result,
                "triplets": triplets,
            }, f, indent=2, ensure_ascii=False)

    def _parse_action(self, response: str) -> str:
        """Extract action from <action> or [action] tags."""
        for pattern in [r"<action>(.*?)</action>", r"\[action\](.*?)\[/action\]"]:
            m = re.search(pattern, response, re.DOTALL)
            if m:
                return m.group(1).strip()
        return response.strip()

    def _format_history(self, recent: list) -> str:
        """Format recent (obs, action) pairs for the history section."""
        lines = []
        for i, (obs, act) in enumerate(recent, 1):
            lines.append(f"[Observation {i}: '{obs}', Action {i}: '{act}']")
        return "\n".join(lines)

    # ── Execution ─────────────────────────────────────────────────────

    def execute(self, task_id: int) -> dict:
        goal = self._dataset.get_task_goal(task_id)
        print(f"\n  --- Executing task {task_id}: {goal}", flush=True)
        if self._skill_mgr:
            self.load_skills(str(self._skills_dir))
        calls_before = self._total_calls
        env, env_id = self._dataset.create_env(task_id)
        obs_tuple, info = env.reset()
        obs = obs_tuple[0]
        actions = []
        traj_lines = []
        recent = []  # recent (obs, action) pairs for history
        history_window = 2
        triplets = []  # collect (system, user, assistant) per step

        # Build system prompt with skills
        skill_section = self._skill_mgr.format_for_prompt() if self._skill_mgr else "(none)"
        system = SYSTEM_PROMPT.format(task_goal=goal, skill_section=skill_section)

        for step in range(self.max_turns):
            cmds = info.get("admissible_commands", [])
            if cmds and isinstance(cmds[0], list):
                cmds = cmds[0]
            admissible = "; ".join(cmds) if cmds else ""

            # Build user prompt with current progress (history embedded)
            history_text = self._format_history(recent[-history_window:]) if recent else "(none)"
            user = USER_PROMPT.format(
                step_count=step,
                current_step=step + 1,
                obs=obs,
                admissible=admissible,
                history=history_text,
            )

            # Each turn: system + fresh user message (no history accumulation)
            turn_msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            response = self._chat(turn_msgs)
            action = self._parse_action(response)
            triplets.append({
                "step": step,
                "system": system,
                "user": user,
                "assistant": response,
            })

            think_m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            reasoning = think_m.group(1).strip() if think_m else ""

            actions.append(action)
            traj_lines.append(f"Agent: {action}")
            print(f"    [{step}] {action}" + (f"  (reasoning: {reasoning[:80]})" if reasoning else ""), flush=True)

            if not action or len(action) < 3:
                continue

            try:
                obs2, r, done, info2 = env.step([action])
                ob = obs2[0]
                won = info2.get("won", [False])
                won_flag = isinstance(won, (list, tuple)) and len(won) > 0 and won[0]

                if won_flag or "You win!" in ob:
                    ALFWorldDataset.close_env(env, env_id)
                    print(f"    -> {ob}", flush=True)
                    traj_lines.append(f"Obs: {ob}")
                    res = {
                        "success": True, "trajectory": "\n".join(traj_lines),
                        "actions": actions, "reasoning": reasoning,
                        "api_calls": self._total_calls - calls_before,
                        "loaded_skill": self._loaded_skill,
                    }
                    self._save_triplets(task_id, triplets, res)
                    return res

                print(f"    -> {ob}", flush=True)
                traj_lines.append(f"Obs: {ob}")
                recent.append((obs, action))
                obs = ob
                info = info2
            except Exception:
                continue

        ALFWorldDataset.close_env(env, env_id)
        res = {
            "success": False, "trajectory": "\n".join(traj_lines),
            "actions": actions, "reasoning": reasoning,
            "api_calls": self._total_calls - calls_before,
            "loaded_skill": self._loaded_skill,
        }
        self._save_triplets(task_id, triplets, res)
        return res

    # ── Reflection (skill evolution) ───────────────────────────────────

    def reflect(self, task_id: int, result: dict):
        """Reflect on task execution and evolve skills via LLM analysis."""
        if result["success"]:
            # Success: optionally generate skills too, but for now skip
            print(f"  >>> Reflection: task succeeded, no new skills needed", flush=True)
            return

        if not self._skill_mgr:
            return

        goal = self._dataset.get_task_goal(task_id)
        traj = result.get("trajectory", "")
        traj_lines = traj.split("\n")
        last_steps = traj_lines[-10:]  # last 10 lines for context

        # Build trajectory summary (last 5 action-obs pairs)
        steps_text = "\n".join(last_steps)

        existing_titles = self._skill_mgr.existing_titles()

        prompt = REFLECT_PROMPT.format(
            task=goal,
            task_type=self._detect_task_type(goal),
            trajectory=steps_text,
            existing_titles=json.dumps(existing_titles),
        )

        response = self._chat([{"role": "user", "content": prompt}], max_tokens=2048)

        # Parse JSON array from response
        new_skills = self._parse_skills_response(response)
        if not new_skills:
            print(f"  >>> Reflection: no skills generated", flush=True)
            return

        # Determine category — try task-specific first
        task_type = self._detect_task_type(goal)
        if task_type in self._skill_mgr.skills.get("task_specific_skills", {}):
            category = task_type
        else:
            category = "general"

        added = 0
        for skill in new_skills:
            if self._skill_mgr.add_skill(skill, category):
                added += 1

        if added:
            self._skill_mgr.save()
            print(f"  >>> Reflection: added {added} new skill(s) ({category})", flush=True)
        else:
            print(f"  >>> Reflection: no new skills (duplicates)", flush=True)

    def _detect_task_type(self, goal: str) -> str:
        """Infer task category from goal string."""
        goal = goal.lower()
        if "look" in goal and "light" in goal:
            return "look_at_obj_in_light"
        if "clean" in goal:
            return "clean"
        if "heat" in goal or "microwave" in goal:
            return "heat"
        if "cool" in goal or "refrigerate" in goal or "fridge" in goal or "chill" in goal:
            return "cool"
        if "two" in goal:
            return "pick_two_obj_and_place"
        if "examine" in goal:
            return "examine"
        return "pick_and_place"

    def _parse_skills_response(self, response: str) -> list:
        """Parse JSON array from reflection LLM response."""
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                skills = json.loads(response[start:end])
                return [
                    s for s in skills
                    if all(k in s for k in ["skill_id", "title", "principle"])
                ]
        except (json.JSONDecodeError, Exception):
            pass
        return []
