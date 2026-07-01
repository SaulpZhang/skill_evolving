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

TEMPLATE_NO_HISTORY = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

TEMPLATE_WITH_MEMORY = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_goal}

## Retrieved Relevant Experience

{skill_section}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {obs}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

REFLECT_PROMPT = """Analyze the trajectory below.

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
        self._reflect_client = OpenAI(
            base_url=os.getenv("REFLECTION_BASE_URL", os.getenv("LLM_BASE_URL")),
            api_key=os.getenv("REFLECTION_API_KEY", os.getenv("LLM_API_KEY")),
            timeout=120,
        )
        self._reflect_model = os.getenv("REFLECTION_MODEL", os.getenv("LLM_MODEL"))
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

    def _chat(self, messages, max_tokens=1024, client=None, model=None):
        """Chat. Optionally override client/model (e.g. for reflection)."""
        self._total_calls += 1
        c = client or self._client
        m = model or self._model
        return c.chat.completions.create(
            model=m, messages=messages, max_tokens=max_tokens,
            temperature=0.0,
        ).choices[0].message.content.strip()

    def _save_reflection(self, task_id: int, prompt: str, response: str):
        """Save reflection API call (system=auto, user=prompt, assistant=response)."""
        if not self._records_dir:
            return
        self._records_dir.mkdir(parents=True, exist_ok=True)
        path = self._records_dir / f"reflection_{task_id}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump({
                "task_id": task_id,
                "type": "reflection",
                "user": prompt,
                "assistant": response,
            }, f, indent=2, ensure_ascii=False)

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

    @staticmethod
    def _clean_action(action: str) -> str:
        """Strip tag remnants, take only the first command if multi-action."""
        a = action.strip()
        a = re.sub(r"\s*[;,]?\s*(</action>|\[/action\]|\])[\s\S]*$", "", a)
        # Take only the first command (before any ; or ,)
        a = a.split(";")[0].split(",")[0].strip()
        return a

    def _parse_action(self, response: str) -> str:
        """Extract action from tags — handles various formats and malformed tags."""
        # 1. Complete tags
        for p in [
            r"<action>(.*?)</action>",              # <action>xxx</action>
            r"\[action\](.*?)\[/action\]",           # [action]xxx[/action]
            r"\[action>\s*(.*?)\s*\]",               # [action> xxx ]
        ]:
            m = re.search(p, response, re.DOTALL)
            if m:
                return self._clean_action(m.group(1))
        # 2. Opening tag only — take the rest
        m = re.search(r"(?:<action>|\[action>|\[action\])\s*(.*)", response, re.DOTALL)
        if m:
            return self._clean_action(m.group(1))
        # 3. Fallback: strip <think> reasoning, take first line
        clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        clean = re.sub(r"</?[a-z]+>|\[/?[a-z]+\>?|\[\/?[a-z]+\]", "", clean).strip()
        if clean:
            return clean.split("\n")[0].strip()
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
        history_window = 5
        triplets = []  # collect (system, user, assistant) per step

        # Build skill section
        skill_section = self._skill_mgr.format_for_prompt() if self._skill_mgr else ""
        has_skills = bool(skill_section and skill_section != "(none)")

        for step in range(self.max_turns):
            cmds = info.get("admissible_commands", [])
            if cmds and isinstance(cmds[0], list):
                cmds = cmds[0]
            admissible = "; ".join(cmds) if cmds else ""

            # Single user message per turn (SkillRL style — no system role)
            if step == 0 and not has_skills and not recent:
                prompt = TEMPLATE_NO_HISTORY.format(obs=obs, admissible=admissible)
            else:
                action_history = self._format_history(recent[-history_window:]) if recent else "(none)"
                prompt = TEMPLATE_WITH_MEMORY.format(
                    task_goal=goal,
                    skill_section=skill_section,
                    step_count=step,
                    history_length=min(len(recent), history_window),
                    action_history=action_history,
                    current_step=step + 1,
                    obs=obs,
                    admissible=admissible,
                )

            turn_msgs = [{"role": "user", "content": prompt}]
            response = self._chat(turn_msgs)
            action = self._parse_action(response)
            triplets.append({
                "step": step,
                "user": prompt,
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
        if not self._skill_mgr:
            return

        outcome = "succeeded" if result["success"] else "failed"
        goal = self._dataset.get_task_goal(task_id)
        traj = result.get("trajectory", "")
        steps_text = traj
        existing_titles = self._skill_mgr.existing_titles()

        prompt = REFLECT_PROMPT.format(
            outcome=outcome, task=goal,
            task_type=self._detect_task_type(goal),
            trajectory=steps_text,
            existing_titles=json.dumps(existing_titles),
        )

        response = self._chat([{"role": "user", "content": prompt}], max_tokens=128 * 1024,
                               client=self._reflect_client, model=self._reflect_model)
        self._save_reflection(task_id, prompt, response)

        # Parse reflection JSON response
        result_json = self._parse_reflection_json(response)
        if not result_json:
            print(f"  >>> Reflection: no skills generated", flush=True)
            return

        added = 0

        if result.get("success"):
            # Success: extract planning_pattern as a general skill
            pattern = result_json.get("planning_pattern", "")
            title = result_json.get("title", "")
            principle = result_json.get("principle", "")
            if pattern and title:
                skill = {"skill_id": f"dyn_{int(time.time())}", "title": title, "principle": pattern}
                if self._skill_mgr.add_skill(skill, "general"):
                    added += 1
        else:
            # Failure: extract mistakes_to_avoid
            for m in result_json.get("mistakes_to_avoid", []):
                desc = m.get("trigger_condition", "")
                fix = m.get("bad_action", "")
                if desc:
                    mistake = {
                        "mistake_id": f"mist_{int(time.time())}_{added}",
                        "description": desc,
                        "how_to_avoid": fix,
                    }
                    if self._skill_mgr.add_skill(mistake, "common_mistakes"):
                        added += 1

        if added:
            self._skill_mgr.save()
            print(f"  >>> Reflection: added {added} new item(s)", flush=True)

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

    def _parse_reflection_json(self, response: str) -> dict:
        """Parse reflection JSON: {planning_pattern} or {mistakes_to_avoid}."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, Exception):
            pass
        return {}
