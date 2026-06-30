"""Simple skill evolving agent for ALFWorld.

After each task, reflects on the trajectory and saves skills
in Anthropic SKILL.md format under spg_bandit/skills/<run_id>/.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from spg_bandit.modules.dataset.alfworld import ALFWorldDataset
from spg_bandit.modules.skill_evolving.base import BaseSkillEvolving

load_dotenv(Path(__file__).parents[4] / ".env")

SYSTEM_PROMPT = """You are in a text-based household. Complete the task by issuing commands.

Rules:
- Output ONLY the command.
- Use EXACT object names from the observation.

you have access to the following skills to help you complete the task:
{skill_hint}"""

REFLECT_PROMPT = """You completed a task and {outcome}.

Task: {task}

Skills used: {skills_used}

Existing skills: {existing_skills}

Last actions: {trajectory}

You can learn something from the above information, especially the trace. You have four options:
- SKILL: name | description | content
- UPDATE: name | description | content
- DELETE: name
- NO CHANGE

Description must say what the skill IS and WHEN to use it.

You should only output one of the four options above. If you want to create a new skill, use SKILL. If you want to update an existing skill, use UPDATE. If you want to delete an existing skill, use DELETE. If you don't want to change anything, use NO CHANGE.

Your response:"""


class SimpleAgent(BaseSkillEvolving):
    """ALFWorld agent with ReAct loop and post-task skill evolution."""

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
        self._loaded_skill = None
        self._loaded_skills_list = []

    def load_skills(self, skills_dir: str):
        """Load skills from a directory for later use."""
        self._skills_dir = Path(skills_dir)
        self._loaded_skills_list = []
        if self._skills_dir.exists():
            for d in sorted(self._skills_dir.iterdir()):
                if d.is_dir() and (d / "SKILL.md").exists():
                    self._loaded_skills_list.append(d.name)
            if self._loaded_skills_list:
                print(f"  >>> Loaded {len(self._loaded_skills_list)} skills from {skills_dir}", flush=True)

    def get_usage(self) -> dict:
        return {"api_calls": self._total_calls}

    def reset(self):
        self._total_calls = 0
        self._loaded_skill = None

    def _chat(self, messages, max_tokens=4096):
        self._total_calls += 1
        return self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens,
            temperature=0.3,
        ).choices[0].message.content.strip()

    def _save_messages(self, task_id: int, messages: list, prefix: str = ""):
        """Save messages history as JSON."""
        if not self._records_dir:
            return
        tag = f"{prefix}task_{task_id}" if prefix else f"task_{task_id}"
        path = self._records_dir / f"{tag}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

    def _available_skills(self) -> str:
        """List available skills for system prompt."""
        if not self._skills_dir or not self._loaded_skills_list:
            return ""
        lines = ["\n\nAvailable skills:"]
        for name in self._loaded_skills_list:
            sf = self._skills_dir / name / "SKILL.md"
            if sf.exists():
                meta = sf.read_text().split("---")
                desc = ""
                if len(meta) > 1:
                    for line in meta[1].split("\n"):
                        if line.startswith("description:"):
                            desc = line.split(":", 1)[1].strip()
                lines.append(f"  {name}: {desc}")
        lines.append("\nTo use: USE SKILL: <name>")
        return "\n".join(lines)

    def _save_skill(self, name, description, content):
        if not self._skills_dir:
            return
        d = self._skills_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "scripts").mkdir(exist_ok=True)
        (d / "references").mkdir(exist_ok=True)
        (d / "assets").mkdir(exist_ok=True)
        text = f"---\nname: {name}\ndescription: {description}\n---\n\n{content.strip()}\n"
        (d / "SKILL.md").write_text(text)
        if name not in self._loaded_skills_list:
            self._loaded_skills_list.append(name)
        print(f"  >>> Skill saved: {name}", flush=True)

    def _existing_skills_str(self) -> str:
        if not self._skills_dir or not self._skills_dir.exists():
            return "(none)"
        names = sorted(d.name for d in self._skills_dir.iterdir() if d.is_dir())
        return ", ".join(names) if names else "(none)"

    def reflect(self, task_id: int, result: dict):
        """Reflect on task execution and evolve skills."""
        goal = self._dataset.get_task_goal(task_id)
        success = result["success"]
        traj = result["trajectory"]
        outcome = "succeeded" if success else "failed"
        skills_used = result.get("loaded_skill", "(none)")
        existing = self._existing_skills_str()
        prompt = REFLECT_PROMPT.format(
            outcome=outcome, task=goal, trajectory=traj,
            skills_used=skills_used, existing_skills=existing,
        )
        result_text = self._chat([{"role": "user", "content": prompt}])
        result_upper = result_text.upper().strip()

        # Save reflection messages
        self._save_messages(task_id, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": result_text},
        ], prefix="reflection_")

        if not result_text or result_upper == "NO CHANGE":
            print(f"  >>> Reflection: {'no response' if not result_text else 'no changes needed'}", flush=True)
            return

        for line in result_text.split("\n"):
            line = line.strip()
            upper = line.upper()
            if upper.startswith("SKILL:") or upper.startswith("UPDATE:"):
                prefix = "SKILL:" if upper.startswith("SKILL:") else "UPDATE:"
                rest = line[len(prefix):].strip()
                parts = rest.split("|", 2)
                if len(parts) >= 2:
                    name = parts[0].strip().replace(" ", "_")
                    desc = parts[1].strip()
                    content = parts[2].strip() if len(parts) > 2 else ""
                    if upper.startswith("SKILL:"):
                        self._save_skill(name, desc, content)
                    else:
                        self._save_skill(name, desc, content)  # same as create
                        print(f"  >>> Skill updated: {name}", flush=True)
            elif upper.startswith("DELETE:"):
                name = line[len("DELETE:"):].strip()
                if self._skills_dir:
                    d = self._skills_dir / name
                    if d.exists():
                        import shutil
                        shutil.rmtree(d)
                        print(f"  >>> Skill deleted: {name}", flush=True)

        print(f"  >>> Reflection: {result_text[:200]}", flush=True)

    def execute(self, task_id: int) -> dict:
        goal = self._dataset.get_task_goal(task_id)
        print(f"\n  --- Executing task {task_id}: {goal}", flush=True)
        if self._skills_dir:
            self.load_skills(str(self._skills_dir))
        calls_before = self._total_calls
        env, env_id = self._dataset.create_env(task_id)
        obs_tuple, info = env.reset()
        obs = obs_tuple[0]
        actions = []
        traj_lines = []

        hint = self._available_skills()
        system = SYSTEM_PROMPT.format(skill_hint=hint)
        messages = [{"role": "system", "content": system}]
        skill_injected = False

        for step in range(self.max_turns):
            cmds = info.get("admissible_commands", [])
            if cmds and isinstance(cmds[0], list):
                cmds = cmds[0]
            admissible = "; ".join(cmds) if cmds else ""

            msg = (
                f"Task: {goal}\n\n{obs}\n\n"
                f"Valid commands: {admissible}\n\n"
                "you can use the skills to assist you or give the next command to complete the task. If you want to use a skill, type 'USE SKILL: <name>' to load it. If you want give a command, type the command exactly as it appears in the valid commands list."
            )
            messages.append({"role": "user", "content": msg})
            action = self._chat(messages)
            actions.append(action)
            traj_lines.append(f"Agent: {action}")
            print(f"    [{step}] {action}", flush=True)

            # Handle USE SKILL request
            if action.upper().startswith("USE SKILL:") and self._skills_dir:
                name = action.split(":", 1)[-1].strip()
                sf = self._skills_dir / name / "SKILL.md"
                if sf.exists():
                    self._loaded_skill = name
                    content = sf.read_text()
                    messages.append({"role": "assistant", "content": action})
                    messages.append({"role": "user", "content": f"--- SKILL: {name} ---\n{content}\n---"})
                    print(f"  >>> Loaded skill: {name}", flush=True)
                    continue
                else:
                    available = []
                    if self._skills_dir and self._skills_dir.exists():
                        available = [d.name for d in self._skills_dir.iterdir() if d.is_dir()]
                    messages.append({"role": "assistant", "content": action})
                    messages.append({"role": "user", "content": f"Skill not found. Available: {', '.join(available)}"})
                    continue

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
                    traj = "\n".join(traj_lines)
                    messages.append({"role": "user", "content": f"Result: {ob}"})
                    self._save_messages(task_id, messages)
                    return {"success": True, "trajectory": traj, "actions": actions, "api_calls": self._total_calls - calls_before, "loaded_skill": self._loaded_skill}
                print(f"    -> {ob}", flush=True)
                traj_lines.append(f"Obs: {ob}")
                messages.append({"role": "assistant", "content": action})
                obs = ob
                info = info2
            except Exception:
                continue

        ALFWorldDataset.close_env(env, env_id)
        traj = "\n".join(traj_lines)
        self._save_messages(task_id, messages)
        return {"success": False, "trajectory": traj, "actions": actions, "api_calls": self._total_calls - calls_before, "loaded_skill": self._loaded_skill}
