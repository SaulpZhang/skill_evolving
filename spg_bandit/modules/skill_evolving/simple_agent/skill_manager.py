"""Skill manager: JSON-based skill bank with load/save/format/add/remove."""

import json
from pathlib import Path
from typing import Any


CATEGORY_HEADINGS = {
    "general_skills": "### General Principles",
    "task_specific": "### Pick And Place Skills",
    "common_mistakes": "### Mistakes to Avoid",
}


class SkillManager:
    """Load/save/format skills from a skills.json file (SkillRL format)."""

    def __init__(self, skills_dir: str):
        self._path = Path(skills_dir) / "skills.json"
        self.skills: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {
            "general_skills": [],
            "task_specific_skills": {},
            "common_mistakes": [],
        }

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.skills, indent=2, ensure_ascii=False))

    @property
    def count(self) -> dict:
        ts = sum(len(v) for v in self.skills.get("task_specific_skills", {}).values())
        return {
            "general": len(self.skills.get("general_skills", [])),
            "task_specific": ts,
            "common_mistakes": len(self.skills.get("common_mistakes", [])),
        }

    # ── CRUD ──────────────────────────────────────────────────────────

    def add_skill(self, skill: dict, category: str = "general") -> bool:
        """Add a skill. category='general' or a task_type key for task_specific."""
        sid = skill.get("skill_id")
        if sid and self._has_id(sid):
            return False
        if category == "general":
            self.skills.setdefault("general_skills", []).append(skill)
        elif category == "common_mistakes":
            self.skills.setdefault("common_mistakes", []).append(skill)
        else:
            self.skills.setdefault("task_specific_skills", {}).setdefault(category, []).append(skill)
        return True

    def remove_skill(self, skill_id: str) -> bool:
        for s in self.skills.get("general_skills", []):
            if s.get("skill_id") == skill_id:
                self.skills["general_skills"].remove(s)
                return True
        for tt in self.skills.get("task_specific_skills", {}).values():
            for s in tt:
                if s.get("skill_id") == skill_id:
                    tt.remove(s)
                    return True
        for s in self.skills.get("common_mistakes", []):
            if s.get("mistake_id") == skill_id:
                self.skills["common_mistakes"].remove(s)
                return True
        return False

    def _has_id(self, skill_id: str) -> bool:
        for s in self.skills.get("general_skills", []):
            if s.get("skill_id") == skill_id:
                return True
        for tt in self.skills.get("task_specific_skills", {}).values():
            for s in tt:
                if s.get("skill_id") == skill_id:
                    return True
        for s in self.skills.get("common_mistakes", []):
            if s.get("mistake_id") == skill_id:
                return True
        return False

    # ── Prompt formatting ─────────────────────────────────────────────

    def format_for_prompt(self, task_type: str = "") -> str:
        """Format all skills into the ## Retrieved Relevant Experience section."""
        sections = []

        # General skills
        gen = self.skills.get("general_skills", [])
        if gen:
            lines = ["### General Principles"]
            for s in gen:
                lines.append(f"- **{s['title']}**: {s['principle']}")
            sections.append("\n".join(lines))

        # Task-specific skills (filtered by task_type if provided)
        ts = self.skills.get("task_specific_skills", {})
        task_skills = ts.get(task_type, []) if task_type else []
        if not task_type:
            task_skills = [s for v in ts.values() for s in v]
        if task_skills:
            heading = "### Pick And Place Skills"
            if task_type:
                heading = f"### {task_type.replace('_', ' ').title()} Skills"
            lines = [heading]
            for s in task_skills:
                lines.append(f"- **{s['title']}**: {s['principle']}")
                if s.get("when_to_apply"):
                    lines.append(f"  _Apply when: {s['when_to_apply']}_")
            sections.append("\n".join(lines))

        # Common mistakes
        cm = self.skills.get("common_mistakes", [])
        if cm:
            lines = ["### Mistakes to Avoid"]
            for m in cm:
                lines.append(f"- **Don't**: {m['description']}")
                if m.get("how_to_avoid"):
                    lines.append(f"  **Instead**: {m['how_to_avoid']}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "(none)"

    def existing_titles(self) -> list[str]:
        titles = [s["title"] for s in self.skills.get("general_skills", []) if s.get("title")]
        for tt in self.skills.get("task_specific_skills", {}).values():
            titles.extend(s["title"] for s in tt if s.get("title"))
        return titles
