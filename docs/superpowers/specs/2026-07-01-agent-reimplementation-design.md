# Agent Reimplementation Design

## Goal
Reimplement `SimpleAgent` to use the SkillRL-inspired structured prompt / JSON skill format, improving task success rate.

## Scope
- Rewrite `SimpleAgent` in `agent.py` — prompt structure, skill format, reflection output
- No changes to other modules (selector, dataset, main.py, eval flow)

## Skill Format

Skills stored in a single `skills.json` under `spg_bandit/skills/<run_id>/skills.json`:

```json
{
  "general": [
    {"skill_id": "gen_001", "title": "Systematic Exploration",
     "principle": "Search every plausible surface exactly once before revisiting.",
     "when_to_apply": "Anytime the goal object count is not yet met."}
  ],
  "task_specific": [
    {"skill_id": "pnp_001", "title": "Direct Path Planning",
     "principle": "Navigate directly to the target receptacle after acquiring the object.",
     "when_to_apply": "After picking up the object, before exploring further."}
  ]
}
```

## Prompt Structure

### System Prompt
```
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_goal}

## Retrieved Relevant Experience
{formatted_categories}
```

Each category section:
```
### {category_name}
- **{title}**: {principle}
  *Apply when*: {when_to_apply}
```

### User Message (per turn)
```
## Current Progress
Prior to this step, you have already taken N step(s).
Below are the most recent observations:
[Structued recent history]

You are now at step N and your current observation is:
{obs}

Your admissible actions are: [{admissible_actions}]

Now it's your turn to take an action.
You should first reason step-by-step within <think> </think> tags.
Then present your action within <action> </action> tags.
```

## Model Output Parsing

Extract action from `<action>command</action>` tags.
Extract reasoning from `<think>text</think>` for logging.

## Reflection

### Prompt
```
You completed the task and {outcome}.

Task: {task}
Trajectory: {trajectory}

Based on this experience, generate new skills or update existing ones.
Return JSON format:
{"actions": [
  {"type": "create", "category": "general|task_specific",
   "skill_id": "...", "title": "...", "principle": "...", "when_to_apply": "..."},
  {"type": "update", ...},
  {"type": "delete", "skill_id": "..."}
]}
If no changes needed, return {"actions": []}
```

### Processing
- Parse JSON from reflection output
- Apply changes to `skills.json`
- Reload skills list

## Files to Modify
- `spg_bandit/modules/skill_evolving/simple_agent/agent.py` — rewrite SimpleAgent
- `spg_bandit/modules/skill_evolving/base.py` — update abstract method signatures if needed

## Non-Changes
- No changes to selector, dataset, main.py, eval flow, wandb
- No RL, no SFT — only basic agent + reflection
