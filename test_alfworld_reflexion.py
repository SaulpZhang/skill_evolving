"""Quick test: run Reflexion on 1 ALFWorld task with Gemma via Ollama native API."""

import sys, textworld, textworld.gym
sys.path.insert(0, ".")
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from spg_bandit.methods.reflexion_alfworld import ReflexionAgent

cache_dir = "/Users/saulpzhang/.cache/alfworld/json_2.1.1/valid_seen"
game_file = (
    cache_dir
    + "/pick_clean_then_place_in_recep-Ladle-None-CounterTop-8"
    + "/trial_T20190909_121908_219603/game.tw-pddl"
)
goal = "Place a clean ladle on a counter."

# Create env
wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
req = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
env_id = textworld.gym.register_games(
    [game_file], req, batch_size=1, asynchronous=True,
    max_episode_steps=30, wrappers=wrappers,
)
env = textworld.gym.make(env_id)

# Run agent (uses Ollama native API, not OpenAI-compatible)
agent = ReflexionAgent(model="gemma4-26b", max_turns=30, base_url="http://localhost:11434/v1")
print(f"Task: {goal}")
print(f"Model: {agent.model} via Ollama native API\n")

success, trajectory, _ = agent.run(env, goal)

print(f"\n{'='*60}")
print(f"Result: {'✅ SUCCESS' if success else '❌ FAIL'}")
print(f"API calls: {agent.get_usage()['api_calls']}")
print(f"Reflections stored: {len(agent.reflections)}")
if agent.reflections:
    print(f"  Last: {agent.reflections[-1]}")
print(f"\nTrajectory:\n{trajectory}")

env.close()
