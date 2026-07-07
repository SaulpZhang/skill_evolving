"""W&B integration helper."""

import os


def init_wandb(config: dict, run_id: str = None, run_name: str = None,
               enabled: bool = True):
    if not enabled:
        return False
    api_key = os.getenv("wandb_key")
    if not api_key:
        print("W&B: no api_key found in .env, skipping")
        return False
    try:
        import wandb
        wandb.init(
            entity="saulpzhang",
            project="spg-bandit-v1",
            id=run_id,
            name=run_name,
            config=config,
        )
        wandb.define_metric("_step_evolving", hidden=True)
        wandb.define_metric("_step_evaluating", hidden=True)
        wandb.define_metric("_step_mirt", hidden=True)
        wandb.define_metric("_step_spg", hidden=True)
        wandb.define_metric("evolving/*", step_metric="_step_evolving")
        wandb.define_metric("evaluating/*", step_metric="_step_evaluating")
        wandb.define_metric("mirt/*", step_metric="_step_mirt")
        wandb.define_metric("spg/*", step_metric="_step_spg")
        wandb.define_metric("profile/*", step_metric="_step_evolving")
        print(f"W&B: initialized (run: {run_id})")
        return True
    except Exception as e:
        print(f"W&B: init failed ({e}), skipping")
        return False


def log_metrics(metrics: dict):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception as e:
        print(f"W&B log error: {e}")


def finish_wandb():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"W&B finish error: {e}")
