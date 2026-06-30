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
            project="spg-bandit",
            id=run_id,
            name=run_name,
            config=config,
            resume="allow",
        )
        print(f"W&B: initialized (run: {run_id})")
        return True
    except Exception as e:
        print(f"W&B: init failed ({e}), skipping")
        return False


def log_metrics(metrics: dict, step: int = None):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def finish_wandb():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
