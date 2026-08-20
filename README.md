# SPG-Bandit

Skill-Profile-Guided Bandit 实验框架。

## 环境要求

- Python 3.10+
- ALFWorld（embodied environment）
- OpenAI 兼容的 LLM API（如 vLLM）
- W&B 账号（可选，用于实验追踪）

## 安装

### 1. 创建 conda 环境

```bash
conda create -n alfworld python=3.10
conda activate alfworld
```

### 2. 安装依赖

```bash
pip install -r spg_bandit/requirements.txt
pip install alfworld   # ALFWorld 环境
alfworld-download
```

### 3. 配置 `.env`

在项目根目录创建 `.env`：

```env
# LLM API（vLLM 或其他 OpenAI 兼容服务）
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=your-model-name

# 可选：独立的 reflection 模型（不设置则复用 LLM）
REFLECTION_BASE_URL=
REFLECTION_API_KEY=
REFLECTION_MODEL=

# W&B（可选）
wandb_key=your-wandb-key
```

### 4. 配置 SSH 隧道（远程 vLLM, 可选）

如果 LLM 部署在远程服务器：

```bash
ssh -L 8000:127.0.0.1:8000 root@connect.bjb1.seetacloud.com
```

确保 `localhost:8000` 可访问后再运行实验。

## Config 参数说明

所有配置在 `spg_bandit/config/spg.yaml` 中。完整参数：

```yaml
# 全局设置
embedding_model: all-MiniLM-L6-v2   # 任务 embedding 模型
embedding_type: local               # local / openai / ollama
embedding_cache: true               # 复用项目 cache/ 中已保存的 embedding
# embedding_cache_dir: cache        # 可选；相对项目根目录的自定义目录
# embedding_cache_save_interval: 100 # 每新增多少条后落盘一次
max_turns: 51                       # 每任务最大执行步数

# 数据集（省略时默认为 alfworld）
dataset:
  name: alfworld
  # params:                         # 可选：传给数据集 adapter 的参数
  #   data_root: /path/to/data

# Evolving（bandit）阶段
evolve:
  split: valid_seen

# Evaluation 阶段
evaluate:
  split: valid_unseen               # 评估用 held-out split

# 实验随机种子
experiment:
  seed: 42

# Selector 选择
selector: spg_bandit                # spg_bandit 或 uniform

# Agent 配置
skill_evolving:
  name: simple_agent

# SPG-Bandit 专属参数
spg_bandit:
  warmup_ratio: 0.3                 # evolve 任务总数中用于 warmup 的比例
  window_size: 20                   # sliding-window ridge 的窗口长度，≤ warmup 步数
  K: 2                              # MIRT skill 维度
  d_f: 16                           # MLP 特征维度
  alpha: 0.1                        # UCB 探索系数
  tau: 0.1                          # gap 温度参数
```

参数组合与行为：

| scenario | warmup 行为 | bandit 行为 | eval 行为 |
|---|---|---|---|
| `uniform` | 无（n_warm=0） | 均匀循环选任务 | uniform 无 reflection |
| `spg_bandit` | 各 task type 近似均衡采样，MIRT 建模 | gap-weighted UCB 选任务 | uniform 无 reflection |

数据集规模：

| split | 任务数 |
|---|---|
| `valid_seen` | 140 |
| `valid_unseen` | 134 |
| `train` | 3553 |

每个阶段会加载对应 split 的全部任务。SPG-Bandit 的 warmup 从 evolve pool 采样，并使各 task type
的抽样次数尽量均衡；同一 task 在 warmup 内最多抽取一次。步数为
`round(evolve_pool_size * warmup_ratio)`；剩余步数用于 bandit。滑动窗口以最后
`window_size` 条 warmup 观测初始化，并在 bandit 阶段逐轮淘汰最旧观测；uniform 不执行 warmup。

## 运行实验

```bash
# Uniform 基线
python spg_bandit/main.py -c uniform --evaluating

# SPG-Bandit
python spg_bandit/main.py -c spg --evaluating

# 不带 evaluation（仅跑 bandit）
python spg_bandit/main.py -c spg

# 指定 seed
python spg_bandit/main.py -c spg --seed 123

# 关闭 W&B
python spg_bandit/main.py -c spg --no-wandb
```

### 命令行参数

| 参数 | 说明 |
|---|---|
| `-c / --config` | 使用的配置文件名（不含 .yaml） |
| `--run_id` | 自定义 run ID，默认格式：selector_agent_时间戳 |
| `--no-wandb` | 不记录 W&B |
| `--seed` | 覆盖 config 中的 seed |
| `--log-file` | 同时写日志文件 |
| `--evaluating` | 跑完 bandit 后执行 evaluation 阶段 |
| `--warmup-data` | 加载已保存的 warmup 数据，跳过 warmup 执行 |

## 扩展：自定义 Skill Evolving 方法

框架支持替换不同的 skill evolving 实现，只需实现 `BaseSkillEvolving` 接口。

### 接口定义

```python
class BaseSkillEvolving(ABC):
    
    def execute(self, task_id: int) -> dict:
        """执行任务，返回 {"success": bool, "trajectory": str, "api_calls": int, ...}"""
    
    def load_skills(self, skills_dir: str):
        """从目录加载已有技能（可选）"""
    
    def reflect(self, task_id: int, result: dict):
        """执行后反思，更新技能库（可选）"""

    def finalize(self):
        """实验结束时刷新批量反思缓存（可选）"""
    
    def get_usage(self) -> dict:
        """返回 API 调用统计"""
    
    def reset(self):
        """重置状态"""
```

### 实现步骤

1. 在 `modules/skill_evolving/` 下新建目录，实现 `BaseSkillEvolving`

```
modules/skill_evolving/my_method/
  __init__.py        # export class
  agent.py           # 你的实现
```

2. 在 `modules/skill_evolving/__init__.py` 中注册：

```python
from .my_method import MyAgent  # noqa: F401
```

3. 在 config 中指定：

```yaml
skill_evolving:
  name: my_method
```

### 必要实现

`execute()` 需要返回一个 dict，至少包含：

```python
{
    "success": bool,     # 任务是否成功
    "trajectory": str,   # 执行轨迹文本
    "api_calls": int,    # 本次任务消耗的 API 调用数
}
```

> `delta`（技能变化量）由 SPG-Bandit selector 通过 MIRT profile 变动自行计算，不需要 agent 返回。

框架会自动处理数据集加载、selector 调度、日志记录和 W&B 追踪。

## 扩展数据集

数据集通过 registry 加载。selector 不依赖具体环境，只要求 adapter
提供 `TaskPool` 和统一的环境协议：

```python
from spg_bandit.modules.dataset.base import BaseDataset, TaskPool

class MyDataset(BaseDataset):
    name = "my_dataset"

    @property
    def task_pool(self) -> TaskPool:
        ...  # embeddings + metadata["goal"] + metadata["task_type"]

    def get_task_goal(self, task_id: int) -> str: ...
    def load(self): ...
    def create_env(self, task_id: int): ...

    # 如果环境遵循 Gym/Gymnasium，可直接使用 BaseDataset 的 reset_env、
    # step_env 和 close_env；否则只需重写这些方法。
    # 需要自定义提示词时重写 build_action_prompt；需要自定义 skill
    # 分类时重写 get_skill_task_type。
```

注册后即可在配置中使用：

```python
from spg_bandit.modules.dataset import register_dataset
register_dataset("my_dataset", MyDataset)
```

```yaml
dataset:
  name: my_dataset
  params:
    data_root: /path/to/data
```

也可以直接使用外部类路径：
`dataset.name: my_package.my_module:MyDataset`。如果需要 WebShop、Search
等非 Gym 环境，只需在对应 adapter 中把动作、成功判断和轨迹格式映射到
`EnvironmentState`/`EnvironmentStep`，selector 和 warmup 无需修改。

## 输出结构

```
logs/<run_id>/
  records/
    config.yaml                  # 实验配置
    <selector>_steps.jsonl       # 每步记录
    <selector>_warmup_data.json  # warmup 数据（可复用到 --warmup-data）
    evaluating_result.json       # evaluation 结果
    evaluating_steps.jsonl       # evaluation 每步记录
    comparison.json              # 汇总结果
  <selector>/
    messages/                    # SkillRL/SimpleAgent 的逐步消息
    expel/
      lifecycle.jsonl            # load/finalize 与源码 revision
      trials.jsonl               # 完整 ReAct trial、模型输出和轨迹
      retrievals.jsonl           # rules/reflections/success demos 检索结果
      experiences.jsonl          # 成功/失败 experience event stream
      task_reflections.jsonl     # New Plan 的 prompt/response/error
      reflection_memory.jsonl    # 实际写入任务记忆的 reflection
      insight_prompts.jsonl      # success/failure 与 all-success insight prompt
      insight_updates.jsonl      # 原始响应、规则操作和强度变化
      learning_steps.jsonl       # 每个外层 task 的学习事务
      errors.jsonl               # reflection/insight API 与解析错误（发生时）
      current_rules.json         # 当前带 strength 的规则列表
      prompts/                   # 每次 insight 的完整可复现 prompt
      rule_snapshots/            # 每次有效规则更新的快照
    evaluation/expel/            # 同一次运行内只读评估的独立 ExpeL 日志

skills/<run_id>/skills.json      # SkillRL SkillBank（SkillRL 配置）
skills/<run_id>/expel_state.json # ExpeL rule bank 与 experience store
```

## SkillRL adapter

Run the direct SkillRL SkillBank integration with `-c skillrl`. It reuses
the vendored SkillRL `SkillsOnlyMemory`, seed ALFWorld skills, retrieval
formatter, and skill-update prompt/parser from `resource/skillrl`. Its ALFWorld
defaults match the upstream training rollout contract: train split, eight
rollouts per selected task, two history steps, temperature 1.0, 512 output
tokens, and top-k 6 skill retrieval. Each rollout remains an individual
Bernoulli observation, while SPG performs one grouped MIRT update for the task
selection. The policy is frozen: the original SkillRL GRPO/FSDP trainer is not
run by this OpenAI-compatible environment runner.

The initial, un-evolved ALFWorld SkillBank is copied independently into each
run at `skills/<run_id>/skills.json`. Use the same bank for a Uniform control:

```bash
python spg_bandit/main.py -c skillrl --evaluating --no-wandb       # SPG-Bandit
python spg_bandit/main.py -c skillrl_uniform --evaluating --no-wandb # Uniform
```

Both configs point to `resource/skillrl/memory_data/alfworld/claude_style_skills.json`.
Set `skill_evolving.skill_bank_path` to another JSON SkillBank later without
changing the agent implementation.

## ExpeL adapter

`-c expel` runs the embedded ExpeL method with SPG-Bandit; `-c expel_uniform`
is the matching Uniform control. The runtime audits the exact ALFWorld ReAct
and reflection examples against the bundled source at `docs/ExpeL` and also
contains a compressed snapshot of the same upstream revision for server
deployments that omit `docs/`; it does not import the legacy LangChain stack.
Failed trials create task-local `New plan` reflections; successful trials are
retrieved by task similarity; global
insights are extracted from success/failure pairs and groups of successful
tasks. Rules use ExpeL's original `AGREE`/`REMOVE`/`EDIT`/`ADD` operations and
strength changes (+1, -1/-3, +1, +2).

```bash
python spg_bandit/main.py -c expel --evaluating --no-wandb
python spg_bandit/main.py -c expel_uniform --evaluating --no-wandb
```

`skill_evolving.mode: spg_online` lets SPG select one trial at a time and
injects a saved reflection when that task is selected again. Set it to
`paper_faithful` to execute ExpeL's contiguous reflection retries within one
selection. `insight_strategy: incremental` evolves rules online;
`deferred` performs insight extraction in `finalize` after gathering trials.

For SPG, `gain_measurement: mirt_transition` is the default/proposal path: the
MLP target is the selected task's immediate MIRT profile change. The old fixed
pre/post probe procedure is retained only as an explicit
`gain_measurement: probe` ablation and requires `probe_size`. Set
`spg_bandit.device: auto` (default) or `cuda` to run warmup MLP training and
the embedding-to-`a,d` ridge regression on a visible GPU. MIRT-EM itself
currently uses SciPy on CPU.
