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
max_turns: 51                       # 每任务最大执行步数

# Warmup 阶段
warmup:
  split: valid_seen                 # 数据集 split
  n_warm: 60                        # warmup 步数
  tasks_per_type: 10                # 每 type 取多少任务, 0=全部

# Evolving（bandit）阶段
evolve:
  split: valid_seen                 # 数据集 split
  task_types: all                   # 任务类型筛选
  tasks_per_type: 0                 # 0 = 所有可用任务

# 实验参数
experiment:
  n_bandit: 0                       # bandit 步数, 0 = 和 evolving 任务数相同
  seed: 42

# Evaluation 阶段
evaluate:
  split: valid_unseen               # 评估用 held-out split
  task_types: all
  tasks_per_type: 0

# Selector 选择
selector: spg_bandit                # spg_bandit 或 uniform

# Agent 配置
skill_evolving:
  name: simple_agent

# SPG-Bandit 专属参数
spg_bandit:
  K: 2                              # MIRT skill 维度
  d_f: 16                           # MLP 特征维度
  alpha: 0.1                        # UCB 探索系数
  tau: 0.1                          # gap 温度参数
```

参数组合与行为：

| scenario | warmup 行为 | bandit 行为 | eval 行为 |
|---|---|---|---|
| `uniform` | 无（n_warm=0） | 均匀循环选任务 | uniform 无 reflection |
| `spg_bandit` | 均匀采样，MIRT 建模 | gap-weighted UCB 选任务 | uniform 无 reflection |

数据集规模：

| split | 任务数 |
|---|---|
| `valid_seen` | 140 |
| `valid_unseen` | 134 |
| `train` | 3553 |

`tasks_per_type: 0` 表示加载对应 split 的全部任务。

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
    messages/                    # 每步 API 请求/响应 和 reflection 记录

skills/<run_id>/skills.json      # 实验过程中积累的 skill 库
```
