# Upstream provenance

This embedded implementation follows
[LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL) at commit
`e41ec9a24823e7b560c561ab191441b56d9bcefc`.

The ALFWorld ReAct and reflection prompt literals in `prompts.py` are derived
from `prompts/alfworld.py` at that revision. The source snapshot is audited by
SHA-256 at runtime. ExpeL is licensed under the Apache License 2.0; the bundled
upstream license is available at `docs/ExpeL/LICENSE`.

Framework-specific adaptations are limited to the outer scheduling and
persistence seams: SPG can schedule one trial at a time (`spg_online`), while
`paper_faithful` retains contiguous reflection retries. The ExpeL actor,
task reflection, success-trajectory retrieval, insight extraction, and rule
strength semantics live entirely in this module and do not depend on
`SimpleAgent`.
