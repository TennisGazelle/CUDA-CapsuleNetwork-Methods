# Documentation index

This repository has two documentation audiences:

- **Human/research context** lives here under `docs/` and in the root-level historical essay.
- **Agent change-making context** lives under `.cursor/` and is routed from `AGENTS.md`.

The split is intentional. Research explanations should remain readable without loading implementation rules, while an implementation agent should be able to reach the exact file/gotcha it needs without rereading the thesis.

## Start here by task

| Question | Document |
|---|---|
| What was this thesis actually about? | [`THESIS.md`](THESIS.md) |
| How does the surviving code fit together? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| How does the CUDA data layout/kernel mapping work? | [`CUDA_ARCHITECTURE.md`](CUDA_ARCHITECTURE.md) |
| What is a Capsule Network and how does routing work? | [`CAPSULE_NETWORKS.md`](CAPSULE_NETWORKS.md) |
| Why did Capsule Networks not become the dominant ML architecture? | [`../WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md`](../WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md) |
| What is currently suspicious or broken? | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |
| Can the original experiments be reproduced? | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) |
| What did the documentation/repository audit find? | [`REPO_AUDIT.md`](REPO_AUDIT.md) |
| What is the restoration target? | [`../SPEC.md`](../SPEC.md) |
| What can another agent implement next? | [`../PLAN.md`](../PLAN.md) |

## Primary historical sources

- Daniel Lopez, **A GPU Acceleration Method for Dynamically Routed Capsule Networks**: https://www.cse.unr.edu/~fredh/papers/conf/194-amlfdrcl/paper.pdf
- Daniel Lopez, **Evolving GPU-Accelerated Capsule Networks**: https://www.cse.unr.edu/~fredh/papers/thesis/071-lopez/thesis.pdf
- Sara Sabour, Nicholas Frosst, Geoffrey Hinton, **Dynamic Routing Between Capsules**: https://arxiv.org/abs/1710.09829

## Documentation policy

When a modernization change affects a statement here, update the statement in the same PR. Historical claims should not be silently rewritten to match modern results. Add a modern result alongside the historical result and explain the changed environment/methodology.
