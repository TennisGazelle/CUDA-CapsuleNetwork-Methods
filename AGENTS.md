# Agent onboarding

**Start here:** [`.cursor/AI_QUICK_INDEX.md`](.cursor/AI_QUICK_INDEX.md).

This repository is a historical research codebase. The most important rule is to distinguish **preservation work** from **algorithm changes**. The `CUDAify` lineage contains the implementation associated with the 2018 CUDA paper and master's thesis. Do not silently rewrite historical behavior while modernizing build, tests, documentation, or packaging.

## Harness stubs

| Harness | Entry point |
|---|---|
| Codex / cross-tool | this file |
| Cursor | [`.cursor/AI_QUICK_INDEX.md`](.cursor/AI_QUICK_INDEX.md) and [`.cursor/rules/`](.cursor/rules/) |
| Claude Code | [`CLAUDE.md`](CLAUDE.md) |
| Gemini CLI | [`GEMINI.md`](GEMINI.md) |
| GitHub Copilot | [`.github/copilot-instructions.md`](.github/copilot-instructions.md) |

All harnesses should converge on the same rules and task router. Keep stubs thin and link-first.

## Critical rules

Read [`.cursor/rules/README.md`](.cursor/rules/README.md) before making changes.

In particular:

1. **Historical preservation:** record a bug before fixing it when it may affect published results. Preserve a way to reconstruct historical behavior.
2. **Documentation sync:** behavior/build/layout changes update the corresponding docs in the same PR.
3. **Verify rather than infer:** this code contains stale comments, hard-coded dimensions, old CUDA assumptions, and experimental paths. Trace the implementation before asserting how it works.
4. **Tests before optimization:** add CPU/GPU numerical parity tests around primitives before replacing kernels or changing memory layouts.
5. **No benchmark laundering:** historical speedups must retain their original baseline/compiler context.

## Skills

Reusable skills are pinned through the [`ai-skills`](.agents/ai-skills) submodule. The committed links under [`.agents/skills/`](.agents/skills/) expose them to compatible harnesses; Claude reaches the same set through [`.claude/skills`](.claude/skills).

After updating the submodule, run:

```bash
./.agents/ai-skills/scripts/install-into-repo.sh
```

Clone with `--recurse-submodules`, or initialize after cloning with `git submodule update --init --recursive`.

## Human/research docs vs. agent docs

- Human/research context: [`README.md`](README.md), [`WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md`](WHY_CAPSULE_NETWORKS_DID_NOT_TAKE_OVER.md), and [`docs/`](docs/README.md).
- Agent change-making context: [`.cursor/`](.cursor/AI_QUICK_INDEX.md).
- Execution roadmap: [`SPEC.md`](SPEC.md) and [`PLAN.md`](PLAN.md).

## Highest-risk code paths

Before changing these, read [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md):

- `src/CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.cu`
- `src/models/CUUnifiedBlob.cu`
- `src/CapsuleNetwork/CapsuleNetwork.cpp`
- `src/GA/`
- `CMakeLists.txt`

The current modernization work intentionally does not declare published accuracy results reproduced until the exact experiment path and evaluation semantics have been reconstructed.
