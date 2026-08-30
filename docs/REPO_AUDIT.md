# Documentation and repository audit — 2026-08-30

This audit was performed against the historical `CUDAify` lineage before the revival documentation was added. It follows the repository's shared `doc-audit` expectations: map entry points, verify code/docs alignment, identify stale/broken guidance, and produce an actionable remediation plan.

## Executive summary

The codebase contained significantly more thesis-era functionality than the default branch/README exposed. The largest documentation defect was therefore **branch truth**: a reader landing on `master` could reasonably conclude that the CUDA implementation was absent or unfinished even though the open `CUDAify` PR contained the actual CUDA network, GA, Unified Memory abstraction, database cache, and experiment tooling.

The second major issue was **research-code ambiguity**: old comments and README instructions mixed working paths, unfinished paths, assignment-specific timing instructions, known crashes, and later thesis experiments without a stable map of what produced published results.

The revival remediates discoverability and records the remaining technical gaps instead of attempting to silently repair them.

## Verified historical structure

- [x] `master` contains the older sequential/reference implementation.
- [x] open PR #3 is `CUDAify -> master` and contains the major CUDA implementation.
- [x] `CUDAify` includes `CUCapsuleNetwork`, `CUUnifiedBlob`, CUDA convolution, NSGA-II code, PostgreSQL-backed chromosome caching, SLURM/command helpers, and expanded experiment code.
- [x] open PR #5 is `smallNorbReader -> CUDAify` and adds dataset abstraction/smallNORB work.
- [x] `hpcvis3-leftover` contains a 2020 recovery commit but forks from an earlier point in CUDAify history; it is not a simple successor to the final `CUDAify` tip.
- [x] the old `CUDAify` README documents old Cubix/SLURM assumptions, manual tensor-channel recompilation, pending unit tests, and an ignorable destructor segfault.
- [x] the build uses old CMake/FindCUDA, `sm_30`, Armadillo, pthreads, and PQXX.

## Stale or misleading documentation found

### README branch/version ambiguity

**Pre-revival problem:** default-branch README described the project as if CUDA work were pending/partial and did not direct readers to the open CUDA branch.

**Remediation:** root README now states branch lineage prominently and links the paper/thesis.

### CUDAify README contradictory routing statement

The historical CUDAify README contains wording equivalent to dynamic routing being implemented in CPU and CUDA followed by a parenthetical suggesting it is not yet in CUDA. Code and thesis show the CUDA routing path exists. This is stale development-era text.

**Remediation:** preserve the old version in history; new README describes verified branch contents.

### Build instructions are environment-specific

Old instructions assume:

- CLion;
- a build directory whose spelling mattered historically;
- Cubix paths;
- old CUDA toolchain behavior;
- old CMake modules;
- local PostgreSQL/PQXX availability.

**Remediation:** new README labels these as historical assumptions. A modern build path is a planned work package rather than invented documentation.

### "Ignore the segmentation fault"

The old README explicitly says a destructor segmentation fault can be ignored for assignment speedup purposes.

This is honest but inappropriate as current contributor guidance.

**Remediation:** retain as known historical context in `KNOWN_ISSUES.md`; do not repeat it as an acceptable modern success criterion.

## Entry-point audit

### Before revival

- `AGENTS.md`: missing
- `CLAUDE.md`: missing
- `GEMINI.md`: missing
- Copilot instructions: missing
- canonical agent hub/task router: missing
- `ai-skills` integration: missing

### Revival remediation

- [x] add `AGENTS.md` hub;
- [x] add thin harness stubs;
- [x] add `.cursor/AI_QUICK_INDEX.md` task router;
- [x] add canonical `.cursor/rules/`;
- [x] pin `.agents/ai-skills` submodule;
- [x] expose shared skills through `.agents/skills` and `.claude/skills`.

## Human documentation coverage before revival

| Topic | Pre-revival state | Revival state |
|---|---|---|
| project purpose | brief/stale README | expanded README + thesis map |
| branch lineage | absent | documented |
| Capsule math | mostly paper/thesis only | `CAPSULE_NETWORKS.md` |
| CUDA layout/kernel map | paper/thesis + source only | `CUDA_ARCHITECTURE.md` |
| system architecture | source only | `ARCHITECTURE.md` |
| known correctness risks | comments/debug code | `KNOWN_ISSUES.md` |
| reproducibility policy | absent | `REPRODUCIBILITY.md` |
| why CapsNets lost adoption | absent | root long-form essay |
| target restoration state | absent | `SPEC.md` |
| agent-attackable work plan | absent | `PLAN.md` |

## Code/document mismatch flags discovered

### Test evaluation semantics

The thesis describes test evaluation as forward-only. The surviving CUDA `tally(false)` path appears to backpropagate and periodically update weights. This is a critical mismatch requiring experiment-commit reconstruction.

### Reconstruction regularizer

The architecture description in the original CapsNet literature includes reconstruction, and sequential code contains decoder machinery, but portions are commented in the surviving training path. Documentation must not imply it was active in every experiment.

### Configuration generality

`CapsNetConfig` exposes dimensions, but many downstream expressions still assume 28x28 MNIST, 6x6 capsule spatial layout, and ten classes. The code is configurable in important dimensions but not fully general.

### "CUDA speedup" interpretation

Historical numbers compare the custom CUDA path primarily with the repository's sequential implementation. They should not be described as modern optimized GPU-vs-GPU speedups.

## Link audit

The pre-revival README's external Armadillo links were historical but did not provide a modern reproducibility route. Internal documentation was too sparse for meaningful hub/spoke link checking.

The revival creates explicit internal hubs. CI adds a relative Markdown link checker so future broken links are caught automatically.

## Actionability gaps remaining after this PR

Documentation can describe the problems, but the following code gaps remain intentionally unresolved:

- [ ] exact experiment commit provenance;
- [ ] modern CMake/CUDA build path;
- [ ] CPU/GPU primitive parity suite;
- [ ] pure evaluation API;
- [ ] confirmed fix for suspected sequential backprop indexing;
- [ ] deterministic RNG plumbing;
- [ ] NSGA-II correctness fixtures;
- [ ] database schema/cache fingerprinting;
- [ ] self-hosted GPU CI or equivalent runtime environment;
- [ ] modern performance baseline;
- [ ] data/artifact provenance cleanup.

These are decomposed in root [`PLAN.md`](../PLAN.md).

## Branch reconciliation recommendations

### `smallNorbReader`

Do not merge wholesale into the revival branch merely because it is newer in one dimension. Extract as a dedicated future work package after:

1. dataset abstraction tests;
2. image-shape generalization tests;
3. bounds/ownership audit;
4. determining whether smallNORB is wanted for historical reproduction or modern research.

### `hpcvis3-leftover`

Mine selectively for:

- stream/thread experiments;
- scripts/results not present in final CUDAify;
- provenance artifacts.

Do not fast-forward it over CUDAify because its recovery commit sits on an older fork and includes massive historical data additions.

## Recommended order

1. preserve/document branch truth (this PR);
2. add host-only tests and CI (this PR foundation);
3. establish modern build configuration without changing math;
4. add CUDA runtime parity tests on a GPU environment;
5. reconstruct experiment commits and test-set behavior;
6. reproduce historical performance/accuracy separately;
7. fix confirmed numerical defects in corrected mode;
8. add modern baselines;
9. only then begin architecture/routing research.
