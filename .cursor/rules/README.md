# Agent rules (canonical)

Do not duplicate these rules into harness stubs. Link here and apply them.

| Rule | File | Apply when |
|---|---|---|
| Preserve historical research behavior | [`historical-preservation.mdc`](historical-preservation.mdc) | Any behavior, math, evaluation, or benchmark change |
| Keep docs synchronized | [`documentation-sync.mdc`](documentation-sync.mdc) | Code/build/layout/workflow changes |
| Verify ambiguous behavior | [`implementation-clarity.mdc`](implementation-clarity.mdc) | Stale comments, hard-coded paths, unclear experiment lineage |

## Repository-wide principles

- **DRY documentation:** one canonical home per fact; hubs link to spokes.
- **No assumptions:** source code and published artifacts outrank comments and memory.
- **Preservation before repair:** suspected thesis-era defects are evidence. Record them before altering them.
- **Tests are the migration boundary:** modernization is safe only when historical and corrected behavior can be distinguished numerically.
- **Documentation changes with code:** a PR that changes behavior without updating the relevant spoke is incomplete.
- **Do not claim GPU runtime validation from compilation alone:** standard CI runners do not provide an NVIDIA GPU.
- **Do not rewrite the thesis:** historical claims stay historically contextualized, including baseline/compiler limitations.
