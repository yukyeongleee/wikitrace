# Memory

WikiTrace's memory has three layers, each linked to the one below it.

| Layer | One document per | Content |
|---|---|---|
| `raw/` | training run | the run's task, outcome and full turn log |
| `compilation/` | training run | the compiler's account of the run: key decisions of a success, or likely causes of a failure |
| `insight/` | insight | one reusable guideline, at most three per run, linked to the run it came from |

## Building the memory

`wikitrace build` writes the memory of one trial to a corpus directory. Suites that share training tasks share it: both ALFWorld suites, both PlanCraft suites and both AppWorld suites. The directory name `rep-<NN>` is the trial number.

```
outputs/corpus/<benchmark>/<train_group>/rep-<NN>/
  raw/step-NNNN/                  metadata.md and the turn log of one training run
  compilation/step-NNNN.md        the compilation of that run
  compilation/compilations.jsonl  the compilation records, with each run's insights
  recommendations/<suite>.json    the three most similar training runs of each test task
  corpus.json                     benchmark, train group and build fingerprint
```

Each training run is an episode without memory. The compiler reads one finished run at a time. It sees the task, the turns and the success or failure of the run, never the gold answer.

## What the agent sees

The `wikitrace` method mounts the memory read-only at `/work/corpus` in the sandbox. The `kinj` method mounts nothing and places the same insights and recommendations in the task message.

```
/work/corpus/
  insight/_index.md         insight ids and titles
  insight/insight-NNNN.md
  compilation/_index.md
  compilation/step-NNNN.md
  raw/_index.md
  raw/step-NNNN/
```

Two read-only commands help the agent read a training run: `/work/tools/actions step-NNNN` prints every action of the run, one per line, and `/work/tools/turn step-NNNN FROM [TO]` prints whole turns.

An insight document looks like this.

```
---
type: insight
id: insight-NNNN
title: <title>
source_step: step-NNNN
---

## Insight
<guideline>

## Applicability
<when it applies>

## Source episode
step-NNNN (success | failure)
- Task: <the source run's instruction>
- Compilation: /work/corpus/compilation/step-NNNN.md
- Raw trajectory: /work/corpus/raw/step-NNNN/
```

A compilation has the sections `Task`, then `Key Decisions` (success) or `Failure Causes` (failure), then `Insights from this run`, which links back to its insight documents.

## Links

- An insight links down to the compilation and the raw directory of its source run.
- A compilation links up to the insights drawn from it.
- The task message lists the three recommended training runs and the insights drawn from each.
