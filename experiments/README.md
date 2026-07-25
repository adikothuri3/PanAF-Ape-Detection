# Experiments

## What lives here

- [`experiment_log.md`](experiment_log.md) — the running research log. One entry per working
  session, newest at the bottom, with a copyable template at the top.

Generated outputs do **not** live here. They go to `artifacts/`, which is git-ignored. This
directory holds the narrative: what was tried, what happened, and what it meant.

## Why keep a log at all

Three reasons, in order of how often they matter:

1. **You will not remember.** In four weeks, "the threshold that worked" will be gone. The log is
   the only place that survives.
2. **Failures are expensive to rediscover.** The reason a log records dead ends is that an
   undocumented dead end gets walked down twice.
3. **A result you cannot explain is not a result.** The write-up at the end of Phase 1 is assembled
   from log entries. If they are thin, the write-up is speculation.

## How to log an experiment

1. Copy the template block from the top of `experiment_log.md` to the bottom of the file.
2. Fill in **objective and hypothesis before running anything.** A hypothesis written afterwards is
   just a description of the outcome.
3. Work. Paste exact commands as you run them, and exact errors as you hit them.
4. Fill in observations, results and interpretation while the run is still fresh.
5. Commit the log entry in the same commit as any code or config change it describes.

## Rules

- **Never invent a number.** Write "not measured" rather than an estimate. A blank field is honest;
  a plausible number is a fabrication that will be quoted later.
- **Paste verbatim errors**, including the traceback. Paraphrases are not debuggable.
- **Record the model variant and confidence threshold** in every entry involving inference. A
  detection count without them cannot be compared to anything.
- **Record whether the working tree was dirty.** A commit SHA from a modified tree describes code
  that did not run.
- **Record what failed.** An entry with no failures is usually an entry that was written from
  memory.
- **Link to run metadata** in `artifacts/metadata/` once runs produce it, so numbers in the log
  trace back to the run that generated them.

## Relationship to other documents

| Document | Purpose |
| --- | --- |
| `experiments/experiment_log.md` | Chronological, exhaustive, includes failures |
| `reports/phase1_writeup_template.md` | One page, synthesised, for a reader who was not there |
| `artifacts/metadata/*.json` | Machine-readable record of what each run actually did |

The log is the raw material; the write-up is the argument. Do not skip the log and write the report
from memory.
