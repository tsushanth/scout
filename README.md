# scout

**Approach-first issue resolver.** Commodity issue→PR bots jump straight to a
diff. scout inserts the one decision that actually needs a human — the
*approach* — and gates it behind a decision tree so it only asks when there's a
real choice to make.

Given a repo and an issue, scout:

1. gathers lightweight repo context (tree + README + the files most relevant to
   the issue),
2. **triages** whether the issue has one obvious approach or several
   meaningfully-different ones,
3. if several, presents 2–3 **distinct approaches** with real,
   codebase-grounded tradeoffs for you to pick — instead of wastefully opening
   a PR per approach; if one, proceeds straight to it,
4. emits the chosen approach as a **layered plan**: approach → each file as a
   black box (what it does, how it's tested) → risks.

The actual diff/PR generation is deliberately out of scope for this first cut.
The approach front-end is the differentiated part, and the part to prove first:
does the triage correctly tell "obvious fix" from "real fork," and are the
approaches genuinely distinct with real tradeoffs?

## Use

```sh
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# Remote mode: point it straight at a GitHub issue. scout fetches the issue
# (title/body/comments) and shallow-clones the repo into a temp dir.
python3 scout.py --issue-url https://github.com/owner/repo/issues/123
python3 scout.py --issue-url https://github.com/owner/repo/issues/123 --approach 2

# Local mode: a repo path + the issue text.
# straightforward issue → goes straight to a plan
python3 scout.py ~/some/repo --issue "Crash when config file is missing; should default."

# issue with real tradeoffs → forks into approaches to pick from
python3 scout.py ~/some/repo --issue "Add caching to the metrics endpoint."
python3 scout.py ~/some/repo --issue "Add caching to the metrics endpoint." --approach 2
```

Remote mode needs the [`gh`](https://cli.github.com) CLI authenticated
(`gh auth status`). The shallow clone is deleted when scout exits.

### Scan mode — rank a repo's open-issue queue

```sh
python3 scout.py scan https://github.com/owner/repo --limit 10
python3 scout.py scan https://github.com/owner/repo --limit 20 --model claude-haiku-4-5
```

`scan` clones the repo once, then triages N open issues concurrently into a
structured verdict — `{fixable, confidence, effort, claimed, blocker_type}` —
and ranks **fixable + unclaimed** to the top. It's a pre-filter to point a
contributor (or an execution agent) at the issues worth working, and a far
better signal than `age + comment-count + keyword` heuristics: it reads the
actual code and issue thread and separates "fixable with a clear path" from
`false_premise`, `generated_code` (fix upstream), `needs_design`,
`needs_maintainer`, `out_of_scope`, and already-`claimed`.

```text
       #  verdict   conf  effort claim blocker          one-liner
  #18283  ✓ GO      0.80  small  no                     Clear self-contained fix: wrap tool.execute ...
  #18256  ✓ GO      0.72  medium no                     Concrete, well-localized bug in save-queue ...
  #18347  · skip    0.72  large  yes   needs_design     Already assigned; broad design change ...
```

`--model` lets you run the bulk pre-filter on a cheaper model and reserve
`claude-opus-4-8` (the default) for the deep `--issue-url` pass on a pick.

### Execute mode — hand the plan to Claude Code (stops before PR)

```sh
python3 scout.py --issue-url https://github.com/owner/repo/issues/123 --execute
# triage -> [approach] -> plan -> claude --print writes the change -> diff
```

With `--execute`, after the plan scout creates a `scout/issue-N` branch in the
clone and runs `claude --print` inside it (`--allowedTools Edit,Write,Bash`) to
implement the change and run tests, then stages everything and prints the diff.
**It stops there — nothing is pushed and no PR is opened.** scout writes and
stages the code; you review the diff, then push to your fork and open the PR by
hand, where your pre-flight rules apply (existing/parallel PRs, the repo's base
branch, no AI attribution). The clone is kept so you can `cd` in and push
(`--no-keep` deletes it instead).

`--execute` requires `--issue-url` (it works on a throwaway clone, never your
local repo). The executor is `claude` by default; override with
`SCOUT_EXECUTOR`.

### Find mode — discover latent issues, write OVERVIEW.md

Where `scan`/`--issue-url` *consume* reported issues, `find` *produces* them:
it audits the code itself for issues nobody has filed yet.

```sh
python3 scout.py find ~/some/repo                 # writes OVERVIEW.md into the repo
python3 scout.py find https://github.com/owner/repo  # clones; writes ./OVERVIEW-<repo>.md
```

Pipeline: **comprehend** (map modules/areas/test-shape) → **discover** (two
lenses — correctness bugs *and* engineering health) → **verify** (an
adversarial refute pass on every candidate, concurrent — the gate that keeps it
from being a slop generator) → write an `OVERVIEW.md` with the code map plus the
*verified* findings, each ranked by severity with `file` evidence links and a
one-line proposed fix.

Any finding then hands to the resolver: `scout.py <repo> --issue "<finding>"` →
triage → approaches → plan → execute. So the loop closes — **find** discovers,
the **overview** presents, **scout** resolves.

Knobs: `--per-lens N` (candidates per lens), `--max-files N` (source files
sampled into the digest), `--workers N` (concurrent verification), `--model`,
`--subscription`. Large files are sampled, not read whole — per-file chunking is
the next step for very large modules.

`--approach N` skips the interactive prompt (use it in non-TTY contexts, or to
re-run after seeing the fork).

### Credentials

scout uses, in order:

1. `ANTHROPIC_API_KEY` if set — a standard inference key.
2. Otherwise, on macOS, the **Claude Code subscription** (OAuth token from the
   keychain) — no key to manage.

Pass `--subscription` (or set `SCOUT_SUBSCRIPTION=1`) to force the subscription
and **never use an API key even if one is set** — so the planning calls run on
the same Claude Code subscription as the `--execute` step. The subscription
shares a rate-limit pool with any live Claude Code session, so run it when
nothing else is competing, and drop `scan --workers` if you hit 429s.

## Why it's not just another issue→PR bot

The value isn't the diff — a machine writes diffs fine. The value is putting the
human judgment where it's cheap and high-leverage: **on the approach, before 400
lines get written**, not on a finished PR you have to reject and re-request. The
triage gate keeps it from nagging you on trivial fixes.

Built on the Anthropic API (`claude-opus-4-8`, structured outputs).
