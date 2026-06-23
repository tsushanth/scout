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

# straightforward issue → goes straight to a plan
python3 scout.py ~/some/repo --issue "Crash when config file is missing; should default."

# issue with real tradeoffs → forks into approaches to pick from
python3 scout.py ~/some/repo --issue "Add caching to the metrics endpoint."
python3 scout.py ~/some/repo --issue "Add caching to the metrics endpoint." --approach 2
```

`--approach N` skips the interactive prompt (use it in non-TTY contexts, or to
re-run after seeing the fork).

### Credentials

scout uses, in order:

1. `ANTHROPIC_API_KEY` if set — a standard inference key. Best for any real use.
2. On macOS, the **Claude Code OAuth token** from the keychain as a fallback —
   your personal Claude subscription credential. Convenient for local testing
   (no key to manage), but it shares a rate-limit pool with any live Claude Code
   session, so it can 429 under contention. Use a real key for anything
   sustained.

## Why it's not just another issue→PR bot

The value isn't the diff — a machine writes diffs fine. The value is putting the
human judgment where it's cheap and high-leverage: **on the approach, before 400
lines get written**, not on a finished PR you have to reject and re-request. The
triage gate keeps it from nagging you on trivial fixes.

Built on the Anthropic API (`claude-opus-4-8`, structured outputs).
