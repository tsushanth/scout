#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
scout — approach-first issue resolver.

Commodity issue→PR bots jump straight to a diff. scout inserts the decision
that actually needs a human: the *approach*. Given a repo + an issue, it

  1. gathers lightweight repo context,
  2. triages whether the issue has one obvious approach or several
     meaningfully-different ones (the decision-tree gate),
  3. if several, presents distinct approaches with real, codebase-grounded
     tradeoffs for you to pick — instead of wastefully opening a PR per
     approach; if one, proceeds straight to it,
  4. emits the chosen approach as a layered implementation plan
     (approach → files it touches → what each does).

Generating the diff/PR is deliberately out of scope for this first cut — the
approach front-end is the differentiated part to prove first.

Usage:
    export ANTHROPIC_API_KEY=...
    python3 scout.py <repo-dir> --issue "the issue text"
    python3 scout.py <repo-dir> --issue-file issue.txt
    python3 scout.py <repo-dir> --issue "..." --approach 2   # pick after a triage fork
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import anthropic
    from pydantic import BaseModel, Field
except ImportError:
    sys.exit("Missing deps. Run: pip install -r requirements.txt")

MODEL = "claude-opus-4-8"

# Directories that never carry signal worth spending context on.
IGNORE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    "target", ".next", ".turbo", "vendor", ".idea", ".vscode", "coverage",
    ".pytest_cache", ".mypy_cache", "out", ".cache",
}
CODE_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".php", ".cs", ".swift", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".scala", ".sql", ".sh", ".toml", ".yaml", ".yml", ".json",
}
MAX_TREE_ENTRIES = 400
MAX_RELEVANT_FILES = 8
MAX_FILE_CHARS = 6000


# ---- structured-output schemas (validated by the SDK; the model retries on mismatch)

class Triage(BaseModel):
    needs_approach_choice: bool = Field(
        description="True only if the issue admits 2+ meaningfully different "
        "approaches with real tradeoffs a human should weigh. False for a "
        "typo, an obvious one-line fix, or where any competent solution "
        "converges on the same shape."
    )
    reasoning: str = Field(
        description="One or two sentences: why this is a single obvious "
        "approach, or what the axis of genuine disagreement is."
    )
    single_approach_summary: str = Field(
        description="When needs_approach_choice is False, a short description "
        "of the one approach to take. Empty string otherwise."
    )


class Approach(BaseModel):
    name: str = Field(description="Short label, e.g. 'Minimal patch' or 'Refactor the resolver'.")
    summary: str = Field(description="What this approach does, in 1-3 sentences.")
    tradeoffs: str = Field(description="The concrete cost/benefit vs. the other approaches — not generic.")
    codebase_fit: str = Field(description="How it fits the existing design/patterns in THIS repo.")
    files_to_touch: list[str] = Field(description="Repo-relative paths this approach would change or add.")


class ApproachSet(BaseModel):
    approaches: list[Approach] = Field(description="2-3 genuinely distinct approaches.")


class FileChange(BaseModel):
    path: str = Field(description="Repo-relative path.")
    what_it_does: str = Field(description="Black-box: what this file's change accomplishes, not line detail.")
    how_tested: str = Field(description="How the change would be verified (existing test, new test, manual).")


class Plan(BaseModel):
    approach_recap: str = Field(description="One-paragraph statement of the chosen approach.")
    files: list[FileChange] = Field(description="The files the change touches, each as a black box.")
    risks: str = Field(description="What could go wrong / what the reviewer should poke at.")


# ---- context gathering


def build_tree(root: Path) -> str:
    lines: list[str] = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.startswith("."))
        rel = Path(dirpath).relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        if depth > 4:
            dirnames[:] = []
            continue
        for f in sorted(filenames):
            if count >= MAX_TREE_ENTRIES:
                lines.append("  ... (tree truncated)")
                return "\n".join(lines)
            p = rel / f if rel != Path(".") else Path(f)
            lines.append(f"  {p}")
            count += 1
    return "\n".join(lines)


def read_capped(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:MAX_FILE_CHARS] + ("\n... (truncated)" if len(text) > MAX_FILE_CHARS else "")


def keywords(issue: str) -> list[str]:
    stop = {
        "the", "and", "for", "that", "this", "with", "when", "from", "have",
        "should", "would", "could", "into", "what", "which", "while", "does",
        "doesn", "isn", "are", "not", "but", "you", "your", "our", "its",
        "issue", "bug", "fix", "error", "fails", "failing", "broken",
    }
    words = "".join(c.lower() if c.isalnum() else " " for c in issue).split()
    seen: dict[str, int] = {}
    for w in words:
        if len(w) >= 4 and w not in stop:
            seen[w] = seen.get(w, 0) + 1
    return [w for w, _ in sorted(seen.items(), key=lambda kv: -kv[1])][:12]


def relevant_files(root: Path, issue: str) -> list[tuple[str, str]]:
    kws = keywords(issue)
    scored: list[tuple[int, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() not in CODE_EXTS:
                continue
            try:
                if p.stat().st_size > 200_000:
                    continue
                body = p.read_text(encoding="utf-8", errors="replace").lower()
            except Exception:
                continue
            name_hits = sum(2 for k in kws if k in f.lower())
            body_hits = sum(1 for k in kws if k in body)
            score = name_hits + body_hits
            if score > 0:
                scored.append((score, p))
    scored.sort(key=lambda t: -t[0])
    out: list[tuple[str, str]] = []
    for _, p in scored[:MAX_RELEVANT_FILES]:
        out.append((str(p.relative_to(root)), read_capped(p)))
    return out


def gather_context(root: Path, issue: str) -> str:
    parts = [f"# Repo: {root.name}\n", "## File tree\n", build_tree(root), "\n"]
    for cand in ("README.md", "readme.md", "README.rst", "README"):
        rp = root / cand
        if rp.exists():
            parts += ["## README\n", read_capped(rp), "\n"]
            break
    files = relevant_files(root, issue)
    if files:
        parts.append("## Files most relevant to the issue\n")
        for path, body in files:
            parts += [f"\n### {path}\n```\n", body, "\n```\n"]
    return "".join(parts)


# ---- LLM steps

def claude_code_oauth_token() -> str | None:
    """Read the Claude Code OAuth access token from the macOS keychain.

    This is your personal Claude subscription credential (scoped for Claude
    Code). It works for inference, but draws from the same rate-limit pool as a
    live Claude Code session — so it can 429 under contention. Fine for the odd
    local test; use a real API key for anything sustained.
    """
    if sys.platform != "darwin":
        return None
    try:
        raw = subprocess.check_output(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        d = json.loads(raw)
        return (d.get("claudeAiOauth") or d).get("accessToken")
    except Exception:
        return None


def make_client() -> "anthropic.Anthropic":
    """Prefer ANTHROPIC_API_KEY; fall back to the Claude Code OAuth token."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(max_retries=4)
    tok = claude_code_oauth_token()
    if tok:
        print("No ANTHROPIC_API_KEY — using Claude Code OAuth token "
              "(shared rate-limit pool; may throttle).", file=sys.stderr)
        return anthropic.Anthropic(
            auth_token=tok,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
            max_retries=4,
        )
    sys.exit("No credential: set ANTHROPIC_API_KEY or sign in to Claude Code.")


SYSTEM = (
    "You are a senior engineer triaging an issue against a specific codebase. "
    "You reason about the existing design before proposing change. You are "
    "blunt about tradeoffs and never pad. The codebase context you are given "
    "is partial (a tree plus the most relevant files), so reason from patterns "
    "you can see and say when something needs confirmation."
)


def parse(client, schema, instruction: str, context: str, issue: str):
    return client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"=== ISSUE ===\n{issue}\n\n"
                f"=== CODEBASE CONTEXT ===\n{context}"
            ),
        }],
        output_format=schema,
    ).parsed_output


def triage(client, context, issue) -> Triage:
    return parse(
        client, Triage,
        "Decide whether this issue needs an approach discussion. Most issues "
        "have one obvious approach — say so and don't manufacture choices. "
        "Only set needs_approach_choice=true when there is a real fork with "
        "tradeoffs a human would want to steer (e.g. quick patch vs. proper "
        "refactor, library A vs. B, narrow vs. comprehensive fix).",
        context, issue,
    )


def approaches(client, context, issue) -> ApproachSet:
    return parse(
        client, ApproachSet,
        "Generate 2-3 GENUINELY DIFFERENT approaches to this issue. They must "
        "differ in strategy, not in trivial detail. For each, give the real "
        "tradeoff against the others and how it fits this repo's existing "
        "patterns. Order from least to most invasive.",
        context, issue,
    )


def plan(client, context, issue, chosen: str) -> Plan:
    return parse(
        client, Plan,
        "Produce a layered implementation plan for the chosen approach below. "
        "Describe each file as a black box (what its change accomplishes and "
        "how it's tested), not line-by-line. Then state the risks a reviewer "
        f"should poke at.\n\n=== CHOSEN APPROACH ===\n{chosen}",
        context, issue,
    )


# ---- presentation

def hr(label=""):
    bar = "─" * 70
    return f"\n{bar}\n{label}\n{bar}" if label else f"\n{bar}"


def show_approaches(aset: ApproachSet):
    print(hr("APPROACHES — pick one (this is the human decision)"))
    for i, a in enumerate(aset.approaches, 1):
        print(f"\n[{i}] {a.name}")
        print(f"    {a.summary}")
        print(f"    tradeoff : {a.tradeoffs}")
        print(f"    fit      : {a.codebase_fit}")
        print(f"    touches  : {', '.join(a.files_to_touch) or '(unspecified)'}")


def show_plan(p: Plan):
    print(hr("IMPLEMENTATION PLAN (approach → files → diff comes next)"))
    print(f"\n{p.approach_recap}\n")
    for fc in p.files:
        print(f"  • {fc.path}")
        print(f"      does   : {fc.what_it_does}")
        print(f"      tested : {fc.how_tested}")
    print(f"\n  risks: {p.risks}")


# ---- main

def main():
    ap = argparse.ArgumentParser(description="Approach-first issue resolver.")
    ap.add_argument("repo", help="Path to the repo to work on.")
    ap.add_argument("--issue", help="Issue text.")
    ap.add_argument("--issue-file", help="File containing the issue text.")
    ap.add_argument("--approach", type=int, help="Pick approach N after a triage fork.")
    args = ap.parse_args()

    root = Path(args.repo).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    if args.issue_file:
        issue = Path(args.issue_file).read_text(encoding="utf-8")
    elif args.issue:
        issue = args.issue
    else:
        sys.exit("Provide --issue or --issue-file.")
    client = make_client()

    print("Reading the codebase...", file=sys.stderr)
    context = gather_context(root, issue)

    print("Triaging the approach...", file=sys.stderr)
    t = triage(client, context, issue)

    print(hr("TRIAGE"))
    print(f"\nNeeds approach choice: {t.needs_approach_choice}")
    print(f"Reasoning: {t.reasoning}")

    if not t.needs_approach_choice:
        print(f"\nOne obvious approach: {t.single_approach_summary}")
        print("\nProceeding straight to the plan (no approach fork needed).", file=sys.stderr)
        show_plan(plan(client, context, issue, t.single_approach_summary))
        return

    print("Generating distinct approaches...", file=sys.stderr)
    aset = approaches(client, context, issue)
    show_approaches(aset)

    n = args.approach
    if n is None and sys.stdin.isatty():
        try:
            n = int(input(f"\nPick an approach [1-{len(aset.approaches)}]: ").strip())
        except (ValueError, EOFError):
            n = None
    if n is None:
        print(f"\nRe-run with --approach N (1-{len(aset.approaches)}) to get the plan for one.")
        return
    if not (1 <= n <= len(aset.approaches)):
        sys.exit(f"--approach must be 1..{len(aset.approaches)}")

    chosen = aset.approaches[n - 1]
    print(f"\nChosen: [{n}] {chosen.name}", file=sys.stderr)
    chosen_desc = f"{chosen.name}: {chosen.summary}\nTradeoff: {chosen.tradeoffs}\nFit: {chosen.codebase_fit}"
    show_plan(plan(client, context, issue, chosen_desc))


if __name__ == "__main__":
    main()
