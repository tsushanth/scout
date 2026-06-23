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
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

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


class ScanVerdict(BaseModel):
    """A fast, structured pre-filter verdict for ranking an issue queue."""
    fixable: bool = Field(
        description="True if there's a clear, self-contained fix path a "
        "contributor could land without a maintainer decision first."
    )
    confidence: float = Field(description="0.0-1.0 confidence in the fixable call.")
    effort: Literal["small", "medium", "large"] = Field(
        description="Rough size of the change: small (a few lines/one file), "
        "medium (a few files), large (cross-cutting)."
    )
    claimed: bool = Field(
        description="True if someone already appears to be working on it "
        "(assignee, linked PR, or an 'I'll take this' comment)."
    )
    blocker_type: Literal[
        "none", "needs_design", "false_premise", "generated_code",
        "needs_maintainer", "too_vague", "out_of_scope", "already_fixed",
    ] = Field(description="Why it's NOT cleanly fixable; 'none' when fixable.")
    one_line: str = Field(description="One-line rationale for the verdict.")


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

# ---- remote mode: fetch a GitHub issue + its repo via `gh`

ISSUE_URL_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)")
MAX_COMMENTS = 6
MAX_COMMENT_CHARS = 1200


def parse_issue_url(url: str) -> tuple[str, str, int]:
    m = ISSUE_URL_RE.match(url.strip())
    if not m:
        sys.exit(f"Not a github issue URL (need .../owner/repo/issues/N): {url}")
    return m.group(1), m.group(2), int(m.group(3))


def fetch_issue_text(owner: str, repo: str, number: int) -> str:
    """Render the issue (title/state/labels/body + top comments) as one blob.

    Comments matter for the pre-filter: maintainer guidance and 'I'll take
    this' claims live there, and they're exactly what tells fixable-with-a-
    clear-path apart from unclaimed-but-murky.
    """
    try:
        out = subprocess.check_output(
            ["gh", "issue", "view", str(number), "--repo", f"{owner}/{repo}",
             "--json", "title,body,state,labels,comments"],
            text=True, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"gh issue view failed: {e.stderr.strip() or e}")
    d = json.loads(out)
    parts = [
        f"TITLE: {d.get('title', '')}",
        f"STATE: {d.get('state', '')}",
        f"LABELS: {', '.join(l.get('name', '') for l in d.get('labels', []))}",
        "", d.get("body") or "(no body)",
    ]
    comments = d.get("comments") or []
    if comments:
        parts.append("\n--- COMMENTS ---")
        for c in comments[:MAX_COMMENTS]:
            who = (c.get("author") or {}).get("login", "?")
            parts.append(f"\n[{who}]: {(c.get('body') or '')[:MAX_COMMENT_CHARS]}")
        if len(comments) > MAX_COMMENTS:
            parts.append(f"\n... (+{len(comments) - MAX_COMMENTS} more comments)")
    return "\n".join(parts)


def clone_repo(owner: str, repo: str, dest: str) -> None:
    """Shallow-clone the default branch (auth via gh, so private repos work)."""
    try:
        subprocess.check_call(
            ["gh", "repo", "clone", f"{owner}/{repo}", dest, "--", "--depth=1", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"clone failed for {owner}/{repo}: {e}")


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


def parse(client, schema, instruction: str, context: str, issue: str, model: str = MODEL):
    return client.messages.parse(
        model=model,
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


def scan_triage(client, context, issue, model: str = MODEL) -> ScanVerdict:
    return parse(
        client, ScanVerdict,
        "You are pre-filtering an issue queue to find good contribution "
        "targets. Give a fast structured verdict: is there a clear, "
        "self-contained fix path a contributor could land WITHOUT a maintainer "
        "decision first? Read the actual code context before deciding. Mark "
        "fixable=false (with the right blocker_type) for false premises, "
        "auto-generated code that must change upstream, vague reports, design "
        "questions, or things already fixed. Detect if it's already claimed.",
        context, issue, model=model,
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

def resolve_repo_and_issue(args) -> tuple[Path, str, str | None]:
    """Return (repo_root, issue_text, tmpdir_to_clean).

    Remote mode (--issue-url): fetch the issue via gh and shallow-clone the
    repo into a temp dir. Local mode: a repo path + --issue/--issue-file.
    """
    if args.issue_url:
        owner, repo, number = parse_issue_url(args.issue_url)
        print(f"Fetching {owner}/{repo}#{number}...", file=sys.stderr)
        issue = fetch_issue_text(owner, repo, number)
        tmpdir = tempfile.mkdtemp(prefix="scout-")
        root = Path(tmpdir) / repo
        print(f"Cloning {owner}/{repo} (shallow)...", file=sys.stderr)
        clone_repo(owner, repo, str(root))
        return root, issue, tmpdir
    if not args.repo:
        sys.exit("Provide a repo path with --issue/--issue-file, or use --issue-url.")
    root = Path(args.repo).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    if args.issue_file:
        return root, Path(args.issue_file).read_text(encoding="utf-8"), None
    if args.issue:
        return root, args.issue, None
    sys.exit("Provide --issue or --issue-file (or --issue-url).")


def run(client, root: Path, issue: str, approach_n: int | None) -> Plan | None:
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
        p = plan(client, context, issue, t.single_approach_summary)
        show_plan(p)
        return p

    print("Generating distinct approaches...", file=sys.stderr)
    aset = approaches(client, context, issue)
    show_approaches(aset)

    n = approach_n
    if n is None and sys.stdin.isatty():
        try:
            n = int(input(f"\nPick an approach [1-{len(aset.approaches)}]: ").strip())
        except (ValueError, EOFError):
            n = None
    if n is None:
        print(f"\nRe-run with --approach N (1-{len(aset.approaches)}) to get the plan for one.")
        return None
    if not (1 <= n <= len(aset.approaches)):
        sys.exit(f"--approach must be 1..{len(aset.approaches)}")

    chosen = aset.approaches[n - 1]
    print(f"\nChosen: [{n}] {chosen.name}", file=sys.stderr)
    chosen_desc = f"{chosen.name}: {chosen.summary}\nTradeoff: {chosen.tradeoffs}\nFit: {chosen.codebase_fit}"
    p = plan(client, context, issue, chosen_desc)
    show_plan(p)
    return p


# ---- execute: hand the plan to Claude Code to write the change (stop before PR)

def render_plan_md(issue_url: str, p: Plan) -> str:
    lines = [
        f"# Implement the fix for {issue_url}",
        "",
        "You are in a clone of the target repo on a fresh branch. Implement the "
        "plan below: edit the files, and run the project's tests if you can.",
        "",
        "Hard rules:",
        "- Do NOT push, do NOT open a pull request, do NOT run `gh` or `git push`.",
        "- Do NOT add any AI/Claude/agent attribution to commits or anywhere else.",
        "- Stay within the scope of the plan; don't refactor unrelated code.",
        "- If something in the plan is wrong given the actual code, do the right "
        "thing and note it in your final summary.",
        "",
        "## Approach",
        "",
        p.approach_recap,
        "",
        "## Files to change",
        "",
    ]
    for fc in p.files:
        lines += [f"### {fc.path}", f"- What: {fc.what_it_does}", f"- Testing: {fc.how_tested}", ""]
    lines += ["## Risks to respect", "", p.risks, ""]
    return "\n".join(lines)


def execute_plan(root: Path, issue_url: str, p: Plan) -> None:
    executor = os.environ.get("SCOUT_EXECUTOR", "claude")
    if not shutil.which(executor):
        sys.exit(f"executor '{executor}' not found on PATH "
                 f"(install Claude Code, or set SCOUT_EXECUTOR).")
    _, _, number = parse_issue_url(issue_url)
    branch = f"scout/issue-{number}"
    git = ["git", "-C", str(root)]
    base = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    subprocess.run([*git, "checkout", "-q", "-B", branch], check=True)

    plan_md = render_plan_md(issue_url, p)
    (root.parent / "plan.md").write_text(plan_md)  # reference copy, outside the work tree

    print(hr(f"EXECUTING via '{executor}' (non-interactive) on branch {branch}"))
    print("Handing the plan to the executor; it edits + tests autonomously...\n", file=sys.stderr)
    sys.stdout.flush()  # keep our headers ahead of the subprocess's own output
    subprocess.run([executor, "--print", plan_md, "--allowedTools", "Edit,Write,Bash"], cwd=str(root))

    subprocess.run([*git, "add", "-A"])
    print(hr("DIFF (all changes vs. clone base)"))
    sys.stdout.flush()
    subprocess.run([*git, "--no-pager", "diff", "--cached", base])
    commits = subprocess.check_output([*git, "log", "--oneline", f"{base}..HEAD"], text=True).strip()

    print(hr("REVIEW — nothing has been pushed"))
    if commits:
        print("commits made on the branch:\n" + commits + "\n")
    print("Review the diff above. If it's good, push it yourself (open the PR by "
          "hand so your pre-flight rules apply — closedByPullRequestsReferences, "
          "parallel PRs, repo-specific base branch, no AI attribution):")
    print(f"  cd {root}")
    print(f"  git push <your-fork-remote> {branch}")


def main():
    ap = argparse.ArgumentParser(description="Approach-first issue resolver.")
    ap.add_argument("repo", nargs="?", help="Path to a local repo to work on.")
    ap.add_argument("--issue", help="Issue text (local mode).")
    ap.add_argument("--issue-file", help="File containing the issue text (local mode).")
    ap.add_argument("--issue-url", help="GitHub issue URL; fetches the issue and shallow-clones the repo.")
    ap.add_argument("--approach", type=int, help="Pick approach N after a triage fork.")
    ap.add_argument("--execute", action="store_true",
                    help="After the plan, hand it to Claude Code to write the change (stops before PR). Requires --issue-url.")
    ap.add_argument("--no-keep", action="store_true",
                    help="With --execute, delete the clone on exit instead of keeping it for you to push.")
    args = ap.parse_args()

    if args.execute and not args.issue_url:
        sys.exit("--execute requires --issue-url (it works on a throwaway clone, not your local repo).")

    root, issue, tmpdir = resolve_repo_and_issue(args)
    keep = False
    try:
        p = run(make_client(), root, issue, args.approach)
        if args.execute:
            if p is None:
                print("\nNo plan to execute — the triage forked. Re-run with "
                      "--approach N --execute.", file=sys.stderr)
            else:
                execute_plan(root, args.issue_url, p)
                keep = not args.no_keep
    finally:
        if tmpdir and not keep:
            shutil.rmtree(tmpdir, ignore_errors=True)
        elif tmpdir and keep:
            print(f"\nClone kept at {root} (re-run with --no-keep to auto-delete).", file=sys.stderr)


# ---- scan: triage a repo's open-issue queue

REPO_URL_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git|/)?$")
EFFORT_RANK = {"small": 0, "medium": 1, "large": 2}


def parse_repo_url(url: str) -> tuple[str, str]:
    m = REPO_URL_RE.match(url.strip())
    if not m:
        sys.exit(f"Not a github repo URL (need .../owner/repo): {url}")
    return m.group(1), m.group(2)


def list_open_issues(owner: str, repo: str, limit: int) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["gh", "issue", "list", "--repo", f"{owner}/{repo}", "--state", "open",
             "--limit", str(limit), "--json", "number,title,body,url,labels,assignees"],
            text=True, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"gh issue list failed: {e.stderr.strip() or e}")
    return json.loads(out)


def issue_blob(it: dict) -> str:
    labels = ", ".join(l.get("name", "") for l in it.get("labels", []))
    assignees = ", ".join(a.get("login", "") for a in it.get("assignees", [])) or "(none)"
    body = (it.get("body") or "(no body)")[:4000]
    return (f"TITLE: {it.get('title','')}\nLABELS: {labels}\n"
            f"ASSIGNEES: {assignees}\n\n{body}")


def scan_issues(client, root: Path, issues: list[dict], model: str, workers: int) -> list[dict]:
    def one(it: dict) -> dict:
        blob = issue_blob(it)
        try:
            ctx = gather_context(root, blob)
            v = scan_triage(client, ctx, blob, model=model)
            # gh assignees are a hard claim signal; OR them with the model's read.
            claimed = bool(it.get("assignees")) or v.claimed
            return {"issue": it, "verdict": v, "claimed": claimed, "error": None}
        except Exception as e:
            return {"issue": it, "verdict": None, "claimed": bool(it.get("assignees")),
                    "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, issues))

    def key(r):
        v = r["verdict"]
        if v is None:
            return (2, 0.0, 9)
        good = v.fixable and not r["claimed"]
        return (0 if good else 1, -v.confidence, EFFORT_RANK.get(v.effort, 1))

    results.sort(key=key)
    return results


def print_scan_table(owner: str, repo: str, results: list[dict]):
    print(hr(f"SCAN — {owner}/{repo} — {len(results)} open issues, ranked by fixable + unclaimed"))
    print(f"\n  {'#':>6}  {'verdict':<9} {'conf':>4}  {'effort':<6} {'claim':<5} {'blocker':<16} one-liner")
    print("  " + "─" * 110)
    for r in results:
        it, v = r["issue"], r["verdict"]
        num = f"#{it['number']}"
        if v is None:
            print(f"  {num:>6}  {'ERROR':<9} {'-':>4}  {'-':<6} {'-':<5} {'-':<16} {r['error']}")
            continue
        mark = "✓ GO" if (v.fixable and not r["claimed"]) else "· skip"
        claim = "yes" if r["claimed"] else "no"
        blk = "" if v.blocker_type == "none" else v.blocker_type
        print(f"  {num:>6}  {mark:<9} {v.confidence:>4.2f}  {v.effort:<6} {claim:<5} {blk:<16} {v.one_line[:80]}")
    print("\n  Top picks (✓ GO):")
    gos = [r for r in results if r["verdict"] and r["verdict"].fixable and not r["claimed"]]
    for r in gos[:5]:
        print(f"    {r['issue']['url']}")
    if not gos:
        print("    (none — nothing cleanly fixable + unclaimed in this batch)")


def scan_main(argv):
    ap = argparse.ArgumentParser(prog="scout scan",
                                 description="Triage a repo's open issues into a fixable-issue queue.")
    ap.add_argument("repo_url", help="GitHub repo URL, e.g. https://github.com/owner/repo")
    ap.add_argument("--limit", type=int, default=10, help="How many open issues to scan (default 10).")
    ap.add_argument("--model", default=MODEL, help=f"Model for the scan verdict (default {MODEL}).")
    ap.add_argument("--workers", type=int, default=5, help="Concurrent triage calls (default 5).")
    args = ap.parse_args(argv)

    owner, repo = parse_repo_url(args.repo_url)
    print(f"Listing up to {args.limit} open issues in {owner}/{repo}...", file=sys.stderr)
    issues = list_open_issues(owner, repo, args.limit)
    if not issues:
        print("No open issues found.")
        return
    tmpdir = tempfile.mkdtemp(prefix="scout-")
    root = Path(tmpdir) / repo
    print(f"Cloning {owner}/{repo} (shallow)...", file=sys.stderr)
    clone_repo(owner, repo, str(root))
    try:
        print(f"Triaging {len(issues)} issues ({args.workers} at a time)...", file=sys.stderr)
        results = scan_issues(make_client(), root, issues, args.model, args.workers)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print_scan_table(owner, repo, results)


# ---- sweep: run scout scan across a curated list of repos

DEFAULT_REPOS = [
    # repos with open PRs / warm maintainers
    "https://github.com/drizzle-team/drizzle-orm",
    "https://github.com/mastra-ai/mastra",
    "https://github.com/livekit/agents-js",
    "https://github.com/livekit/agents",
    "https://github.com/supabase/auth",
    "https://github.com/TanStack/db",
    "https://github.com/TanStack/query",
    "https://github.com/PostHog/posthog",
    "https://github.com/PostHog/posthog-js",
    "https://github.com/storybookjs/storybook",
    "https://github.com/expo/expo",
    "https://github.com/better-auth/better-auth",
    "https://github.com/liveblocks/liveblocks",
    "https://github.com/vercel/turborepo",
    "https://github.com/vercel/ai",
    "https://github.com/vercel/next.js",
    "https://github.com/inngest/inngest",
    "https://github.com/BerriAI/litellm",
    # fresh orgs
    "https://github.com/solidjs/solid",
    "https://github.com/solidjs/solid-start",
    "https://github.com/pinecone-io/pinecone-ts-client",
    "https://github.com/pinecone-io/pinecone-python-client",
    "https://github.com/getsentry/sentry-cocoa",
    "https://github.com/getsentry/sentry-java",
    "https://github.com/convex-dev/convex-js",
    "https://github.com/outline/outline",
    "https://github.com/unjs/nf3",
    "https://github.com/unjs/consola",
    "https://github.com/prisma/prisma",
    "https://github.com/cloudflare/workers-sdk",
    "https://github.com/langchain-ai/langchain",
    "https://github.com/RevenueCat/purchases-ios",
    "https://github.com/RevenueCat/purchases-android",
]


def sweep_one(args_tuple) -> list[dict]:
    """Scan one repo; return list of GO results. Runs in a thread."""
    repo_url, limit, model, workers, client = args_tuple
    owner, repo = parse_repo_url(repo_url)
    print(f"  scanning {owner}/{repo}...", file=sys.stderr)
    try:
        issues = list_open_issues(owner, repo, limit)
    except SystemExit:
        print(f"  ! {owner}/{repo} — gh list failed, skipping", file=sys.stderr)
        return []
    if not issues:
        return []
    tmpdir = tempfile.mkdtemp(prefix="scout-")
    root = Path(tmpdir) / repo
    try:
        clone_repo(owner, repo, str(root))
    except SystemExit:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"  ! {owner}/{repo} — clone failed, skipping", file=sys.stderr)
        return []
    try:
        results = scan_issues(client, root, issues, model, workers)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    gos = [r for r in results if r["verdict"] and r["verdict"].fixable and not r["claimed"]]
    for r in gos:
        r["repo_url"] = repo_url
        r["owner"] = owner
        r["repo"] = repo
    return gos


def sweep_main(argv):
    ap = argparse.ArgumentParser(prog="scout sweep",
                                 description="Run scout scan across multiple repos and rank all GOs together.")
    ap.add_argument("repos", nargs="*",
                    help="GitHub repo URLs to sweep. Omit to use the built-in curated list.")
    ap.add_argument("--limit", type=int, default=15,
                    help="Open issues to scan per repo (default 15).")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Model for scan verdicts (default haiku — cheap for bulk).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent triage calls per repo (default 4).")
    ap.add_argument("--repo-workers", type=int, default=3,
                    help="Repos to clone+scan concurrently (default 3).")
    ap.add_argument("--save", default="/tmp/sweep_candidates.json",
                    help="Where to save GO results (default /tmp/sweep_candidates.json).")
    args = ap.parse_args(argv)

    repos = args.repos or DEFAULT_REPOS
    client = make_client()
    print(f"Sweeping {len(repos)} repos ({args.limit} issues each, model={args.model})...",
          file=sys.stderr)

    sweep_args = [(url, args.limit, args.model, args.workers, client) for url in repos]
    all_gos: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.repo_workers) as ex:
        for gos in ex.map(sweep_one, sweep_args):
            all_gos.extend(gos)

    # rank: confidence desc, effort asc
    all_gos.sort(key=lambda r: (-r["verdict"].confidence, EFFORT_RANK.get(r["verdict"].effort, 1)))

    print(hr(f"SWEEP — {len(repos)} repos — {len(all_gos)} GO candidates"))
    print(f"\n  {'REPO':<35} {'#':>6}  {'conf':>4}  {'effort':<6} one-liner")
    print("  " + "─" * 110)
    for r in all_gos[:30]:
        it, v = r["issue"], r["verdict"]
        repo_short = f"{r['owner']}/{r['repo']}"
        print(f"  {repo_short:<35} #{it['number']:<6}  {v.confidence:>4.2f}  {v.effort:<6} {v.one_line[:70]}")
    print(f"\n  URLs for top picks:")
    for r in all_gos[:10]:
        print(f"    {r['issue']['url']}")

    with open(args.save, "w") as f:
        json.dump([{
            "repo": r["repo_url"], "owner": r["owner"],
            "number": r["issue"]["number"], "url": r["issue"]["url"],
            "title": r["issue"].get("title",""),
            "confidence": r["verdict"].confidence,
            "effort": r["verdict"].effort,
            "one_line": r["verdict"].one_line,
        } for r in all_gos], f, indent=2)
    print(f"\n  Saved {len(all_gos)} GOs to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep_main(sys.argv[2:])
    else:
        main()
