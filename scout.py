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
import html
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
DIGEST_FILE_CHARS = 16000   # deeper per-file cap for the whole-repo `find` audit
DIGEST_TOTAL_CHARS = 200000  # cap the whole digest so one request can't blow the rate limit


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


class GraphNode(BaseModel):
    id: str = Field(description="Short stable slug used to reference this node in edges.")
    label: str = Field(description="File path (repo-relative) or component name.")
    kind: Literal["new", "modified", "existing"] = Field(
        description="new = added by this change; modified = changed; existing = "
        "untouched code the change interacts with.")
    summary: str = Field(description="One line: what changes here, or how it's involved.")


class GraphEdge(BaseModel):
    src: str = Field(description="Source node id.")
    dst: str = Field(description="Destination node id.")
    label: str = Field(description="Relationship, e.g. 'calls', 'imports', 'feeds', 'gated by'.")


class ChangeGraph(BaseModel):
    nodes: list[GraphNode] = Field(description="5-12 components: the files changed plus existing pieces they touch.")
    edges: list[GraphEdge] = Field(description="Relationships between nodes; src/dst must be defined node ids.")


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


def read_capped(path: Path, cap: int = MAX_FILE_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:cap] + ("\n... (truncated)" if len(text) > cap else "")


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


def make_client(force_subscription: bool = False) -> "anthropic.Anthropic":
    """Credential resolution.

    Default: use ANTHROPIC_API_KEY if set, else the Claude Code subscription
    (OAuth token from the keychain). With --subscription / SCOUT_SUBSCRIPTION=1,
    always use the subscription and never touch an API key — so scout's planning
    calls run on the same Claude Code subscription as the `--execute` step.
    """
    force = force_subscription or os.environ.get("SCOUT_SUBSCRIPTION") in ("1", "true", "yes")
    if os.environ.get("ANTHROPIC_API_KEY") and not force:
        return anthropic.Anthropic(max_retries=4)
    tok = claude_code_oauth_token()
    if tok:
        print("Using your Claude Code subscription (no API key).", file=sys.stderr)
        return anthropic.Anthropic(
            auth_token=tok,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
            max_retries=4,
        )
    if force:
        sys.exit("--subscription set but no Claude Code token found — sign in to Claude Code.")
    sys.exit("No credential: set ANTHROPIC_API_KEY or sign in to Claude Code.")


SYSTEM = (
    "You are a senior engineer triaging an issue against a specific codebase. "
    "You reason about the existing design before proposing change. You are "
    "blunt about tradeoffs and never pad. The codebase context you are given "
    "is partial (a tree plus the most relevant files), so reason from patterns "
    "you can see and say when something needs confirmation."
)


THINKING_MODELS = {"claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6", "claude-sonnet-4-5"}


def parse(client, schema, instruction: str, context: str, issue: str, model: str = MODEL):
    kwargs = dict(
        model=model,
        max_tokens=16000,
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
    )
    if any(model.startswith(m) for m in THINKING_MODELS):
        kwargs["thinking"] = {"type": "adaptive"}
    return client.messages.parse(**kwargs).parsed_output


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


def change_graph(client, issue: str, rr: "IssueRun") -> ChangeGraph:
    desc = render_plan_md("(this issue)", rr.plan)
    if rr.approaches and rr.chosen_index:
        chosen = rr.approaches.approaches[rr.chosen_index - 1]
        desc += f"\n\nApproach fit with existing code: {chosen.codebase_fit}"
    return parse(
        client, ChangeGraph,
        "Build a small CHANGE GRAPH for a reviewer of the proposed fix below: the "
        "components involved — the files being changed AND the existing pieces they "
        "interact with — plus the relationships between them. 5-12 nodes. Mark each "
        "node new | modified | existing. Node label is the file path or component "
        "name; summary is one line. Every edge's src/dst must be node ids you define.",
        desc, issue,
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


class IssueRun:
    """The structured result of one issue resolution, for the HTML artifact."""
    def __init__(self, triage, approaches=None, chosen_index=None, plan=None):
        self.triage = triage
        self.approaches = approaches      # ApproachSet | None
        self.chosen_index = chosen_index  # 1-based | None
        self.plan = plan                  # Plan | None


def run(client, root: Path, issue: str, approach_n: int | None) -> IssueRun:
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
        return IssueRun(triage=t, plan=p)

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
        return IssueRun(triage=t, approaches=aset)
    if not (1 <= n <= len(aset.approaches)):
        sys.exit(f"--approach must be 1..{len(aset.approaches)}")

    chosen = aset.approaches[n - 1]
    print(f"\nChosen: [{n}] {chosen.name}", file=sys.stderr)
    chosen_desc = f"{chosen.name}: {chosen.summary}\nTradeoff: {chosen.tradeoffs}\nFit: {chosen.codebase_fit}"
    p = plan(client, context, issue, chosen_desc)
    show_plan(p)
    return IssueRun(triage=t, approaches=aset, chosen_index=n, plan=p)


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


def execute_plan(root: Path, issue_url: str, p: Plan) -> str:
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
    diff = subprocess.check_output([*git, "--no-pager", "diff", "--cached", base], text=True)
    print(hr("DIFF (all changes vs. clone base)"))
    print(diff)
    commits = subprocess.check_output([*git, "log", "--oneline", f"{base}..HEAD"], text=True).strip()

    print(hr("REVIEW — nothing has been pushed"))
    if commits:
        print("commits made on the branch:\n" + commits + "\n")
    print("Review the diff above. If it's good, push it yourself (open the PR by "
          "hand so your pre-flight rules apply — closedByPullRequestsReferences, "
          "parallel PRs, repo-specific base branch, no AI attribution):")
    print(f"  cd {root}")
    print(f"  git push <your-fork-remote> {branch}")
    return diff


def issue_title_from(issue: str, fallback: str = "Issue") -> str:
    for line in issue.splitlines():
        if line.startswith("TITLE:"):
            return line[6:].strip() or fallback
    for line in issue.splitlines():
        if line.strip():
            return line.strip()[:90]
    return fallback


def _render_diff(diff: str) -> str:
    rows = []
    for ln in diff.split("\n"):
        e = html.escape(ln)
        if ln.startswith(("+++", "---", "diff ", "index ")):
            cls = "dm"
        elif ln.startswith("+"):
            cls = "da"
        elif ln.startswith("-"):
            cls = "dr"
        elif ln.startswith("@@"):
            cls = "dh"
        else:
            cls = ""
        rows.append(f'<span class="{cls}">{e}</span>' if cls else e)
    return "\n".join(rows)


def split_diff_by_file(diff: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not diff:
        return out
    cur, buf = None, []
    for ln in diff.split("\n"):
        if ln.startswith("diff --git "):
            if cur:
                out[cur] = "\n".join(buf)
            m = re.search(r" b/(\S+)$", ln)
            cur, buf = (m.group(1) if m else None), [ln]
        else:
            buf.append(ln)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def render_change_map(graph: "ChangeGraph", diff: str | None) -> str:
    """A zoomable, pan-able SVG of the change; click a node to see its diff hunk."""
    if not graph or not graph.nodes:
        return ""
    diffmap = split_diff_by_file(diff)

    def hunk_for(label: str) -> str:
        if label in diffmap:
            return diffmap[label]
        base = label.rsplit("/", 1)[-1]
        for p, h in diffmap.items():
            if p.endswith("/" + label) or p.rsplit("/", 1)[-1] == base:
                return h
        return ""

    COLX = {"existing": 160, "modified": 480, "new": 800}
    NW, NH, VGAP = 220, 60, 42
    buckets: dict[str, list] = {"existing": [], "modified": [], "new": []}
    for n in graph.nodes:
        buckets.get(n.kind, buckets["modified"]).append(n)
    pos = {}
    for kind, arr in buckets.items():
        for i, n in enumerate(arr):
            pos[n.id] = (COLX.get(kind, 480), 70 + i * (NH + VGAP), n)
    rows = max((len(a) for a in buckets.values()), default=1)
    W, H = 960, max(70 + rows * (NH + VGAP) + 20, 200)

    fill = {"existing": ("#eef1f4", "#cbd5e1", "#475569"),
            "modified": ("#fef3c7", "#eab308", "#92400e"),
            "new": ("#dcfce7", "#34d399", "#166534")}

    edge_svg = []
    for ed in graph.edges:
        if ed.src not in pos or ed.dst not in pos:
            continue
        sx, sy, _ = pos[ed.src]
        dx, dy, _ = pos[ed.dst]
        x1, y1 = sx + NW / 2, sy + NH / 2
        x2, y2 = dx - NW / 2, dy + NH / 2
        mx = (x1 + x2) / 2
        edge_svg.append(f'<path class=edge d="M{x1:.0f} {y1:.0f} C{mx:.0f} {y1:.0f} {mx:.0f} {y2:.0f} {x2:.0f} {y2:.0f}"/>')
        if ed.label:
            edge_svg.append(f'<text class=el x="{mx:.0f}" y="{(y1+y2)/2-4:.0f}">{html.escape(ed.label)}</text>')

    node_svg, NODE = [], {}
    for nid, (x, y, n) in pos.items():
        f, b, tcol = fill.get(n.kind, fill["modified"])
        lbl = n.label if len(n.label) <= 28 else "…" + n.label[-27:]
        node_svg.append(
            f'<g class=node data-id="{html.escape(nid)}" transform="translate({x-NW/2:.0f},{y:.0f})">'
            f'<rect width="{NW}" height="{NH}" rx="10" fill="{f}" stroke="{b}" stroke-width="1.5"/>'
            f'<text x="13" y="25" class=nl fill="{tcol}">{html.escape(lbl)}</text>'
            f'<text x="13" y="44" class=nk fill="{tcol}">{html.escape(n.kind)}</text></g>')
        hk = hunk_for(n.label)
        NODE[nid] = {"label": n.label, "kind": n.kind, "summary": n.summary,
                     "hunk": _render_diff(hk) if hk else ""}
    nodejson = json.dumps(NODE).replace("</", "<\\/")

    return f"""
<h2>Change map <span class=dim>(scroll to zoom · drag to pan · click a node)</span></h2>
<div class="cmapwrap">
<style>
.cmapwrap{{display:grid;grid-template-columns:1fr 320px;gap:14px;align-items:start;margin-bottom:8px}}
.cmapwrap svg{{width:100%;height:460px;background:#fff;border:1px solid var(--line);border-radius:10px;cursor:grab}}
.cmapwrap svg:active{{cursor:grabbing}}
.node{{cursor:pointer}} .node.sel rect{{stroke:#2563eb;stroke-width:3}}
.nl{{font:600 12.5px ui-monospace,Menlo,monospace}} .nk{{font:10px -apple-system,sans-serif;opacity:.65;text-transform:uppercase;letter-spacing:.05em}}
.edge{{fill:none;stroke:#cbd5e1;stroke-width:1.5}} .el{{font:10px -apple-system,sans-serif;fill:#94a3b8;text-anchor:middle}}
.cdetail{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;position:sticky;top:12px;max-height:520px;overflow:auto}}
.cdetail .dl{{font:600 13px ui-monospace,Menlo,monospace;word-break:break-all}}
.cdetail .dk{{font-size:10.5px;text-transform:uppercase;color:var(--dim);margin:2px 0 8px}}
.cdetail .ds{{font-size:14px;margin-bottom:10px}}
@media(max-width:760px){{.cmapwrap{{grid-template-columns:1fr}}}}
</style>
<svg id=cmap viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet"><g id=cvp>{''.join(edge_svg)}{''.join(node_svg)}</g></svg>
<div class=cdetail id=cdetail><span class=dim>Click a node to see what it changes — and its diff hunk if one exists.</span></div>
</div>
<script>
(function(){{
const NODE={nodejson},W={W},H={H};
const svg=document.getElementById('cmap'),vp=document.getElementById('cvp'),det=document.getElementById('cdetail');
let s=1,tx=0,ty=0,drag=null;
const upd=()=>vp.setAttribute('transform',`translate(${{tx}},${{ty}}) scale(${{s}})`);
const k=()=>{{const r=svg.getBoundingClientRect();return 1/Math.min(r.width/W,r.height/H);}};
svg.addEventListener('wheel',e=>{{e.preventDefault();const r=svg.getBoundingClientRect(),kk=k();
  const mx=(e.clientX-r.left)*kk,my=(e.clientY-r.top)*kk,ns=Math.min(4,Math.max(.3,s*(e.deltaY<0?1.12:.9)));
  tx=mx-(mx-tx)*(ns/s);ty=my-(my-ty)*(ns/s);s=ns;upd();}},{{passive:false}});
svg.addEventListener('mousedown',e=>drag={{x:e.clientX,y:e.clientY,tx,ty}});
window.addEventListener('mousemove',e=>{{if(!drag)return;const kk=k();tx=drag.tx+(e.clientX-drag.x)*kk;ty=drag.ty+(e.clientY-drag.y)*kk;upd();}});
window.addEventListener('mouseup',()=>drag=null);
const esc=t=>(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
document.querySelectorAll('.node').forEach(g=>g.addEventListener('click',ev=>{{ev.stopPropagation();
  document.querySelectorAll('.node').forEach(x=>x.classList.remove('sel'));g.classList.add('sel');
  const n=NODE[g.dataset.id];if(!n)return;
  det.innerHTML='<div class=dl>'+esc(n.label)+'</div><div class=dk>'+esc(n.kind)+'</div><div class=ds>'+esc(n.summary)+'</div>'+(n.hunk?'<pre class="diff">'+n.hunk+'</pre>':'<div class=dim>No diff hunk — run with --execute to fill this.</div>');
}}));
}})();
</script>"""


def render_issue_html(repo_name: str, title: str, issue_url: str | None,
                      base_url: str | None, rr: "IssueRun", diff: str | None,
                      graph: "ChangeGraph | None" = None) -> str:
    e = html.escape

    def flink(path: str) -> str:
        p = e(path)
        if base_url:
            return f'<a href="{base_url}/blob/HEAD/{p}" target="_blank" rel="noopener"><code>{p}</code></a>'
        return f"<code>{p}</code>"

    t = rr.triage
    head_title = f'<a href="{e(issue_url)}" target="_blank" rel="noopener">{e(title)}</a>' if issue_url else e(title)
    forked = t.needs_approach_choice
    triage_label = "Approach fork — a real decision to make" if forked else "One obvious approach"
    triage_col = "#b9770e" if forked else "#15803d"

    appr_html = ""
    if rr.approaches:
        cards = []
        for i, a in enumerate(rr.approaches.approaches, 1):
            chosen = (rr.chosen_index == i)
            files = " ".join(flink(x) for x in a.files_to_touch) or "<span class=dim>—</span>"
            badge = '<span class="cho">✓ chosen</span>' if chosen else ""
            cards.append(f"""
      <div class="appr{' won' if chosen else ''}">
        <div class="ah"><span class="an">{i}. {e(a.name)}</span>{badge}</div>
        <div class="asum">{e(a.summary)}</div>
        <div class="arow"><b>tradeoff</b> {e(a.tradeoffs)}</div>
        <div class="arow"><b>fit</b> {e(a.codebase_fit)}</div>
        <div class="arow"><b>touches</b> {files}</div>
      </div>""")
        appr_html = f'<h2>Approaches</h2><div class="grid">{"".join(cards)}</div>'

    plan_html = ""
    if rr.plan:
        p = rr.plan
        fcards = []
        for fc in p.files:
            fcards.append(f"""
      <div class="fc">
        <div class="fp">{flink(fc.path)}</div>
        <div class="frow"><b>does</b> {e(fc.what_it_does)}</div>
        <div class="frow"><b>tested</b> {e(fc.how_tested)}</div>
      </div>""")
        plan_html = (f'<h2>Plan</h2><div class="summary">{e(p.approach_recap)}</div>'
                     f'{"".join(fcards)}'
                     f'<div class="risks"><b>risks</b> {e(p.risks)}</div>')

    map_html = render_change_map(graph, diff) if graph else ""

    diff_html = ""
    if diff and diff.strip():
        diff_html = f'<h2>Diff <span class=dim>(written by the executor — nothing pushed)</span></h2><pre class="diff">{_render_diff(diff)}</pre>'

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>scout · {e(title)}</title>
<style>
:root{{--bg:#fafafa;--card:#fff;--line:#e5e7eb;--ink:#1f2937;--dim:#6b7280}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:21px;margin:0 0 2px}} h1 a{{color:var(--ink)}}
.repo{{color:var(--dim);font-size:13px;margin:0 0 18px}}
.tri{{border-left:4px solid {triage_col};background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:0 0 8px}}
.tri b{{color:{triage_col}}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:30px 0 12px}}
.summary{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 14px}}
code{{background:#f1f3f5;border-radius:4px;padding:1px 5px;font-size:12.5px}}
a code{{background:#eef2ff;color:#3730a3}}
.grid{{display:flex;gap:14px;flex-wrap:wrap}}
.appr{{flex:1 1 280px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;opacity:.82}}
.appr.won{{opacity:1;border-color:#15803d;box-shadow:0 0 0 1px #15803d}}
.ah{{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px}}
.an{{font-weight:700}} .cho{{font-size:11px;font-weight:700;color:#fff;background:#15803d;border-radius:5px;padding:2px 7px}}
.asum{{font-size:14px;margin-bottom:8px}}
.arow,.frow{{font-size:13px;color:#374151;margin:5px 0}} .arow b,.frow b{{color:var(--dim);text-transform:uppercase;font-size:11px;letter-spacing:.04em;margin-right:6px}}
.fc{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:0 0 10px}}
.fp{{font-weight:600;margin-bottom:6px}}
.risks{{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px 14px;margin:8px 0;font-size:13.5px}}
.risks b{{color:#b9770e;text-transform:uppercase;font-size:11px;letter-spacing:.04em;margin-right:6px}}
pre.diff{{background:#0d1117;color:#c9d1d9;border-radius:10px;padding:16px;overflow:auto;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
pre.diff .da{{color:#3fb950}} pre.diff .dr{{color:#f85149}} pre.diff .dh{{color:#58a6ff}} pre.diff .dm{{color:#8b949e}}
.dim{{color:var(--dim)}} footer{{color:var(--dim);font-size:12px;margin-top:32px}}
</style></head><body><div class=wrap>
<h1>{head_title}</h1>
<p class=repo>{e(repo_name)} &middot; scout resolution</p>
<div class=tri><b>{e(triage_label)}</b> — {e(t.reasoning)}</div>
{map_html}
{appr_html}
{plan_html}
{diff_html}
<footer>Generated by <code>scout</code>. Plan is a proposal; the diff (if any) was written on a throwaway clone and not pushed.</footer>
</div></body></html>"""


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
    ap.add_argument("--html", action="store_true",
                    help="Emit a self-contained interactive artifact: triage → approaches → plan → diff.")
    ap.add_argument("--subscription", action="store_true",
                    help="Use your Claude Code subscription for the planning calls too (never an API key).")
    args = ap.parse_args()

    if args.execute and not args.issue_url:
        sys.exit("--execute requires --issue-url (it works on a throwaway clone, not your local repo).")

    root, issue, tmpdir = resolve_repo_and_issue(args)
    keep = False
    diff = None
    try:
        client = make_client(args.subscription)
        rr = run(client, root, issue, args.approach)
        if args.execute:
            if rr.plan is None:
                print("\nNo plan to execute — the triage forked. Re-run with "
                      "--approach N --execute.", file=sys.stderr)
            else:
                diff = execute_plan(root, args.issue_url, rr.plan)
                keep = not args.no_keep
        if args.html:
            graph = None
            if rr.plan:
                print("Building the change map...", file=sys.stderr)
                try:
                    graph = change_graph(client, issue, rr)
                except Exception as exc:
                    print(f"  (change map skipped: {exc})", file=sys.stderr)
            if args.issue_url:
                owner, repo, number = parse_issue_url(args.issue_url)
                base_url = f"https://github.com/{owner}/{repo}"
                hpath = Path.cwd() / f"RESOLVE-{repo}-{number}.html"
            else:
                base_url, hpath = None, Path.cwd() / "scout-resolve.html"
            hpath.write_text(
                render_issue_html(root.name, issue_title_from(issue), args.issue_url,
                                  base_url, rr, diff, graph),
                encoding="utf-8")
            print(f"\nWrote {hpath}  (open in a browser)")
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
    ap.add_argument("--subscription", action="store_true",
                    help="Use your Claude Code subscription instead of an API key.")
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
        results = scan_issues(make_client(args.subscription), root, issues, args.model, args.workers)
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
    ap.add_argument("--subscription", action="store_true",
                    help="Use your Claude Code subscription instead of an API key.")
    args = ap.parse_args(argv)

    repos = args.repos or DEFAULT_REPOS
    client = make_client(args.subscription)
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


# ---- find: audit a repo for latent issues, write an OVERVIEW.md

class Area(BaseModel):
    name: str = Field(description="Name of the functional area / module.")
    does: str = Field(description="What this area does, in one sentence.")
    key_files: list[str] = Field(description="A few repo-relative key files in this area.")


class RepoMap(BaseModel):
    summary: str = Field(description="One paragraph: what this repo is and does.")
    areas: list[Area] = Field(description="The main functional areas.")
    test_shape: str = Field(description="How the project is tested (frameworks, where tests live, gaps).")


class Finding(BaseModel):
    title: str = Field(description="Short, specific title of the issue.")
    category: Literal["bug", "health"] = Field(description="Correctness bug, or engineering-health item.")
    kind: Literal[
        "logic_error", "unhandled_error", "edge_case", "race", "resource_leak",
        "wrong_api_usage", "dead_code", "missing_test", "inconsistency",
        "risky_todo", "perf", "other",
    ] = Field(description="The specific kind of issue.")
    file: str = Field(description="Repo-relative path of the file the issue is in.")
    locator: str = Field(description="Function/symbol/region pointing at the code (not whole-file).")
    evidence: str = Field(description="Why it's a problem, referencing the specific code.")
    proposed_change: str = Field(description="One-line direction for the fix.")
    severity: Literal["high", "medium", "low"] = Field(description="Impact if left unfixed.")


class FindingSet(BaseModel):
    findings: list[Finding] = Field(description="Concrete, code-grounded findings. Fewer is fine; none if clean.")


class Verdict(BaseModel):
    real: bool = Field(description="True only if the finding is a genuine issue confirmable from the code.")
    confidence: float = Field(description="0.0-1.0 confidence that it's real.")
    severity: Literal["high", "medium", "low"] = Field(description="Re-assessed severity.")
    rationale: str = Field(description="Why it's real, or why it's refuted (handled elsewhere / false positive).")
    proposed_change: str = Field(description="Tightened one-line fix (if real).")


SOURCE_HINT_DIRS = ("src/", "lib/", "app/", "core/", "pkg/", "internal/", "packages/", "cmd/")
TEST_HINTS = (".test.", ".spec.", "_test.", "/tests/", "/__tests__/", "/test/", "/spec/")
GEN_HINTS = ("generated", "/dist/", "/build/", ".min.", "/vendor/", ".pb.")
SEV_RANK = {"high": 0, "medium": 1, "low": 2}


def digest_files(root: Path, max_files: int) -> list[tuple[str, Path]]:
    """Pick the most central source files for a whole-repo audit digest."""
    cand: list[tuple[int, int, str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() not in CODE_EXTS:
                continue
            try:
                sz = p.stat().st_size
            except Exception:
                continue
            if sz > 200_000 or sz < 60:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            low = rel.lower()
            score = 0
            if any(h in low for h in SOURCE_HINT_DIRS):
                score += 3
            if any(h in low for h in TEST_HINTS):
                score -= 2
            if any(h in low for h in GEN_HINTS):
                score -= 4
            score -= rel.count("/")          # prefer shallower
            if 800 <= sz <= 20000:
                score += 1                   # prefer meaty-but-readable files
            cand.append((score, -sz, rel, p))
    cand.sort(key=lambda t: (-t[0], t[1]))
    return [(rel, p) for _, _, rel, p in cand[:max_files]]


def repo_digest(root: Path, max_files: int) -> tuple[str, list[str]]:
    parts = [f"# Repo: {root.name}\n", "## File tree\n", build_tree(root), "\n"]
    for cand in ("README.md", "readme.md", "README.rst", "README"):
        rp = root / cand
        if rp.exists():
            parts += ["## README\n", read_capped(rp), "\n"]
            break
    files = digest_files(root, max_files)
    total = sum(len(s) for s in parts)
    included: list[str] = []
    body: list[str] = []
    for rel, p in files:
        chunk = read_capped(p, DIGEST_FILE_CHARS)
        if included and total + len(chunk) > DIGEST_TOTAL_CHARS:
            break  # keep the whole digest under the rate-limit-safe budget
        body += [f"\n### {rel}\n```\n", chunk, "\n```\n"]
        total += len(chunk) + len(rel) + 12
        included.append(rel)
    parts.append(f"\n## Source files ({len(included)} sampled for audit)\n")
    parts += body
    return "".join(parts), included


def audit_call(client, schema, instruction: str, context: str, model: str):
    return client.messages.parse(
        model=model, max_tokens=16000, thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{instruction}\n\n=== CODEBASE ===\n{context}"}],
        output_format=schema,
    ).parsed_output


def comprehend(client, digest: str, model: str) -> RepoMap:
    return audit_call(
        client, RepoMap,
        "Map this repository for a newcomer: a one-paragraph summary of what it is "
        "and does, the main functional areas (name, what each does, key files), and "
        "how it's tested (including obvious gaps).",
        digest, model,
    )


LENSES = {
    "bug": "Find concrete CORRECTNESS bugs: logic errors, unhandled errors/exceptions, "
           "edge cases, off-by-one, race conditions, resource leaks, incorrect API usage.",
    "health": "Find ENGINEERING-HEALTH issues: dead/unreachable code, missing tests on "
              "important paths, inconsistent patterns, risky TODO/FIXME, clear performance "
              "footguns, fragile error handling.",
}


def discover(client, digest: str, lens: str, model: str, cap: int) -> list[Finding]:
    instr = (
        f"{LENSES[lens]} Report up to {cap} of the most important, each grounded in "
        "specific code you can point to (file + symbol/region) with a one-line fix. "
        "Do NOT invent issues or give generic advice; if the code is clean, return "
        "fewer or none. Only report what you can defend from the code shown."
    )
    return audit_call(client, FindingSet, instr, digest, model).findings


def verify(client, root: Path, f: Finding, model: str) -> Verdict:
    fp = root / f.file
    code = read_capped(fp, DIGEST_FILE_CHARS) if fp.exists() else "(file not present at that path in the repo)"
    instr = (
        "Adversarially VERIFY this finding against the actual file below. Try to "
        "REFUTE it: is it a real issue, or is it already handled, a false positive, "
        "or not reproducible? Default real=false if you cannot confirm it from the "
        "code. If real, give severity and a tightened one-line fix.\n\n"
        f"=== FINDING ===\ntitle: {f.title}\nfile: {f.file}\nkind: {f.kind}\n"
        f"locator: {f.locator}\nevidence: {f.evidence}\nproposed: {f.proposed_change}"
    )
    return audit_call(client, Verdict, instr, f"### {f.file}\n```\n{code}\n```", model)


def dedup_findings(findings: list[Finding]) -> list[Finding]:
    # Two lenses often surface the same issue with slightly different titles, so
    # also dedup on the proposed fix (identical fix in the same file == same issue).
    seen_title, seen_fix, out = set(), set(), []
    for f in findings:
        kt = (f.file.lower(), f.title.lower()[:50])
        kf = (f.file.lower(), " ".join(f.proposed_change.lower().split())[:60])
        if kt in seen_title or kf in seen_fix:
            continue
        seen_title.add(kt)
        seen_fix.add(kf)
        out.append(f)
    return out


def write_overview(path: Path, root: Path, rmap: RepoMap, sampled: list[str],
                   confirmed: list[tuple[Finding, Verdict]], n_candidates: int):
    L = [
        f"# Overview — {root.name}",
        "",
        "_Generated by `scout find`: an LLM audit of the codebase. Findings are "
        "adversarially verified but are still **candidates** — confirm before acting. "
        f"This audit sampled {len(sampled)} source files, so it is not exhaustive._",
        "",
        "## What this is",
        "",
        rmap.summary,
        "",
        "## Map",
        "",
    ]
    for a in rmap.areas:
        links = ", ".join(f"[`{kf}`]({kf})" for kf in a.key_files)
        L += [f"- **{a.name}** — {a.does}", f"  - {links}" if links else "", ""]
    L += ["## Test shape", "", rmap.test_shape, "",
          f"## Findings — {len(confirmed)} verified (of {n_candidates} candidates)", ""]
    if not confirmed:
        L.append("_No findings survived verification in this pass._")
    for f, v in confirmed:
        L += [
            f"### [{v.severity.upper()}] {f.title}  · `{f.category}/{f.kind}`",
            f"- **where:** [`{f.file}`]({f.file}) — {f.locator}",
            f"- **why:** {v.rationale}",
            f"- **fix:** {v.proposed_change}",
            f"- **confidence:** {v.confidence:.2f}",
            "",
        ]
    L += ["---", "",
          "Resolve any finding with scout:", "",
          "```sh",
          "scout.py <repo> --issue \"<paste the finding's title + why>\"",
          "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


SEV_HEX = {"high": "#c0392b", "medium": "#b9770e", "low": "#6b7280"}


def render_overview_html(repo_name: str, base_url: str | None, rmap: RepoMap,
                         sampled: list[str], confirmed: list[tuple[Finding, Verdict]],
                         n_candidates: int) -> str:
    """A self-contained, interactive HTML 'artifact' of the audit (no server/deps)."""
    e = html.escape

    def flink(path: str) -> str:
        p = e(path)
        if base_url:
            return f'<a href="{base_url}/blob/HEAD/{p}" target="_blank" rel="noopener"><code>{p}</code></a>'
        return f"<code>{p}</code>"

    counts = {"high": 0, "medium": 0, "low": 0}
    for _, v in confirmed:
        counts[v.severity] = counts.get(v.severity, 0) + 1

    cards = []
    for f, v in confirmed:
        col = SEV_HEX.get(v.severity, "#6b7280")
        pct = int(round(v.confidence * 100))
        cards.append(f"""
    <div class="card" data-sev="{e(v.severity)}" data-cat="{e(f.category)}">
      <div class="ch">
        <span class="pill" style="background:{col}">{e(v.severity.upper())}</span>
        <span class="ct">{e(f.title)}</span>
        <span class="tag">{e(f.category)}/{e(f.kind)}</span>
      </div>
      <div class="meta">{flink(f.file)} &middot; {e(f.locator)}</div>
      <div class="why">{e(v.rationale)}</div>
      <div class="fix"><b>fix</b> {e(v.proposed_change)}</div>
      <div class="bar"><i style="width:{pct}%;background:{col}"></i></div>
      <div class="cn">confidence {pct}%</div>
    </div>""")

    areas = []
    for a in rmap.areas:
        kf = " ".join(flink(x) for x in a.key_files) or "<span class=dim>—</span>"
        areas.append(f"<details><summary><b>{e(a.name)}</b> — {e(a.does)}</summary>"
                     f"<div class=kf>{kf}</div></details>")

    chip = lambda key, label: f'<button class="chip on" data-f="{key}">{label}</button>'
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>scout · {e(repo_name)}</title>
<style>
:root{{--bg:#fafafa;--card:#fff;--line:#e5e7eb;--ink:#1f2937;--dim:#6b7280}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:860px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:22px;margin:0 0 4px}}
.sub{{color:var(--dim);margin:0 0 18px;font-size:13px}}
.summary{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 24px}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:28px 0 12px}}
details{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:0 0 8px}}
summary{{cursor:pointer}} .kf{{margin-top:8px;font-size:13px;color:var(--dim)}}
code{{background:#f1f3f5;border-radius:4px;padding:1px 5px;font-size:12.5px}}
a code{{background:#eef2ff;color:#3730a3}}
.bars{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}}
.chip{{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:5px 12px;font-size:13px;cursor:pointer;color:var(--dim)}}
.chip.on{{background:var(--ink);color:#fff;border-color:var(--ink)}}
.card{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:14px 16px;margin:0 0 12px}}
.card[data-sev=high]{{border-left-color:#c0392b}} .card[data-sev=medium]{{border-left-color:#b9770e}} .card[data-sev=low]{{border-left-color:#6b7280}}
.ch{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.pill{{color:#fff;font-size:11px;font-weight:700;border-radius:5px;padding:2px 7px;letter-spacing:.03em}}
.ct{{font-weight:600;flex:1;min-width:200px}}
.tag{{font-size:12px;color:var(--dim);background:#f1f3f5;border-radius:5px;padding:2px 7px}}
.meta{{font-size:13px;color:var(--dim);margin:8px 0}}
.why{{margin:8px 0;font-size:14px}}
.fix{{font-size:13.5px;background:#f6f8fa;border-radius:6px;padding:8px 10px;margin:8px 0}}
.fix b{{color:#15803d;margin-right:6px}}
.bar{{height:5px;background:#eef0f2;border-radius:3px;overflow:hidden}} .bar i{{display:block;height:100%}}
.cn{{font-size:11px;color:var(--dim);margin-top:4px}}
.dim{{color:var(--dim)}} footer{{color:var(--dim);font-size:12px;margin-top:32px}}
.hidden{{display:none}}
</style></head><body><div class=wrap>
<h1>{e(repo_name)}</h1>
<p class=sub>scout audit &middot; {len(confirmed)} verified findings (of {n_candidates} candidates) &middot;
{counts['high']} high &middot; {counts['medium']} medium &middot; {counts['low']} low &middot;
{len(sampled)} files sampled</p>
<div class=summary>{e(rmap.summary)}</div>
<h2>Map</h2>
{''.join(areas)}
<h2>Test shape</h2>
<div class=summary>{e(rmap.test_shape)}</div>
<h2>Findings</h2>
<div class=bars>
{chip('all','All')}{chip('high','High')}{chip('medium','Medium')}{chip('low','Low')}
{chip('bug','Bugs')}{chip('health','Health')}
</div>
{''.join(cards) or '<p class=dim>No findings survived verification.</p>'}
<footer>Generated by <code>scout find</code>. Findings are LLM-discovered, adversarially verified — still candidates; confirm before acting. Sample, not exhaustive.</footer>
</div>
<script>
const cards=[...document.querySelectorAll('.card')];
let sev='all', cat='all';
function apply(){{cards.forEach(c=>{{const s=(sev==='all'||c.dataset.sev===sev),k=(cat==='all'||c.dataset.cat===cat);c.classList.toggle('hidden',!(s&&k));}});}}
document.querySelectorAll('.chip').forEach(b=>b.onclick=()=>{{
  const f=b.dataset.f;
  if(f==='all'){{sev='all';cat='all';}}
  else if(f==='bug'||f==='health'){{cat=(cat===f?'all':f);}}
  else {{sev=(sev===f?'all':f);}}
  document.querySelectorAll('.chip').forEach(x=>{{const xf=x.dataset.f;
    x.classList.toggle('on', xf==='all'?(sev==='all'&&cat==='all'):(xf===sev||xf===cat));}});
  apply();
}});
</script></body></html>"""


def print_find_summary(confirmed: list[tuple[Finding, Verdict]]):
    print(hr(f"VERIFIED FINDINGS — {len(confirmed)}"))
    for f, v in confirmed:
        print(f"\n  [{v.severity.upper()}] {f.title}  ({f.category}/{f.kind}, conf {v.confidence:.2f})")
        print(f"      {f.file} — {f.locator}")
        print(f"      fix: {v.proposed_change}")


def find_main(argv):
    ap = argparse.ArgumentParser(prog="scout find",
                                 description="Audit a repo: discover + verify latent issues, write OVERVIEW.md.")
    ap.add_argument("repo", help="Local repo path OR a github repo URL (shallow-cloned).")
    ap.add_argument("--model", default=MODEL, help=f"Model (default {MODEL}).")
    ap.add_argument("--max-files", type=int, default=24, help="Source files to sample into the digest.")
    ap.add_argument("--per-lens", type=int, default=8, help="Max candidate findings per lens.")
    ap.add_argument("--workers", type=int, default=5, help="Concurrent verification calls.")
    ap.add_argument("--out", default="OVERVIEW.md", help="Artifact filename written into a local repo.")
    ap.add_argument("--html", action="store_true",
                    help="Also emit a self-contained interactive OVERVIEW.html artifact.")
    ap.add_argument("--subscription", action="store_true",
                    help="Use your Claude Code subscription instead of an API key.")
    args = ap.parse_args(argv)

    tmpdir = None
    if args.repo.startswith("http"):
        owner, repo = parse_repo_url(args.repo)
        tmpdir = tempfile.mkdtemp(prefix="scout-")
        root = Path(tmpdir) / repo
        print(f"Cloning {owner}/{repo} (shallow)...", file=sys.stderr)
        clone_repo(owner, repo, str(root))
        out_path = Path.cwd() / f"OVERVIEW-{repo}.md"  # survives clone cleanup
        base_url = f"https://github.com/{owner}/{repo}"
    else:
        root = Path(args.repo).expanduser().resolve()
        if not root.is_dir():
            sys.exit(f"Not a directory: {root}")
        out_path = root / args.out
        base_url = None

    client = make_client(args.subscription)
    try:
        print("Building repo digest...", file=sys.stderr)
        digest, sampled = repo_digest(root, args.max_files)
        print(f"  sampled {len(sampled)} source files", file=sys.stderr)

        print("Comprehending the codebase...", file=sys.stderr)
        rmap = comprehend(client, digest, args.model)

        print("Discovering issues (bug + health lenses)...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=2) as ex:
            lens_res = list(ex.map(
                lambda lens: discover(client, digest, lens, args.model, args.per_lens),
                ["bug", "health"]))
        cands = dedup_findings([f for r in lens_res for f in r])
        print(f"  {len(cands)} candidate findings; verifying (adversarial)...", file=sys.stderr)

        def _ver(f):
            try:
                return (f, verify(client, root, f, args.model))
            except Exception:
                return (f, None)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            verdicts = list(ex.map(_ver, cands))
        confirmed = [(f, v) for f, v in verdicts if v and v.real]
        confirmed.sort(key=lambda fv: (SEV_RANK.get(fv[1].severity, 3), -fv[1].confidence))
        print(f"  {len(confirmed)} verified (dropped {len(cands) - len(confirmed)} unverified)", file=sys.stderr)

        write_overview(out_path, root, rmap, sampled, confirmed, len(cands))
        print(f"\nWrote {out_path}")
        if args.html:
            html_path = out_path.with_suffix(".html")
            html_path.write_text(
                render_overview_html(root.name, base_url, rmap, sampled, confirmed, len(cands)),
                encoding="utf-8")
            print(f"Wrote {html_path}  (open in a browser)")
        print_find_summary(confirmed)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _dispatch():
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub == "scan":
        scan_main(sys.argv[2:])
    elif sub == "sweep":
        sweep_main(sys.argv[2:])
    elif sub == "find":
        find_main(sys.argv[2:])
    else:
        main()


if __name__ == "__main__":
    try:
        _dispatch()
    except anthropic.RateLimitError:
        sys.exit("\nRate limited (429). On --subscription this is the shared Claude "
                 "Code pool — close other Claude Code sessions and retry, or drop "
                 "--subscription to use ANTHROPIC_API_KEY.")
    except KeyboardInterrupt:
        sys.exit(130)
