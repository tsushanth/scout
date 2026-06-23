#!/usr/bin/env python3
"""Stream 2 issue scanner — scores open issues across repos for OSS PR candidates."""
import json, subprocess, sys
from datetime import datetime, timezone

REPOS = [
    # repos with existing PRs / warm maintainers
    "drizzle-team/drizzle-orm",
    "mastra-ai/mastra",
    "livekit/agents-js",
    "livekit/agents",
    "supabase/auth",
    "TanStack/db",
    "TanStack/query",
    "PostHog/posthog",
    "PostHog/posthog-js",
    "storybookjs/storybook",
    "expo/expo",
    "better-auth/better-auth",
    "liveblocks/liveblocks",
    "vercel/turborepo",
    "vercel/ai",
    "vercel/next.js",
    "inngest/inngest",
    "BerriAI/litellm",
    # fresh orgs
    "solidjs/solid",
    "solidjs/solid-start",
    "pinecone-io/pinecone-ts-client",
    "pinecone-io/pinecone-python-client",
    "getsentry/sentry-cocoa",
    "getsentry/sentry-java",
    "convex-dev/convex-js",
    "outline/outline",
    "unjs/nf3",
    "unjs/consola",
    "prisma/prisma",
    "cloudflare/workers-sdk",
    "cloudflare/agents-starter",
    "langchain-ai/langchain",
    "RevenueCat/purchases-ios",
    "RevenueCat/purchases-android",
]

SKIP_AUTHORS = {"tsushanth"}
SKIP_LABELS  = {"wontfix", "invalid", "duplicate", "stale", "question", "feature", "enhancement", "discussion"}
SKIP_TITLE_KW = {"feature request", "feat:", "enhancement:", "question:", "[feat]", "[feature]"}

def age_score(created: str) -> int:
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - dt).days
        if days <= 7:   return 3
        if days <= 21:  return 2
        if days <= 60:  return 1
    except Exception:
        pass
    return 0

def forensic_score(body: str) -> int:
    if not body: return 0
    signals = 0
    if "```" in body:                         signals += 1
    if any(c in body for c in ["#L", ".py:", ".ts:", ".go:", ".rs:", ".swift:"]):
        signals += 1
    if "http" in body and "github.com" in body: signals += 1
    import re
    if re.search(r'line \d+|:\d+\b', body):   signals += 1
    if 4 <= signals:  return 4
    if signals == 3:  return 3
    if signals == 2:  return 2
    if signals == 1:  return 1
    return 0

def score_issue(issue: dict) -> int:
    title  = (issue.get("title") or "").lower()
    body   = issue.get("body") or ""
    labels = [l.get("name","").lower() for l in (issue.get("labels") or [])]
    comments = len(issue.get("comments") or [])

    # hard skips
    if any(lbl in SKIP_LABELS for lbl in labels): return -1
    if any(kw in title for kw in SKIP_TITLE_KW):  return -1
    if issue.get("assignees"):                      return -1

    s  = age_score(issue.get("createdAt",""))
    if s == 0: return -1   # too old
    s += 2 if "bug" in labels else 0
    s += forensic_score(body)
    s += 1 if len(body) >= 600 else 0
    s += 2 if comments == 0 else (1 if comments <= 2 else 0)
    return s

def fetch_issues(repo: str, limit: int = 25) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["gh","issue","list","--repo",repo,"--state","open",
             "--limit",str(limit),"--json",
             "number,title,body,createdAt,labels,assignees,comments,url,author"],
            text=True, stderr=subprocess.DEVNULL, timeout=20,
        )
        return json.loads(out)
    except Exception:
        return []

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    candidates = []
    for repo in REPOS:
        print(f"  scanning {repo}...", file=sys.stderr)
        issues = fetch_issues(repo, limit)
        for iss in issues:
            if (iss.get("author") or {}).get("login","") in SKIP_AUTHORS:
                continue
            s = score_issue(iss)
            if s >= 5:
                candidates.append((s, repo, iss))

    candidates.sort(key=lambda x: -x[0])
    print(f"\n{'SCORE':>5}  {'REPO':<40} {'#':>6}  TITLE")
    print("─"*120)
    for s, repo, iss in candidates[:30]:
        title = (iss.get("title") or "")[:70]
        print(f"{s:>5}  {repo:<40} #{iss['number']:<6}  {title}")
    print(f"\n{len(candidates)} candidates found across {len(REPOS)} repos")

    # save for follow-up
    with open("/tmp/stream2_candidates.json","w") as f:
        json.dump([{"score":s,"repo":r,"issue":i} for s,r,i in candidates[:30]], f, indent=2)
    print("Saved top 30 to /tmp/stream2_candidates.json", file=sys.stderr)

if __name__ == "__main__":
    main()
