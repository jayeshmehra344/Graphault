import ast
import re
import os
import sys
import time
import base64
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
from db import get_db

load_dotenv()

BASE_URL = "https://api.github.com"
CLONE_DIR = Path(__file__).parent.parent.parent / "tmp" / "github_repos"
CHECKPOINT_COL = "github_checkpoints"

STAR_RANGES = [
    ("1..10",      "student"),
    ("11..50",     "small"),
    ("51..200",    "medium"),
    ("201..1000",  "large"),
    ("1001..5000", "professional"),
]


# ---------- GitHub API helpers ----------

def _headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN not set in .env")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_get(url, params=None, accept=None, retries=3):
    for attempt in range(retries):
        try:
            hdrs = _headers()
            if accept:
                hdrs["Accept"] = accept
            resp = requests.get(url, headers=hdrs, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429):
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time() + 2, 5)
            print(f"  rate limited, sleeping {wait:.0f}s")
            time.sleep(wait)
            continue
        if resp.status_code in (404, 410, 451):
            return None
        time.sleep(2 ** attempt)
    return None


# ---------- Checkpoint helpers ----------

def _load_checkpoints(db):
    """Return (visited_set, repos_with_bugs_count) from previous runs."""
    docs = list(db[CHECKPOINT_COL].find({}, {"repo_full_name": 1, "had_bugs": 1, "_id": 0}))
    visited = {d["repo_full_name"] for d in docs}
    with_bugs = sum(1 for d in docs if d.get("had_bugs", False))
    return visited, with_bugs


def _save_checkpoint(db, repo_full_name, inserted, had_bugs):
    db[CHECKPOINT_COL].update_one(
        {"repo_full_name": repo_full_name},
        {"$set": {"repo_full_name": repo_full_name, "inserted": inserted, "had_bugs": had_bugs}},
        upsert=True,
    )


# ---------- Candidate collection ----------

def _collect_candidates(target_per_range=600):
    """Search repos across all star range tiers, up to target_per_range each."""
    all_candidates = []
    seen = set()
    for star_range, tier in STAR_RANGES:
        print(f"  stars:{star_range} ({tier}) ...", end=" ", flush=True)
        collected = 0
        page = 1
        while collected < target_per_range and page <= 10:
            resp = _gh_get(
                f"{BASE_URL}/search/repositories",
                params={
                    "q": f"language:python stars:{star_range}",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
            )
            time.sleep(2)
            if resp is None:
                break
            items = resp.json().get("items", [])
            if not items:
                break
            for item in items:
                if item["full_name"] not in seen:
                    seen.add(item["full_name"])
                    all_candidates.append(item)
                    collected += 1
                if collected >= target_per_range:
                    break
            page += 1
        print(f"{collected} found")
    return all_candidates


# ---------- Code analysis ----------

def _get_functions_with_ranges(source_code):
    """Return list of (name, start_line, end_line, snippet) via ast."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    lines = source_code.splitlines()
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno
            snippet = "\n".join(lines[start - 1:end])
            result.append((node.name, start, end, snippet))
    return result


def _parse_changed_lines(patch):
    """Return (old_changed, new_changed) line number sets from a unified diff patch."""
    old_changed, new_changed = set(), set()
    old_line = new_line = 0
    for line in (patch or "").splitlines():
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            old_line = int(m.group(1))
            new_line = int(m.group(2))
            continue
        if line.startswith("-") and not line.startswith("---"):
            old_changed.add(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            new_changed.add(new_line)
            new_line += 1
        elif not line.startswith("\\"):
            old_line += 1
            new_line += 1
    return old_changed, new_changed


# ---------- GitHub data fetchers ----------

def _get_file_content(repo_full_name, path, ref):
    resp = _gh_get(
        f"{BASE_URL}/repos/{repo_full_name}/contents/{path}",
        params={"ref": ref},
    )
    if resp is None:
        return None
    data = resp.json()
    if isinstance(data, list) or data.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        return None


def _get_closing_commit(repo_full_name, issue_number):
    """Return the commit SHA that closed this issue, or None.

    Tries three strategies in order:
    1. Direct close event with commit_id (push directly closed the issue).
    2. Referenced event with commit_id (commit message contained 'fixes #N').
    3. Cross-referenced merged PR via the timeline API — the common case where
       a PR description says 'closes #N' and the merge commit is what we want.
    """
    resp = _gh_get(f"{BASE_URL}/repos/{repo_full_name}/issues/{issue_number}/events")
    if resp is not None:
        referenced_sha = None
        for event in resp.json():
            evt = event.get("event", "")
            if evt == "closed" and event.get("commit_id"):
                return event["commit_id"]
            if evt == "referenced" and event.get("commit_id") and not referenced_sha:
                referenced_sha = event["commit_id"]
        if referenced_sha:
            return referenced_sha

    resp = _gh_get(
        f"{BASE_URL}/repos/{repo_full_name}/issues/{issue_number}/timeline",
        accept="application/vnd.github.mockingbird-preview+json",
    )
    if resp is None:
        return None
    for event in resp.json():
        if event.get("event") != "cross-referenced":
            continue
        source = event.get("source", {})
        if source.get("type") != "issue":
            continue
        src_issue = source.get("issue", {})
        pr_meta = src_issue.get("pull_request", {})
        if not pr_meta or not pr_meta.get("merged_at"):
            continue
        pr_number = src_issue["number"]
        pr_resp = _gh_get(f"{BASE_URL}/repos/{repo_full_name}/pulls/{pr_number}")
        if pr_resp is None:
            continue
        merge_sha = pr_resp.json().get("merge_commit_sha")
        if merge_sha:
            return merge_sha

    return None


def _clone_repo(repo_full_name, default_branch):
    dest = CLONE_DIR / repo_full_name.replace("/", "__")
    if not dest.exists():
        url = f"https://github.com/{repo_full_name}.git"
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", default_branch, url, str(dest)],
            check=False, capture_output=True,
        )
    return dest


# ---------- Core processing ----------

def _process_commit(repo_full_name, commit_sha, repo_name, collection):
    """
    For each Python file changed in commit_sha:
      - old file (parent): functions overlapping changed lines → label=1 (buggy)
                           functions not overlapping             → label=0 (clean)
      - new file (commit): functions overlapping changed lines → label=0 (fixed)
    Returns number of label=1 insertions.
    """
    resp = _gh_get(f"{BASE_URL}/repos/{repo_full_name}/commits/{commit_sha}")
    if resp is None:
        return 0

    data = resp.json()
    if not data.get("parents"):
        return 0
    parent_sha = data["parents"][0]["sha"]

    buggy_inserted = 0
    seen = set()

    for file_info in data.get("files", []):
        path = file_info.get("filename", "")
        if not path.endswith(".py"):
            continue
        status = file_info.get("status", "")
        patch = file_info.get("patch", "")
        if not patch:
            continue

        old_changed, new_changed = _parse_changed_lines(patch)

        if status != "added":
            old_content = _get_file_content(repo_full_name, path, parent_sha)
            time.sleep(0.25)
            if old_content:
                for name, start, end, snippet in _get_functions_with_ranges(old_content):
                    func_lines = set(range(start, end + 1))
                    label = 1 if (func_lines & old_changed) else 0
                    key = (name, label, path, "old")
                    if key not in seen:
                        seen.add(key)
                        collection.insert_one({
                            "func_name": name,
                            "source": "github",
                            "label": label,
                            "code": snippet,
                            "repo": repo_name,
                        })
                        if label == 1:
                            buggy_inserted += 1

        if status != "removed":
            new_content = _get_file_content(repo_full_name, path, commit_sha)
            time.sleep(0.25)
            if new_content:
                for name, start, end, snippet in _get_functions_with_ranges(new_content):
                    func_lines = set(range(start, end + 1))
                    if func_lines & new_changed:
                        key = (name, 0, path, "new")
                        if key not in seen:
                            seen.add(key)
                            collection.insert_one({
                                "func_name": name,
                                "source": "github",
                                "label": 0,
                                "code": snippet,
                                "repo": repo_name,
                            })

    return buggy_inserted


# ---------- Entry point ----------

def load_github(max_repos=1000, bugs_per_repo=15):
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    db = get_db()
    collection = db["labeled_functions"]

    visited, already_with_bugs = _load_checkpoints(db)
    remaining = max(0, max_repos - already_with_bugs)
    print(f"Checkpoint: {len(visited)} repos visited, {already_with_bugs} had bugs. Need {remaining} more with bugs.")

    if remaining == 0:
        print("Target already reached.")
        return

    print(f"\nCollecting repo candidates across {len(STAR_RANGES)} star tiers...")
    target_per_range = min(800, max(200, remaining * 5 // len(STAR_RANGES)))
    candidates = _collect_candidates(target_per_range)
    print(f"Total candidates: {len(candidates)}\n")

    total_inserted = 0
    newly_with_bugs = 0

    for repo_data in candidates:
        if newly_with_bugs >= remaining:
            break

        repo_full_name = repo_data["full_name"]
        if repo_full_name in visited:
            continue
        visited.add(repo_full_name)

        overall = already_with_bugs + newly_with_bugs + 1
        print(f"[{overall}/{max_repos}] {repo_full_name} ({repo_data['stargazers_count']}★)", end=" ", flush=True)

        issues_resp = _gh_get(
            f"{BASE_URL}/repos/{repo_full_name}/issues",
            params={"labels": "bug", "state": "closed", "per_page": bugs_per_repo},
        )
        time.sleep(0.5)

        if issues_resp is None:
            print("→ API error, skipping")
            _save_checkpoint(db, repo_full_name, 0, had_bugs=False)
            continue

        issues = issues_resp.json()
        if not isinstance(issues, list) or not issues:
            print("→ no closed bug issues")
            _save_checkpoint(db, repo_full_name, 0, had_bugs=False)
            continue

        print(f"→ {len(issues)} bug issues found")
        _clone_repo(repo_full_name, repo_data.get("default_branch", "main"))

        repo_inserted = 0
        for issue in issues[:bugs_per_repo]:
            commit_sha = _get_closing_commit(repo_full_name, issue["number"])
            time.sleep(0.5)
            if not commit_sha:
                continue
            n = _process_commit(repo_full_name, commit_sha, repo_data["name"], collection)
            repo_inserted += n
            total_inserted += n
            time.sleep(0.5)

        print(f"  → inserted {repo_inserted} buggy functions")
        _save_checkpoint(db, repo_full_name, repo_inserted, had_bugs=True)
        newly_with_bugs += 1

        if newly_with_bugs % 50 == 0:
            overall = already_with_bugs + newly_with_bugs
            print(f"\n{'='*60}")
            print(f"  PROGRESS: {overall}/{max_repos} repos with bugs | {total_inserted} functions this run")
            print(f"{'='*60}\n")

    overall_total = already_with_bugs + newly_with_bugs
    print(f"\nDone. {newly_with_bugs} new repos with bugs processed ({overall_total} total).")
    print(f"Functions inserted this run: {total_inserted}")


if __name__ == "__main__":
    load_github(max_repos=1000, bugs_per_repo=15)
