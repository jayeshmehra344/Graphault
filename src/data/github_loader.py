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
            end = node.end_lineno  # available in Python 3.8+
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
    # Strategies 1 & 2: regular events endpoint
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

    # Strategy 3: timeline — find a cross-referenced merged PR
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

        # Old file → label buggy (1) or clean (0) for every function
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

        # New file → only the changed functions, as fixed (label=0) examples
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

def load_github(max_repos=15, bugs_per_repo=5):
    CLONE_DIR.mkdir(parents=True, exist_ok=True)

    db = get_db()
    collection = db["labeled_functions"]

    print(f"Searching Python repos (10-500 stars)...")
    repo_candidates = []
    page = 1
    while len(repo_candidates) < max_repos * 3 and page <= 5:
        resp = _gh_get(
            f"{BASE_URL}/search/repositories",
            params={
                "q": "language:python stars:10..500",
                "sort": "updated",
                "order": "desc",
                "per_page": 30,
                "page": page,
            },
        )
        if resp is None:
            break
        items = resp.json().get("items", [])
        if not items:
            break
        repo_candidates.extend(items)
        page += 1
        time.sleep(2)

    total_inserted = 0
    repos_processed = 0
    repos_skipped = 0
    seen_repos = set()

    for repo_data in repo_candidates:
        if repos_processed >= max_repos:
            break

        repo_full_name = repo_data["full_name"]
        if repo_full_name in seen_repos:
            continue
        seen_repos.add(repo_full_name)
        default_branch = repo_data.get("default_branch", "main")
        print(f"\n[{repos_processed + 1}/{max_repos}] {repo_full_name} ({repo_data['stargazers_count']} stars)")

        # Get closed bug issues
        issues_resp = _gh_get(
            f"{BASE_URL}/repos/{repo_full_name}/issues",
            params={"labels": "bug", "state": "closed", "per_page": bugs_per_repo},
        )
        time.sleep(0.5)

        if issues_resp is None:
            repos_skipped += 1
            continue

        issues = issues_resp.json()
        if not isinstance(issues, list) or not issues:
            print("  no closed bug issues, skipping")
            repos_skipped += 1
            continue

        _clone_repo(repo_full_name, default_branch)

        repo_inserted = 0
        for issue in issues[:bugs_per_repo]:
            issue_number = issue["number"]
            commit_sha = _get_closing_commit(repo_full_name, issue_number)
            time.sleep(0.5)

            if not commit_sha:
                continue

            n = _process_commit(repo_full_name, commit_sha, repo_data["name"], collection)
            repo_inserted += n
            total_inserted += n
            time.sleep(0.5)

        print(f"  inserted {repo_inserted} buggy functions")
        repos_processed += 1

    print(f"\nDone. {repos_processed} repos processed, {repos_skipped} skipped.")
    print(f"Total inserted: {total_inserted} labeled functions from GitHub")


if __name__ == "__main__":
    load_github(max_repos=50, bugs_per_repo=5)
