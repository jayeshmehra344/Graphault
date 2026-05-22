import os
import sys
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'parser'))
from parse import parse_file
from db import get_db

# Strict keywords - only genuine bug fixes
BUG_KEYWORDS = [
    "fix bug",
    "bug fix", 
    "hotfix",
    "critical fix",
    "fixes #",
    "fixed #",
    "patch bug",
    "null pointer",
    "segfault",
    "memory leak",
    "regression",
    "security fix",
    "vulnerability"
]

def is_bug_fix_commit(message):
    message = message.lower()
    return any(keyword in message for keyword in BUG_KEYWORDS)

def get_function_ranges(filepath):
    """
    Returns a dict of {func_name: (start_line, end_line)}
    for every function in a file.
    """
    try:
        import ast
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
        ranges = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ranges[node.name] = (node.lineno, node.end_lineno)
        return ranges
    except Exception:
        return {}

def get_changed_lines(repo_path, commit_hash, filepath):
    """
    Returns list of line numbers changed in a specific file
    for a specific commit using git diff.
    """
    diff = subprocess.run(
        ["git", "diff", f"{commit_hash}^!", "--", filepath],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    ).stdout

    changed_lines = []
    current_line = 0

    for line in diff.split("\n"):
        # parse unified diff hunk headers like @@ -10,7 +10,8 @@
        if line.startswith("@@"):
            try:
                parts = line.split("+")[1].split(",")[0]
                current_line = int(parts.strip())
            except Exception:
                continue
        elif line.startswith("+") and not line.startswith("+++"):
            changed_lines.append(current_line)
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1

    return changed_lines

def get_buggy_functions(repo_path):
    """
    For each bug fix commit:
    1. Check commit message strictly
    2. Get exact line numbers changed
    3. Map those lines to specific functions
    Only those exact functions get labeled risky.
    """
    buggy_functions = {}  # {func_name: count} - how many times flagged

    # get full git log
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    ).stdout.strip().split("\n")

    print(f"scanning {len(log)} commits...")

    for line in log:
        if not line:
            continue

        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue

        commit_hash = parts[0]
        message = parts[1]

        if not is_bug_fix_commit(message):
            continue

        # get files changed in this commit
        files = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r",
             "--name-only", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        ).stdout.strip().split("\n")

        for filepath in files:
            if not filepath.endswith(".py"):
                continue

            full_path = os.path.join(repo_path, filepath)
            if not os.path.exists(full_path):
                continue

            # get exact lines changed in this file
            changed_lines = get_changed_lines(
                repo_path, commit_hash, filepath
            )
            if not changed_lines:
                continue

            # get function ranges for this file
            func_ranges = get_function_ranges(full_path)

            # check which function each changed line belongs to
            for func_name, (start, end) in func_ranges.items():
                for line_no in changed_lines:
                    if start <= line_no <= end:
                        # this function was directly modified
                        buggy_functions[func_name] = \
                            buggy_functions.get(func_name, 0) + 1
                        break  # count once per commit per function

    return buggy_functions


def label_repo(repo_name, repo_path):
    db = get_db()
    doc = db["repos"].find_one({"repo": repo_name})

    if not doc:
        print(f"repo {repo_name} not found in MongoDB")
        return

    buggy_functions = get_buggy_functions(repo_path)
    print(f"found {len(buggy_functions)} directly modified functions")

    features = doc["features"]
    labels = {}

    for func_name in features.keys():
        # label risky only if modified in 2+ bug fix commits
        # single occurrence could be coincidence
        count = buggy_functions.get(func_name, 0)
        labels[func_name] = 1 if count >= 2 else 0

    positive = sum(labels.values())
    total = len(labels)
    print(f"labeled: {positive} risky ({positive/total*100:.1f}%), "
          f"{total - positive} safe ({(total-positive)/total*100:.1f}%)")

    db["repos"].update_one(
        {"repo": repo_name},
        {"$set": {"labels": labels}}
    )
    print(f"labels saved for {repo_name}")


if __name__ == "__main__":
    label_repo("flask", "data/cloned/flask")