import ast
import re
import os
import sys
import subprocess
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
from db import get_db

BUGSINPY_REPO = "https://github.com/soarsmu/BugsInPy"
CLONE_DIR = Path(__file__).parent.parent.parent / "tmp" / "bugsinpy"


def _clone_bugsinpy():
    if not CLONE_DIR.exists():
        print("Cloning BugsInPy metadata repo...")
        subprocess.run(
            ["git", "clone", "--depth=1", BUGSINPY_REPO, str(CLONE_DIR)],
            check=True
        )


def extract_functions_from_code(source_code):
    def _walk_ast(tree):
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    try:
        return _walk_ast(ast.parse(source_code))
    except SyntaxError:
        pass

    try:
        indented = "\n".join("    " + line for line in source_code.splitlines())
        return _walk_ast(ast.parse(f"def _wrapper():\n{indented}"))
    except SyntaxError:
        pass

    return re.findall(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", source_code, re.MULTILINE)


def _parse_patch(patch_text):
    """
    Yields (func_name, buggy_code, fixed_code) per Python hunk.
    buggy_code = context + removed lines; fixed_code = context + added lines.
    func_name comes from the @@ header hint (enclosing function in the diff).
    """
    results = []
    in_python = False
    func_name = "unknown"
    buggy, fixed = [], []
    in_hunk = False

    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            if in_hunk and in_python and (buggy or fixed):
                results.append((func_name, "\n".join(buggy), "\n".join(fixed)))
            in_python = ".py" in line
            in_hunk = False
            buggy, fixed = [], []
            continue

        if not in_python:
            continue

        if line.startswith("--- ") or line.startswith("+++ "):
            continue

        m = re.match(r"^@@[^@]*@@\s*(.*)", line)
        if m:
            if in_hunk and (buggy or fixed):
                results.append((func_name, "\n".join(buggy), "\n".join(fixed)))
            hint = m.group(1).strip()
            fn = re.search(r"def\s+([a-zA-Z_]\w*)", hint)
            if fn:
                func_name = fn.group(1)
            buggy, fixed = [], []
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if line.startswith("-"):
            buggy.append(line[1:])
        elif line.startswith("+"):
            fixed.append(line[1:])
        elif line.startswith(" "):
            buggy.append(line[1:])
            fixed.append(line[1:])
        # skip "\ No newline at end of file"

    if in_hunk and in_python and (buggy or fixed):
        results.append((func_name, "\n".join(buggy), "\n".join(fixed)))

    return results


def load_bugsinpy(limit=10000):
    _clone_bugsinpy()

    db = get_db()
    collection = db["labeled_functions"]

    projects_dir = CLONE_DIR / "projects"
    inserted = 0
    skipped = 0

    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir() or inserted >= limit:
            break

        bugs_dir = project_dir / "bugs"
        if not bugs_dir.exists():
            continue

        bug_dirs = sorted(
            (d for d in bugs_dir.iterdir() if d.name.isdigit()),
            key=lambda d: int(d.name)
        )

        for bug_dir in bug_dirs:
            if inserted >= limit:
                break

            patch_file = bug_dir / "bug_patch.txt"
            if not patch_file.exists():
                skipped += 1
                continue

            patch_text = patch_file.read_text(encoding="utf-8", errors="replace")
            hunks = _parse_patch(patch_text)

            if not hunks:
                skipped += 1
                continue

            for hunk_func, buggy_code, fixed_code in hunks:
                if buggy_code.strip():
                    names = extract_functions_from_code(buggy_code) or [hunk_func]
                    for name in names:
                        collection.insert_one({
                            "func_name": name,
                            "source": "bugsinpy",
                            "label": 1,
                            "code": buggy_code,
                            "repo": project_dir.name,
                        })
                        inserted += 1

                if fixed_code.strip():
                    names = extract_functions_from_code(fixed_code) or [hunk_func]
                    for name in names:
                        collection.insert_one({
                            "func_name": name,
                            "source": "bugsinpy",
                            "label": 0,
                            "code": fixed_code,
                            "repo": project_dir.name,
                        })
                        inserted += 1

    print(f"inserted {inserted} labeled functions from BugsInPy")
    print(f"skipped {skipped} bugs with missing or non-Python patches")


if __name__ == "__main__":
    load_bugsinpy()
