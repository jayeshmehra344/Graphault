import ast
import textwrap
import torch
from torch_geometric.data import Data
from typing import Optional

# ── Vocabulary ──────────────────────────────────────────────────────────────

NODE_TYPES = [
    # Module / function roots
    'Module', 'FunctionDef', 'AsyncFunctionDef', 'ClassDef',
    # Statements
    'Return', 'Delete', 'Assign', 'AugAssign', 'AnnAssign',
    'For', 'AsyncFor', 'While', 'If', 'With', 'AsyncWith',
    'Raise', 'Try', 'ExceptHandler', 'Assert',
    'Import', 'ImportFrom', 'Global', 'Nonlocal',
    'Expr', 'Pass', 'Break', 'Continue',
    # Expressions
    'BoolOp', 'NamedExpr', 'BinOp', 'UnaryOp', 'Lambda', 'IfExp',
    'Dict', 'Set', 'List', 'Tuple',
    'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
    'Await', 'Yield', 'YieldFrom',
    'Compare', 'Call', 'FormattedValue', 'JoinedStr',
    'Constant', 'Attribute', 'Subscript', 'Starred', 'Name', 'Slice',
    # Boolean / comparison operators
    'And', 'Or',
    'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
    # Arithmetic / bitwise operators
    'Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow', 'FloorDiv', 'MatMult',
    'BitOr', 'BitAnd', 'BitXor', 'LShift', 'RShift',
    'Not', 'Invert', 'UAdd', 'USub',
    # Function signature nodes
    'arguments', 'arg', 'keyword',
    # Misc
    'alias', 'withitem',
    'UNKNOWN',
]

NODE_TYPE_INDEX: dict[str, int] = {t: i for i, t in enumerate(NODE_TYPES)}
VOCAB_SIZE: int = len(NODE_TYPES)

# Edge type indices stored in edge_attr
EDGE_PARENT_CHILD = 0   # structural: parent → child
EDGE_CHILD_PARENT = 1   # structural: child → parent (reverse)
EDGE_NEXT_STMT    = 2   # control flow: statement → next statement
NUM_EDGE_TYPES    = 3


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_code(code: str) -> Optional[ast.AST]:
    """Try to parse code as-is; fall back to dedented version."""
    code = textwrap.dedent(code)
    for attempt in (code, f"def _wrapper():\n{textwrap.indent(code, '    ')}"):
        try:
            return ast.parse(attempt)
        except SyntaxError:
            continue
    return None


def _find_func_root(tree: ast.AST) -> Optional[ast.AST]:
    """Return the outermost FunctionDef/AsyncFunctionDef, or Module if none."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    # No function def found — use the module body directly
    if isinstance(tree, ast.Module) and tree.body:
        return tree
    return None


def _assign_node_ids(root: ast.AST) -> tuple[dict, list[int]]:
    """
    DFS traversal of root subtree.
    Returns (node_ids: {id(node) -> graph_idx}, type_indices: [int]).
    """
    node_ids: dict[int, int] = {}
    type_indices: list[int] = []

    def _visit(node: ast.AST):
        key = id(node)
        if key in node_ids:
            return
        node_ids[key] = len(node_ids)
        type_name = type(node).__name__
        type_indices.append(NODE_TYPE_INDEX.get(type_name, NODE_TYPE_INDEX['UNKNOWN']))
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(root)
    return node_ids, type_indices


def _add_parent_child_edges(
    root: ast.AST,
    node_ids: dict,
    src: list, dst: list, etype: list,
):
    """Add bidirectional parent↔child edges for every edge in the AST."""
    for node in ast.walk(root):
        p = node_ids.get(id(node))
        if p is None:
            continue
        for child in ast.iter_child_nodes(node):
            c = node_ids.get(id(child))
            if c is None:
                continue
            src.append(p); dst.append(c); etype.append(EDGE_PARENT_CHILD)
            src.append(c); dst.append(p); etype.append(EDGE_CHILD_PARENT)


def _add_control_flow_edges(
    root: ast.AST,
    node_ids: dict,
    src: list, dst: list, etype: list,
):
    """
    Add next-statement edges between consecutive statements
    in every statement block (body, orelse, handlers, finalbody).
    """
    def _seq_edges(stmts: list):
        for i in range(len(stmts) - 1):
            a = node_ids.get(id(stmts[i]))
            b = node_ids.get(id(stmts[i + 1]))
            if a is not None and b is not None:
                src.append(a); dst.append(b); etype.append(EDGE_NEXT_STMT)

    for node in ast.walk(root):
        for attr in ('body', 'orelse', 'handlers', 'finalbody'):
            block = getattr(node, attr, None)
            if isinstance(block, list) and len(block) > 1:
                _seq_edges(block)


# ── Public API ────────────────────────────────────────────────────────────────

def build_ast_graph(code: str, label: int = 0) -> Optional[Data]:
    """
    Build a per-function AST graph from source code.

    Nodes : one AST node per graph node, feature = one-hot of node type
            shape [N, VOCAB_SIZE]
    Edges : parent→child, child→parent (structural) +
            stmt→next_stmt (control flow)
    edge_attr : LongTensor of shape [E] — edge type index (0/1/2)
    y     : FloatTensor([label])

    Returns None if the code cannot be parsed.
    """
    tree = _parse_code(code)
    if tree is None:
        return None

    root = _find_func_root(tree)
    if root is None:
        return None

    node_ids, type_indices = _assign_node_ids(root)
    n = len(type_indices)
    if n == 0:
        return None

    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []

    _add_parent_child_edges(root, node_ids, src, dst, etype)
    _add_control_flow_edges(root, node_ids, src, dst, etype)

    # One-hot node features
    x = torch.zeros(n, VOCAB_SIZE, dtype=torch.float)
    for i, t in enumerate(type_indices):
        x[i, t] = 1.0

    edge_index = (torch.tensor([src, dst], dtype=torch.long)
                  if src else torch.zeros((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(etype, dtype=torch.long)
                 if etype else torch.zeros(0, dtype=torch.long))

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(label)], dtype=torch.float),
    )


def build_graph_skeleton(code: str, label: int = 0) -> Optional[tuple]:
    """
    Build edge structure + label for a function, without node features.

    Returns (edge_index, edge_attr, y, num_nodes) or None if unparseable.
    Node count and order match build_ast_graph / precompute_codebert.py's
    pre-order DFS (_assign_node_ids), so row i of a CodeBERT feature cache
    entry corresponds to node i here.
    """
    tree = _parse_code(code)
    if tree is None:
        return None

    root = _find_func_root(tree)
    if root is None:
        return None

    node_ids, type_indices = _assign_node_ids(root)
    n = len(type_indices)
    if n == 0:
        return None

    src: list[int] = []
    dst: list[int] = []
    etype: list[int] = []

    _add_parent_child_edges(root, node_ids, src, dst, etype)
    _add_control_flow_edges(root, node_ids, src, dst, etype)

    edge_index = (torch.tensor([src, dst], dtype=torch.long)
                  if src else torch.zeros((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(etype, dtype=torch.long)
                 if etype else torch.zeros(0, dtype=torch.long))

    return edge_index, edge_attr, torch.tensor([float(label)], dtype=torch.float), n


def build_ast_graph_codebert(code: str, feat: torch.Tensor, label: int = 0) -> Optional[Data]:
    """
    Build a per-function AST graph using precomputed CodeBERT node features.

    feat must be a FloatTensor[num_nodes, 768] in the same pre-order DFS
    order as _assign_node_ids (i.e. the order produced by
    precompute_codebert.py). Returns None if the code cannot be parsed.

    Raises ValueError if feat.shape[0] does not match the AST node count —
    a mismatch means the node ordering is misaligned and must not be
    silently zero-filled or truncated.
    """
    skeleton = build_graph_skeleton(code, label=label)
    if skeleton is None:
        return None
    edge_index, edge_attr, y, n = skeleton

    if feat.shape[0] != n:
        raise ValueError(
            f"CodeBERT feature/AST node count mismatch: "
            f"cache has {feat.shape[0]} nodes, AST has {n} nodes"
        )

    return Data(x=feat, edge_index=edge_index, edge_attr=edge_attr, y=y)


def get_dfs_ordered_nodes(code: str) -> list:
    """
    Return the AST node objects in the same pre-order DFS order that
    _assign_node_ids uses, rooted at _find_func_root(tree).

    node[i] in the returned list corresponds exactly to graph node i
    produced by build_ast_graph / build_graph_skeleton for the same code.
    Returns an empty list if the code cannot be parsed.
    """
    tree = _parse_code(code)
    if tree is None:
        return []
    root = _find_func_root(tree)
    if root is None:
        return []

    visited: set = set()
    ordered: list = []

    def _visit(node: ast.AST) -> None:
        key = id(node)
        if key in visited:
            return
        visited.add(key)
        ordered.append(node)
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(root)
    return ordered


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CLEAN_FUNC = """
def add_positive(values):
    total = 0
    for v in values:
        if v > 0:
            total += v
    return total
"""

    BUGGY_FUNC = """
def find_index(items, target):
    # off-by-one: should be range(len(items))
    for i in range(len(items) - 1):
        if items[i] == target:
            return i
    return -1
"""

    SNIPPET_NO_DEF = """
    result = []
    for x in data:
        if x is not None:
            try:
                result.append(process(x))
            except ValueError:
                pass
    return result
"""

    for name, code, label in [
        ("clean function",        CLEAN_FUNC,    0),
        ("buggy function",        BUGGY_FUNC,    1),
        ("raw snippet (no def)",  SNIPPET_NO_DEF, 1),
    ]:
        data = build_ast_graph(code, label=label)
        if data is None:
            print(f"[{name}] FAILED to parse")
            continue

        n_nodes = data.x.shape[0]
        n_edges = data.edge_index.shape[1]
        n_pc    = (data.edge_attr == EDGE_PARENT_CHILD).sum().item()
        n_cp    = (data.edge_attr == EDGE_CHILD_PARENT).sum().item()
        n_cf    = (data.edge_attr == EDGE_NEXT_STMT).sum().item()

        # Dominant node types
        type_ids = data.x.argmax(dim=1).tolist()
        from collections import Counter
        top = Counter(NODE_TYPES[i] for i in type_ids).most_common(5)

        print(f"\n[{name}]  label={int(data.y.item())}")
        print(f"  nodes      : {n_nodes}")
        print(f"  edges      : {n_edges}  "
              f"(par->child: {n_pc}, child->par: {n_cp}, next-stmt: {n_cf})")
        print(f"  x shape    : {list(data.x.shape)}")
        print(f"  top types  : {top}")

    print(f"\nVOCAB_SIZE = {VOCAB_SIZE}")
    print("build_ast_graph() returns Data(x, edge_index, edge_attr, y)")
