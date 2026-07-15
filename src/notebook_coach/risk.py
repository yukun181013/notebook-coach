"""Statically detect high-risk execution patterns in notebook snapshots."""

from __future__ import annotations

import ast
import hashlib
import operator
import re
from collections.abc import Iterable, Mapping
from typing import Any

from notebook_coach.sanitize import redact_text, summarize_text


_NETWORK_MODULES = {"aiohttp", "httpx", "requests", "socket", "urllib"}
_DELETE_CALLS = {
    "os.remove",
    "os.rmdir",
    "os.unlink",
    "shutil.rmtree",
}
_SUBPROCESS_CALLS = {
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
}
_PATH_DELETE_METHODS = {"rmdir", "unlink"}
_PATH_READ_METHODS = {"open", "read_bytes", "read_text"}
_VALID_CELL_TYPES = {"code", "markdown", "raw"}

_SEVERITIES = {
    "analysis_limit": "high",
    "credential_read": "high",
    "filesystem_delete": "high",
    "network": "high",
    "package_install": "high",
    "risk_metadata": "high",
    "shell": "high",
    "subprocess": "high",
    "syntax_error": "low",
}
_EXPLANATIONS = {
    "analysis_limit": "Cell source exceeded static analysis limits.",
    "credential_read": "Cell may read a credential file.",
    "filesystem_delete": "Cell may delete files or directories.",
    "network": "Cell may access a network resource.",
    "package_install": "Cell may install or change packages.",
    "risk_metadata": "Cell has redacted source without trusted risk metadata.",
    "shell": "Cell may execute a shell command.",
    "subprocess": "Cell may start a subprocess.",
    "syntax_error": "Cell source could not be parsed as Python.",
}

_SOURCE_RISK_CATEGORIES = frozenset(_SEVERITIES) - {"risk_metadata"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REDACTED_MARKER = "[REDACTED]"

_MAX_ANALYSIS_SOURCE_CHARS = 150_000
_MAX_ANALYSIS_AST_NODES = 20_000
_MAX_ANALYSIS_AST_DEPTH = 200

_PACKAGE_MAGIC = re.compile(
    r"^(?P<prefix>!|%{1,2})\s*(?:pip|conda)\s+install\b",
    re.IGNORECASE,
)
_SHELL_LINE_MAGIC = re.compile(r"^%(?:run|sx|system)\b", re.IGNORECASE)
_SHELL_CELL_MAGIC = re.compile(r"^%%(?:bash|sh|zsh|script)\b", re.IGNORECASE)

_FALLBACK_PATTERNS = {
    "filesystem_delete": (
        re.compile(r"\b(?:os\s*\.\s*(?:remove|rmdir|unlink)|shutil\s*\.\s*rmtree)\s*\("),
        re.compile(r"\bPath\s*\([^\r\n]*\)\s*\.\s*(?:rmdir|unlink)\s*\("),
    ),
    "network": (
        re.compile(
            r"\b(?:import|from)\s+(?:aiohttp|httpx|requests|socket|urllib)\b"
        ),
        re.compile(r"\b(?:aiohttp|httpx|requests|socket|urllib)\s*\."),
    ),
    "shell": (re.compile(r"\bos\s*\.\s*system\s*\("),),
    "subprocess": (
        re.compile(r"\b(?:import|from)\s+subprocess\b"),
        re.compile(
            r"\bsubprocess\s*\.\s*"
            r"(?:Popen|call|check_call|check_output|run)\s*\("
        ),
    ),
}


def _safe_explanation(category: str) -> str:
    cleaned, _ = redact_text(_EXPLANATIONS[category])
    single_line = " ".join(cleaned.splitlines())
    return summarize_text(single_line, max_chars=200)["text"]


def _finding(cell_index: int, category: str) -> dict[str, Any]:
    return {
        "cell_index": cell_index,
        "category": category,
        "severity": _SEVERITIES[category],
        "explanation": _safe_explanation(category),
    }


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


def _looks_like_credential_path(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold().rstrip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "~"}]
    if not parts:
        return False

    basename = parts[-1]
    if basename == ".env" or basename.startswith(".env."):
        return True
    if basename == ".netrc":
        return True
    if ".ssh" in parts and (
        basename.startswith("id_") or basename in {"authorized_keys", "known_hosts"}
    ):
        return True
    if ".aws" in parts and basename in {"config", "credentials"}:
        return True
    if ".kube" in parts and basename == "config":
        return True
    return basename == "application_default_credentials.json" and "gcloud" in parts


def _literal_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


_PATH_UNKNOWN = 0
_PATH_MODULE = 1 << 0
_PATH_CONSTRUCTOR = 1 << 1
_PATH_INSTANCE = 1 << 2
_PATH_CREDENTIAL = 1 << 3

_SymbolBinding = frozenset[str]
_PathState = list[tuple[str, dict[str, int]]]
_SymbolState = list[tuple[str, dict[str, _SymbolBinding]]]
_AnalysisState = tuple[_PathState, _SymbolState]
_SYMBOL_SHADOWED: _SymbolBinding = frozenset()

_POTENTIAL_EXCEPTION_NODE_TYPES = (
    ast.Assert,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Attribute,
    ast.AugAssign,
    ast.Await,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Delete,
    ast.Dict,
    ast.DictComp,
    ast.For,
    ast.FormattedValue,
    ast.GeneratorExp,
    ast.IfExp,
    ast.Import,
    ast.ImportFrom,
    ast.ListComp,
    ast.Match,
    ast.Raise,
    ast.Set,
    ast.SetComp,
    ast.Starred,
    ast.Subscript,
    ast.UnaryOp,
    ast.With,
    ast.YieldFrom,
)

_MAX_CONSTANT_DEPTH = 8
_MAX_CONSTANT_ITEMS = 32
_MAX_CONSTANT_TEXT = 1_024
_MAX_CONSTANT_INT_BITS = 64
_UNPROVEN_CONSTANT = object()

_CONSTANT_NUMBER_TYPES = {bool, int, float, complex}
_CONSTANT_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}
_CONSTANT_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: operator.contains(right, left),
    ast.NotIn: lambda left, right: not operator.contains(right, left),
}
_CONSTANT_UNARY_OPERATORS = {
    ast.Not: operator.not_,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}


def _is_bounded_constant_value(value: Any, *, depth: int = 0) -> bool:
    if depth > _MAX_CONSTANT_DEPTH:
        return False
    if value is None or value is Ellipsis or type(value) in {bool, float, complex}:
        return True
    if type(value) is int:
        return value.bit_length() <= _MAX_CONSTANT_INT_BITS
    if type(value) in {str, bytes}:
        return len(value) <= _MAX_CONSTANT_TEXT
    if type(value) in {list, tuple}:
        return len(value) <= _MAX_CONSTANT_ITEMS and all(
            _is_bounded_constant_value(item, depth=depth + 1)
            for item in value
        )
    return False


def _bounded_operation(operation: Any, *values: Any) -> Any:
    try:
        result = operation(*values)
    except (ArithmeticError, LookupError, TypeError, ValueError):
        return _UNPROVEN_CONSTANT
    return result if _is_bounded_constant_value(result) else _UNPROVEN_CONSTANT


def _bounded_constant_value(node: ast.AST, *, depth: int = 0) -> Any:
    if depth > _MAX_CONSTANT_DEPTH:
        return _UNPROVEN_CONSTANT
    if isinstance(node, ast.Constant):
        return (
            node.value
            if _is_bounded_constant_value(node.value, depth=depth)
            else _UNPROVEN_CONSTANT
        )
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > _MAX_CONSTANT_ITEMS:
            return _UNPROVEN_CONSTANT
        values = [
            _bounded_constant_value(element, depth=depth + 1)
            for element in node.elts
        ]
        if any(value is _UNPROVEN_CONSTANT for value in values):
            return _UNPROVEN_CONSTANT
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.UnaryOp):
        operand = _bounded_constant_value(node.operand, depth=depth + 1)
        if operand is _UNPROVEN_CONSTANT:
            return _UNPROVEN_CONSTANT
        operation = _CONSTANT_UNARY_OPERATORS.get(type(node.op))
        if operation is None or (
            not isinstance(node.op, ast.Not)
            and type(operand) not in _CONSTANT_NUMBER_TYPES
        ):
            return _UNPROVEN_CONSTANT
        return _bounded_operation(operation, operand)
    if isinstance(node, ast.BinOp):
        left = _bounded_constant_value(node.left, depth=depth + 1)
        right = _bounded_constant_value(node.right, depth=depth + 1)
        if left is _UNPROVEN_CONSTANT or right is _UNPROVEN_CONSTANT:
            return _UNPROVEN_CONSTANT
        operation = _CONSTANT_BINARY_OPERATORS.get(type(node.op))
        if operation is None or (
            type(left) not in _CONSTANT_NUMBER_TYPES
            or type(right) not in _CONSTANT_NUMBER_TYPES
        ):
            return _UNPROVEN_CONSTANT
        return _bounded_operation(operation, left, right)
    if isinstance(node, ast.BoolOp):
        value: Any = _UNPROVEN_CONSTANT
        for expression in node.values:
            if value is not _UNPROVEN_CONSTANT:
                if isinstance(node.op, ast.And) and not value:
                    return value
                if isinstance(node.op, ast.Or) and value:
                    return value
            value = _bounded_constant_value(expression, depth=depth + 1)
            if value is _UNPROVEN_CONSTANT:
                return _UNPROVEN_CONSTANT
        return value
    if isinstance(node, ast.Compare):
        left = _bounded_constant_value(node.left, depth=depth + 1)
        if left is _UNPROVEN_CONSTANT:
            return _UNPROVEN_CONSTANT
        for comparison, comparator in zip(
            node.ops, node.comparators, strict=True
        ):
            right = _bounded_constant_value(comparator, depth=depth + 1)
            if right is _UNPROVEN_CONSTANT:
                return _UNPROVEN_CONSTANT
            operation = _CONSTANT_COMPARISON_OPERATORS.get(type(comparison))
            if operation is None:
                return _UNPROVEN_CONSTANT
            compared = _bounded_operation(operation, left, right)
            if compared is _UNPROVEN_CONSTANT:
                return _UNPROVEN_CONSTANT
            if not compared:
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        test = _bounded_constant_value(node.test, depth=depth + 1)
        if test is _UNPROVEN_CONSTANT:
            return _UNPROVEN_CONSTANT
        branch = node.body if test else node.orelse
        return _bounded_constant_value(branch, depth=depth + 1)
    if isinstance(node, ast.Subscript):
        value = _bounded_constant_value(node.value, depth=depth + 1)
        index = _bounded_constant_value(node.slice, depth=depth + 1)
        if value is _UNPROVEN_CONSTANT or index is _UNPROVEN_CONSTANT:
            return _UNPROVEN_CONSTANT
        return _bounded_operation(operator.getitem, value, index)
    return _UNPROVEN_CONSTANT


def _potential_exception_is_proven_safe(node: ast.AST) -> bool:
    if isinstance(node, ast.Assert):
        value = _bounded_constant_value(node.test)
        return value is not _UNPROVEN_CONSTANT and bool(value)
    if isinstance(node, ast.GeneratorExp) and node.generators:
        iterable = _bounded_constant_value(node.generators[0].iter)
        return iterable is not _UNPROVEN_CONSTANT and type(iterable) in {
            str,
            bytes,
            list,
            tuple,
        }
    if isinstance(
        node,
        (
            ast.BinOp,
            ast.BoolOp,
            ast.Compare,
            ast.IfExp,
            ast.Subscript,
            ast.UnaryOp,
        ),
    ):
        return _bounded_constant_value(node) is not _UNPROVEN_CONSTANT
    return False


class _FunctionBindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or _module_root(alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    def visit_SetComp(self, node: ast.SetComp) -> None:
        return

    def visit_DictComp(self, node: ast.DictComp) -> None:
        return

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        return

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)


def _argument_names(arguments: ast.arguments) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _function_local_names(
    arguments: ast.arguments, body: list[ast.stmt]
) -> set[str]:
    collector = _FunctionBindingCollector()
    for statement in body:
        collector.visit(statement)
    return (
        _argument_names(arguments) | collector.names
    ) - collector.global_names - collector.nonlocal_names


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _match_pattern_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
    return names


def _match_pattern_is_irrefutable(pattern: ast.pattern) -> bool:
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _match_pattern_is_irrefutable(
            pattern.pattern
        )
    if isinstance(pattern, ast.MatchOr):
        return any(
            _match_pattern_is_irrefutable(item) for item in pattern.patterns
        )
    return False


class _RiskVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.categories: set[str] = set()
        self.path_scopes: _PathState = [("module", {})]
        self.symbol_scopes: _SymbolState = [("module", {})]
        self.try_exception_state_recorders: list[
            tuple[int, list[_AnalysisState]]
        ] = []

    def visit(self, node: ast.AST) -> Any:
        if isinstance(
            node, _POTENTIAL_EXCEPTION_NODE_TYPES
        ) and not _potential_exception_is_proven_safe(node):
            self._record_potential_exception()
        return super().visit(node)

    def _lookup_symbol_name(self, name: str) -> _SymbolBinding | None:
        skip_class_scopes = self.symbol_scopes[-1][0] == "function"
        for scope_kind, bindings in reversed(self.symbol_scopes):
            if skip_class_scopes and scope_kind == "class":
                continue
            if name in bindings:
                return bindings[name]
        return None

    def _resolve_names(self, node: ast.AST) -> _SymbolBinding:
        name = _qualified_name(node)
        if name is None:
            return _SYMBOL_SHADOWED

        root, separator, remainder = name.partition(".")
        binding = self._lookup_symbol_name(root)
        if binding is None:
            return frozenset({name})
        suffix = f".{remainder}" if separator else ""
        return frozenset(f"{qualified}{suffix}" for qualified in binding)

    def _symbol_identity(self, node: ast.AST) -> _SymbolBinding:
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self._resolve_names(node)
        return _SYMBOL_SHADOWED

    def _lookup_path_name(self, name: str) -> int:
        skip_class_scopes = self.path_scopes[-1][0] == "function"
        for scope_kind, bindings in reversed(self.path_scopes):
            if skip_class_scopes and scope_kind == "class":
                continue
            if name in bindings:
                return bindings[name]
        return _PATH_UNKNOWN

    def _path_identity(self, node: ast.AST) -> int:
        if isinstance(node, ast.Name):
            return self._lookup_path_name(node.id)
        if isinstance(node, ast.Attribute) and node.attr == "Path":
            if self._path_identity(node.value) & _PATH_MODULE:
                return _PATH_CONSTRUCTOR
        if isinstance(node, ast.Call) and self._is_path_constructor(node.func):
            identity = _PATH_INSTANCE
            if node.args:
                path = _literal_path(node.args[0])
                if path is not None and _looks_like_credential_path(path):
                    identity |= _PATH_CREDENTIAL
            return identity
        return _PATH_UNKNOWN

    def _is_path_constructor(self, node: ast.AST) -> bool:
        return bool(self._path_identity(node) & _PATH_CONSTRUCTOR)

    def _is_path_constructor_call(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and self._is_path_constructor(node.func)

    def _path_from_constructor_call(self, node: ast.AST) -> str | None:
        if not self._is_path_constructor_call(node):
            return None
        assert isinstance(node, ast.Call)
        if not node.args:
            return None
        return _literal_path(node.args[0])

    def _bind_path_name(self, name: str, identity: int) -> None:
        self.path_scopes[-1][1][name] = identity

    def _bind_symbol_name(self, name: str, identity: _SymbolBinding) -> None:
        self.symbol_scopes[-1][1][name] = identity

    def _bind_path_target(self, target: ast.AST, identity: int) -> None:
        if isinstance(target, ast.Name):
            self._bind_path_name(target.id, identity)
        elif isinstance(target, ast.Starred):
            self._bind_path_target(target.value, _PATH_UNKNOWN)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_path_target(element, _PATH_UNKNOWN)

    def _bind_symbol_target(
        self, target: ast.AST, identity: _SymbolBinding
    ) -> None:
        if isinstance(target, ast.Name):
            self._bind_symbol_name(target.id, identity)
        elif isinstance(target, ast.Starred):
            self._bind_symbol_target(target.value, _SYMBOL_SHADOWED)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_symbol_target(element, _SYMBOL_SHADOWED)

    def _push_scope(
        self,
        scope_kind: str,
        path_bindings: dict[str, int] | None = None,
        symbol_bindings: dict[str, _SymbolBinding] | None = None,
    ) -> None:
        self.path_scopes.append((scope_kind, path_bindings or {}))
        self.symbol_scopes.append((scope_kind, symbol_bindings or {}))

    def _pop_scope(self) -> None:
        self.path_scopes.pop()
        self.symbol_scopes.pop()

    def _copy_path_state(
        self, state: _PathState | None = None
    ) -> _PathState:
        source = self.path_scopes if state is None else state
        return [(scope_kind, dict(bindings)) for scope_kind, bindings in source]

    def _copy_symbol_state(
        self, state: _SymbolState | None = None
    ) -> _SymbolState:
        source = self.symbol_scopes if state is None else state
        return [(scope_kind, dict(bindings)) for scope_kind, bindings in source]

    def _copy_analysis_state(self) -> _AnalysisState:
        return self._copy_path_state(), self._copy_symbol_state()

    def _set_path_state(self, state: _PathState) -> None:
        self.path_scopes = self._copy_path_state(state)

    def _set_symbol_state(self, state: _SymbolState) -> None:
        self.symbol_scopes = self._copy_symbol_state(state)

    def _set_analysis_state(self, state: _AnalysisState) -> None:
        self._set_path_state(state[0])
        self._set_symbol_state(state[1])

    def _visible_before_scope(
        self,
        state: _PathState,
        scope_index: int,
        name: str,
    ) -> int:
        skip_class_scopes = state[-1][0] == "function"
        for scope_kind, bindings in reversed(state[:scope_index]):
            if skip_class_scopes and scope_kind == "class":
                continue
            if name in bindings:
                return bindings[name]
        return _PATH_UNKNOWN

    def _visible_symbol_before_scope(
        self,
        state: _SymbolState,
        scope_index: int,
        name: str,
    ) -> _SymbolBinding | None:
        skip_class_scopes = state[-1][0] == "function"
        for scope_kind, bindings in reversed(state[:scope_index]):
            if skip_class_scopes and scope_kind == "class":
                continue
            if name in bindings:
                return bindings[name]
        return None

    def _merge_path_states(self, states: list[_PathState]) -> None:
        if not states:
            return
        scope_count = len(states[0])
        if any(len(state) != scope_count for state in states):
            raise RuntimeError("Path scope stack became unbalanced.")

        merged: list[tuple[str, dict[str, int]]] = []
        for scope_index in range(scope_count):
            scope_kind = states[0][scope_index][0]
            if any(state[scope_index][0] != scope_kind for state in states):
                raise RuntimeError("Path scope kinds became inconsistent.")
            names = {
                name
                for state in states
                for name in state[scope_index][1]
            }
            bindings: dict[str, int] = {}
            for name in names:
                identity = _PATH_UNKNOWN
                present = False
                for state in states:
                    branch_bindings = state[scope_index][1]
                    if name in branch_bindings:
                        identity |= branch_bindings[name]
                        present = True
                    else:
                        identity |= self._visible_before_scope(
                            state, scope_index, name
                        )
                if present:
                    bindings[name] = identity
            merged.append((scope_kind, bindings))
        self.path_scopes = merged

    def _merge_symbol_states(self, states: list[_SymbolState]) -> None:
        if not states:
            return
        scope_count = len(states[0])
        if any(len(state) != scope_count for state in states):
            raise RuntimeError("Symbol scope stack became unbalanced.")

        merged: _SymbolState = []
        for scope_index in range(scope_count):
            scope_kind = states[0][scope_index][0]
            if any(state[scope_index][0] != scope_kind for state in states):
                raise RuntimeError("Symbol scope kinds became inconsistent.")
            names = {
                name
                for state in states
                for name in state[scope_index][1]
            }
            bindings: dict[str, _SymbolBinding] = {}
            for name in names:
                identities: set[str] = set()
                for state in states:
                    branch_bindings = state[scope_index][1]
                    if name in branch_bindings:
                        identities.update(branch_bindings[name])
                        continue
                    visible = self._visible_symbol_before_scope(
                        state, scope_index, name
                    )
                    identities.update(visible if visible is not None else {name})
                bindings[name] = frozenset(identities)
            merged.append((scope_kind, bindings))
        self.symbol_scopes = merged

    def _merge_analysis_states(self, states: list[_AnalysisState]) -> None:
        self._merge_path_states([state[0] for state in states])
        self._merge_symbol_states([state[1] for state in states])

    def _visit_statements_from(
        self,
        state: _AnalysisState,
        statements: list[ast.stmt],
    ) -> _AnalysisState:
        self._set_analysis_state(state)
        for statement in statements:
            self.visit(statement)
        return self._copy_analysis_state()

    def _record_potential_exception(self) -> None:
        for scope_count, recorder in self.try_exception_state_recorders:
            recorder.append(
                (
                    self._copy_path_state(self.path_scopes[:scope_count]),
                    self._copy_symbol_state(self.symbol_scopes[:scope_count]),
                )
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = _module_root(alias.name)
            local_name = alias.asname or root
            qualified_name = alias.name if alias.asname else root
            self._bind_symbol_name(local_name, frozenset({qualified_name}))
            self._bind_path_name(
                local_name,
                _PATH_MODULE if alias.name == "pathlib" else _PATH_UNKNOWN,
            )
            if root == "subprocess":
                self.categories.add("subprocess")
            if root in _NETWORK_MODULES:
                self.categories.add("network")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = _module_root(module) if module else ""
        if root == "subprocess":
            self.categories.add("subprocess")
        if root in _NETWORK_MODULES:
            self.categories.add("network")

        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            qualified_name = f"{module}.{alias.name}"
            self._bind_symbol_name(local_name, frozenset({qualified_name}))
            self._bind_path_name(
                local_name,
                _PATH_CONSTRUCTOR
                if module == "pathlib" and alias.name == "Path"
                else _PATH_UNKNOWN,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(
            isinstance(target, (ast.List, ast.Tuple, ast.Starred))
            for target in node.targets
        ):
            self._record_potential_exception()
        self.generic_visit(node)
        path_identity = self._path_identity(node.value)
        symbol_identity = self._symbol_identity(node.value)
        for target in node.targets:
            self._bind_path_target(target, path_identity)
            self._bind_symbol_target(target, symbol_identity)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        path_identity = (
            self._path_identity(node.value)
            if node.value is not None
            else _PATH_UNKNOWN
        )
        if self._path_identity(node.annotation) & _PATH_CONSTRUCTOR:
            path_identity |= _PATH_INSTANCE
        symbol_identity = (
            self._symbol_identity(node.value)
            if node.value is not None
            else _SYMBOL_SHADOWED
        )
        self._bind_path_target(node.target, path_identity)
        self._bind_symbol_target(node.target, symbol_identity)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        self._bind_path_target(node.target, self._path_identity(node.value))
        self._bind_symbol_target(node.target, self._symbol_identity(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._bind_path_target(node.target, _PATH_UNKNOWN)
        self._bind_symbol_target(node.target, _SYMBOL_SHADOWED)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        base_state = self._copy_analysis_state()
        body_state = self._visit_statements_from(base_state, node.body)
        else_state = (
            self._visit_statements_from(base_state, node.orelse)
            if node.orelse
            else base_state
        )
        self._merge_analysis_states([body_state, else_state])

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        base_path_state = self._copy_path_state()
        base_symbol_state = self._copy_symbol_state()
        path_outcomes = [base_path_state]
        symbol_outcomes: list[_SymbolState] = []
        symbol_match_is_exhaustive = False
        for case in node.cases:
            self._set_path_state(base_path_state)
            self._set_symbol_state(base_symbol_state)
            self.visit(case.pattern)
            for name in _match_pattern_names(case.pattern):
                self._bind_path_name(name, _PATH_UNKNOWN)
                self._bind_symbol_name(name, _SYMBOL_SHADOWED)
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)
            path_outcomes.append(self._copy_path_state())
            symbol_outcomes.append(self._copy_symbol_state())
            if case.guard is None and _match_pattern_is_irrefutable(case.pattern):
                symbol_match_is_exhaustive = True
        self._merge_path_states(path_outcomes)
        if not symbol_match_is_exhaustive:
            symbol_outcomes.insert(0, base_symbol_state)
        self._merge_symbol_states(symbol_outcomes)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        base_state = self._copy_analysis_state()
        body_state = self._visit_statements_from(base_state, node.body)
        outcomes = [base_state, body_state]
        if node.orelse:
            outcomes.extend(
                self._visit_statements_from(state, node.orelse)
                for state in (base_state, body_state)
            )
        self._merge_analysis_states(outcomes)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        base_state = self._copy_analysis_state()
        self._set_analysis_state(base_state)
        self._bind_path_target(node.target, _PATH_UNKNOWN)
        self._bind_symbol_target(node.target, _SYMBOL_SHADOWED)
        for statement in node.body:
            self.visit(statement)
        body_state = self._copy_analysis_state()

        outcomes = [base_state, body_state]
        if node.orelse:
            outcomes.extend(
                self._visit_statements_from(state, node.orelse)
                for state in (base_state, body_state)
            )
        self._merge_analysis_states(outcomes)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_path_target(item.optional_vars, _PATH_UNKNOWN)
                self._bind_symbol_target(
                    item.optional_vars, _SYMBOL_SHADOWED
                )
        for statement in node.body:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        base_state = self._copy_analysis_state()
        self._set_analysis_state(base_state)
        exception_states: list[_AnalysisState] = []
        self.try_exception_state_recorders.append(
            (len(self.path_scopes), exception_states)
        )
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.try_exception_state_recorders.pop()
        success_state = self._copy_analysis_state()
        if node.orelse:
            success_state = self._visit_statements_from(
                success_state, node.orelse
            )
        outcomes = [success_state]

        if exception_states:
            self._merge_analysis_states(exception_states)
            handler_entry_state = self._copy_analysis_state()
        else:
            handler_entry_state = base_state

        for handler in node.handlers:
            self._set_analysis_state(handler_entry_state)
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._bind_path_name(handler.name, _PATH_UNKNOWN)
                self._bind_symbol_name(handler.name, _SYMBOL_SHADOWED)
            for statement in handler.body:
                self.visit(statement)
            outcomes.append(self._copy_analysis_state())

        if node.finalbody:
            outcomes = [
                self._visit_statements_from(state, node.finalbody)
                for state in outcomes
            ]
        self._merge_analysis_states(outcomes)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try(node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for argument in (node.args.vararg, node.args.kwarg):
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        self._bind_path_name(node.name, _PATH_UNKNOWN)
        self._bind_symbol_name(node.name, _SYMBOL_SHADOWED)
        local_names = _function_local_names(node.args, node.body)
        path_bindings = {
            name: _PATH_UNKNOWN
            for name in local_names
        }
        symbol_bindings = {
            name: _SYMBOL_SHADOWED
            for name in local_names
        }
        self._push_scope("function", path_bindings, symbol_bindings)
        parent_recorders = self.try_exception_state_recorders
        self.try_exception_state_recorders = []
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.try_exception_state_recorders = parent_recorders
            self._pop_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        local_bindings = {
            name: _PATH_UNKNOWN for name in _argument_names(node.args)
        }
        symbol_bindings = {
            name: _SYMBOL_SHADOWED for name in _argument_names(node.args)
        }
        self._push_scope("function", local_bindings, symbol_bindings)
        parent_recorders = self.try_exception_state_recorders
        self.try_exception_state_recorders = []
        try:
            self.visit(node.body)
        finally:
            self.try_exception_state_recorders = parent_recorders
            self._pop_scope()

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_nodes: tuple[ast.AST, ...],
        *,
        deferred: bool = False,
    ) -> None:
        if not generators:
            for result_node in result_nodes:
                self.visit(result_node)
            return

        self.visit(generators[0].iter)
        local_bindings = {
            name: _PATH_UNKNOWN
            for generator in generators
            for name in _target_names(generator.target)
        }
        symbol_bindings = {
            name: _SYMBOL_SHADOWED
            for generator in generators
            for name in _target_names(generator.target)
        }
        self._push_scope("function", local_bindings, symbol_bindings)
        parent_recorders = self.try_exception_state_recorders
        if deferred:
            self.try_exception_state_recorders = []
        try:
            for condition in generators[0].ifs:
                self.visit(condition)
            for generator in generators[1:]:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for result_node in result_nodes:
                self.visit(result_node)
        finally:
            self.try_exception_state_recorders = parent_recorders
            self._pop_scope()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(
            node.generators, (node.elt,), deferred=True
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

        self._push_scope("class")
        for statement in node.body:
            self.visit(statement)
        self._pop_scope()
        self._bind_path_name(node.name, _PATH_UNKNOWN)
        self._bind_symbol_name(node.name, _SYMBOL_SHADOWED)

    def visit_Call(self, node: ast.Call) -> None:
        call_names = self._resolve_names(node.func)

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "get_ipython"
        ):
            method = node.func.attr
            magic_name = _literal_path(node.args[0]) if node.args else None
            if method in {"system", "getoutput"}:
                self.categories.add("shell")
            elif method == "run_line_magic" and magic_name in {
                "run",
                "sx",
                "system",
            }:
                self.categories.add("shell")
            elif method == "run_cell_magic" and magic_name in {
                "bash",
                "script",
                "sh",
                "zsh",
            }:
                self.categories.add("shell")

        if "os.system" in call_names:
            self.categories.add("shell")
        if call_names & _SUBPROCESS_CALLS or any(
            call_name.startswith("subprocess.") for call_name in call_names
        ):
            self.categories.add("subprocess")
        if call_names & _DELETE_CALLS:
            self.categories.add("filesystem_delete")
        if any(
            _module_root(call_name) in _NETWORK_MODULES
            for call_name in call_names
        ):
            self.categories.add("network")

        if isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            receiver_identity = self._path_identity(receiver)
            if (
                node.func.attr in _PATH_DELETE_METHODS
                and receiver_identity
                & (_PATH_INSTANCE | _PATH_CONSTRUCTOR)
            ):
                self.categories.add("filesystem_delete")
            if node.func.attr in _PATH_READ_METHODS:
                path = self._path_from_constructor_call(receiver)
                if receiver_identity & _PATH_CREDENTIAL or (
                    path is not None and _looks_like_credential_path(path)
                ):
                    self.categories.add("credential_read")

        if call_names & {"builtins.open", "io.open", "open"} and node.args:
            path = _literal_path(node.args[0])
            if path is not None and _looks_like_credential_path(path):
                self.categories.add("credential_read")

        self.generic_visit(node)


def _magic_categories(source: str) -> set[str]:
    categories: set[str] = set()
    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue

        package_magic = _PACKAGE_MAGIC.match(stripped)
        if package_magic is not None:
            categories.add("package_install")
            if package_magic.group("prefix") == "!":
                categories.add("shell")
        if (
            stripped.startswith("!")
            or _SHELL_LINE_MAGIC.match(stripped)
            or _SHELL_CELL_MAGIC.match(stripped)
        ):
            categories.add("shell")
    return categories


def _fallback_categories(source: str) -> set[str]:
    categories: set[str] = set()
    for category, patterns in _FALLBACK_PATTERNS.items():
        if any(pattern.search(source) for pattern in patterns):
            categories.add(category)

    credential_literals = re.findall(
        r"(?:open|Path)\s*\(\s*(['\"])(.*?)\1",
        source,
        flags=re.DOTALL,
    )
    if any(_looks_like_credential_path(path) for _, path in credential_literals):
        categories.add("credential_read")
    return categories


def _ast_exceeds_analysis_limits(tree: ast.AST) -> bool:
    stack = [(tree, 0)]
    node_count = 0
    while stack:
        node, depth = stack.pop()
        node_count += 1
        if (
            node_count > _MAX_ANALYSIS_AST_NODES
            or depth > _MAX_ANALYSIS_AST_DEPTH
        ):
            return True
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return False


def _scan_source(source: str, visitor: _RiskVisitor) -> set[str]:
    if len(source) > _MAX_ANALYSIS_SOURCE_CHARS:
        return {"analysis_limit"}

    magic_categories = _magic_categories(source)
    try:
        tree = ast.parse(source)
    except (MemoryError, RecursionError):
        return {"analysis_limit", *magic_categories}
    except (SyntaxError, ValueError, TypeError):
        return {
            "syntax_error",
            *magic_categories,
            *_fallback_categories(source),
        }

    try:
        if _ast_exceeds_analysis_limits(tree):
            return {"analysis_limit", *magic_categories}
    except (MemoryError, RecursionError):
        return {"analysis_limit", *magic_categories}

    visitor.categories = set()
    try:
        visitor.visit(tree)
    except (MemoryError, RecursionError):
        return {"analysis_limit", *magic_categories}
    return set(visitor.categories)


def build_source_risk_metadata(sources: Iterable[str]) -> list[dict[str, Any]]:
    """Statically scan original sources and return source-free risk metadata."""

    visitor = _RiskVisitor()
    metadata: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        metadata.append(
            {
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "categories": sorted(_scan_source(source, visitor)),
            }
        )
    return metadata


def _trusted_risk_categories(
    cell: Mapping[str, Any], source: Mapping[str, Any]
) -> set[str] | None:
    metadata = cell.get("risk")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "source_sha256",
        "categories",
    }:
        return None

    source_sha256 = source.get("sha256")
    metadata_sha256 = metadata.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or _SHA256_PATTERN.fullmatch(source_sha256) is None
        or metadata_sha256 != source_sha256
    ):
        return None

    categories = metadata.get("categories")
    if (
        not isinstance(categories, list)
        or any(
            not isinstance(category, str)
            or category not in _SOURCE_RISK_CATEGORIES
            for category in categories
        )
        or categories != sorted(set(categories))
    ):
        return None
    return set(categories)


def _validated_cells(snapshot: Any) -> list[Mapping[str, Any]]:
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be a mapping with a cells list.")
    cells = snapshot.get("cells")
    if not isinstance(cells, list):
        raise ValueError("snapshot cells must be a list.")
    if not all(isinstance(cell, Mapping) for cell in cells):
        raise ValueError("snapshot cells must contain mappings.")
    return cells


def scan_snapshot(snapshot: Any) -> dict[str, Any]:
    """Return deterministic static findings without executing notebook code."""

    findings: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    visitor = _RiskVisitor()
    for cell in _validated_cells(snapshot):
        cell_index = cell.get("index")
        if (
            isinstance(cell_index, bool)
            or not isinstance(cell_index, int)
            or cell_index < 0
        ):
            raise ValueError("snapshot cell index must be a non-negative integer.")
        if cell_index in seen_indexes:
            raise ValueError("snapshot contains a duplicate cell index.")
        seen_indexes.add(cell_index)

        cell_type = cell.get("cell_type")
        if not isinstance(cell_type, str):
            raise ValueError("snapshot cell_type must be a string.")
        if cell_type not in _VALID_CELL_TYPES:
            raise ValueError(
                "snapshot cell_type must be code, markdown, or raw."
            )

        source = cell.get("source")
        if not isinstance(source, Mapping) or "text" not in source:
            raise ValueError("snapshot cell source.text must be present.")
        source_text = source.get("text")
        if not isinstance(source_text, str):
            raise ValueError("snapshot cell source.text must be a string.")

        if cell_type != "code":
            continue

        categories = _scan_source(source_text, visitor)
        trusted_categories = _trusted_risk_categories(cell, source)
        if trusted_categories is not None:
            categories.update(trusted_categories)
        elif _REDACTED_MARKER in source_text:
            categories.add("risk_metadata")

        for category in categories:
            findings.append(_finding(cell_index, category))

    unique_findings = {
        (
            finding["cell_index"],
            finding["category"],
            finding["explanation"],
        ): finding
        for finding in findings
    }
    ordered_findings = [
        unique_findings[key]
        for key in sorted(unique_findings)
    ]
    return {
        "blocked": any(
            finding["severity"] == "high" for finding in ordered_findings
        ),
        "findings": ordered_findings,
    }
