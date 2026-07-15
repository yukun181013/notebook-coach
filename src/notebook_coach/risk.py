"""Statically detect high-risk execution patterns in notebook snapshots."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Mapping
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
    "credential_read": "high",
    "filesystem_delete": "high",
    "network": "high",
    "package_install": "high",
    "shell": "high",
    "subprocess": "high",
    "syntax_error": "low",
}
_EXPLANATIONS = {
    "credential_read": "Cell may read a credential file.",
    "filesystem_delete": "Cell may delete files or directories.",
    "network": "Cell may access a network resource.",
    "package_install": "Cell may install or change packages.",
    "shell": "Cell may execute a shell command.",
    "subprocess": "Cell may start a subprocess.",
    "syntax_error": "Cell source could not be parsed as Python.",
}

_PACKAGE_MAGIC = re.compile(
    r"^(?P<prefix>!|%{1,2})\s*(?:pip|conda)\s+install\b",
    re.IGNORECASE,
)
_SHELL_CELL_MAGIC = re.compile(r"^%%(?:bash|sh|zsh)\b", re.IGNORECASE)

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


class _RiskVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.categories: set[str] = set()
        self.module_aliases: dict[str, str] = {}
        self.imported_names: dict[str, str] = {}
        self.path_scopes: list[tuple[str, dict[str, int]]] = [("module", {})]
        self.try_exception_state_recorders: list[
            tuple[int, list[list[tuple[str, dict[str, int]]]]]
        ] = []

    def visit(self, node: ast.AST) -> Any:
        if isinstance(
            node, _POTENTIAL_EXCEPTION_NODE_TYPES
        ) and not _potential_exception_is_proven_safe(node):
            self._record_potential_exception()
        return super().visit(node)

    def _resolve_name(self, node: ast.AST) -> str | None:
        name = _qualified_name(node)
        if name is None:
            return None

        if name in self.imported_names:
            return self.imported_names[name]
        root, separator, remainder = name.partition(".")
        imported_name = self.imported_names.get(root)
        if imported_name is not None:
            return imported_name + (separator + remainder if separator else "")
        module = self.module_aliases.get(root)
        if module is not None:
            return module + (separator + remainder if separator else "")
        return name

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
            return _PATH_INSTANCE
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

    def _bind_path_target(self, target: ast.AST, identity: int) -> None:
        if isinstance(target, ast.Name):
            self._bind_path_name(target.id, identity)
        elif isinstance(target, ast.Starred):
            self._bind_path_target(target.value, _PATH_UNKNOWN)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_path_target(element, _PATH_UNKNOWN)

    def _push_path_scope(
        self, scope_kind: str, bindings: dict[str, int] | None = None
    ) -> None:
        self.path_scopes.append((scope_kind, bindings or {}))

    def _pop_path_scope(self) -> None:
        self.path_scopes.pop()

    def _copy_path_state(
        self, state: list[tuple[str, dict[str, int]]] | None = None
    ) -> list[tuple[str, dict[str, int]]]:
        source = self.path_scopes if state is None else state
        return [(scope_kind, dict(bindings)) for scope_kind, bindings in source]

    def _set_path_state(
        self, state: list[tuple[str, dict[str, int]]]
    ) -> None:
        self.path_scopes = self._copy_path_state(state)

    def _visible_before_scope(
        self,
        state: list[tuple[str, dict[str, int]]],
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

    def _merge_path_states(
        self, states: list[list[tuple[str, dict[str, int]]]]
    ) -> None:
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

    def _visit_statements_from(
        self,
        state: list[tuple[str, dict[str, int]]],
        statements: list[ast.stmt],
    ) -> list[tuple[str, dict[str, int]]]:
        self._set_path_state(state)
        for statement in statements:
            self.visit(statement)
        return self._copy_path_state()

    def _record_potential_exception(self) -> None:
        for scope_count, recorder in self.try_exception_state_recorders:
            recorder.append(
                self._copy_path_state(self.path_scopes[:scope_count])
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = _module_root(alias.name)
            local_name = alias.asname or root
            self.module_aliases[local_name] = alias.name if alias.asname else root
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
            self.imported_names[local_name] = f"{module}.{alias.name}"
            self._bind_path_name(
                local_name,
                _PATH_CONSTRUCTOR
                if module == "pathlib" and alias.name == "Path"
                else _PATH_UNKNOWN,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        identity = self._path_identity(node.value)
        for target in node.targets:
            self._bind_path_target(target, identity)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        identity = (
            self._path_identity(node.value)
            if node.value is not None
            else _PATH_UNKNOWN
        )
        if self._path_identity(node.annotation) & _PATH_CONSTRUCTOR:
            identity = _PATH_INSTANCE
        self._bind_path_target(node.target, identity)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        self._bind_path_target(node.target, self._path_identity(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._bind_path_target(node.target, _PATH_UNKNOWN)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        base_state = self._copy_path_state()
        body_state = self._visit_statements_from(base_state, node.body)
        else_state = (
            self._visit_statements_from(base_state, node.orelse)
            if node.orelse
            else base_state
        )
        self._merge_path_states([body_state, else_state])

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        base_state = self._copy_path_state()
        body_state = self._visit_statements_from(base_state, node.body)
        outcomes = [base_state, body_state]
        if node.orelse:
            outcomes.extend(
                self._visit_statements_from(state, node.orelse)
                for state in (base_state, body_state)
            )
        self._merge_path_states(outcomes)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        base_state = self._copy_path_state()
        self._set_path_state(base_state)
        self._bind_path_target(node.target, _PATH_UNKNOWN)
        for statement in node.body:
            self.visit(statement)
        body_state = self._copy_path_state()

        outcomes = [base_state, body_state]
        if node.orelse:
            outcomes.extend(
                self._visit_statements_from(state, node.orelse)
                for state in (base_state, body_state)
            )
        self._merge_path_states(outcomes)

    def visit_For(self, node: ast.For) -> None:
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_for(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_path_target(item.optional_vars, _PATH_UNKNOWN)
        for statement in node.body:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        base_state = self._copy_path_state()
        self._set_path_state(base_state)
        exception_states: list[list[tuple[str, dict[str, int]]]] = []
        self.try_exception_state_recorders.append(
            (len(self.path_scopes), exception_states)
        )
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.try_exception_state_recorders.pop()
        success_state = self._copy_path_state()
        if node.orelse:
            success_state = self._visit_statements_from(
                success_state, node.orelse
            )
        outcomes = [success_state]

        if exception_states:
            self._merge_path_states(exception_states)
            handler_entry_state = self._copy_path_state()
        else:
            handler_entry_state = base_state

        for handler in node.handlers:
            self._set_path_state(handler_entry_state)
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._bind_path_name(handler.name, _PATH_UNKNOWN)
            for statement in handler.body:
                self.visit(statement)
            outcomes.append(self._copy_path_state())

        if node.finalbody:
            outcomes = [
                self._visit_statements_from(state, node.finalbody)
                for state in outcomes
            ]
        self._merge_path_states(outcomes)

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
        local_bindings = {
            name: _PATH_UNKNOWN
            for name in _function_local_names(node.args, node.body)
        }
        self._push_path_scope("function", local_bindings)
        parent_recorders = self.try_exception_state_recorders
        self.try_exception_state_recorders = []
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self.try_exception_state_recorders = parent_recorders
            self._pop_path_scope()

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
        self._push_path_scope("function", local_bindings)
        parent_recorders = self.try_exception_state_recorders
        self.try_exception_state_recorders = []
        try:
            self.visit(node.body)
        finally:
            self.try_exception_state_recorders = parent_recorders
            self._pop_path_scope()

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
        self._push_path_scope("function", local_bindings)
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
            self._pop_path_scope()

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

        self._push_path_scope("class")
        for statement in node.body:
            self.visit(statement)
        self._pop_path_scope()
        self._bind_path_name(node.name, _PATH_UNKNOWN)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._resolve_name(node.func)

        if call_name == "os.system":
            self.categories.add("shell")
        if call_name in _SUBPROCESS_CALLS or (
            call_name is not None and call_name.startswith("subprocess.")
        ):
            self.categories.add("subprocess")
        if call_name in _DELETE_CALLS:
            self.categories.add("filesystem_delete")
        if call_name is not None and _module_root(call_name) in _NETWORK_MODULES:
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
                if path is not None and _looks_like_credential_path(path):
                    self.categories.add("credential_read")

        if call_name in {"builtins.open", "io.open", "open"} and node.args:
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
        if stripped.startswith("!") or _SHELL_CELL_MAGIC.match(stripped):
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


def _scan_source(source: str, visitor: _RiskVisitor) -> set[str]:
    magic_categories = _magic_categories(source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return {
            "syntax_error",
            *magic_categories,
            *_fallback_categories(source),
        }

    visitor.categories = set()
    visitor.visit(tree)
    return set(visitor.categories)


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
        if cell_type != "code":
            continue

        source = cell.get("source")
        if not isinstance(source, Mapping) or "text" not in source:
            raise ValueError("snapshot code cell source.text must be present.")
        source_text = source.get("text")
        if not isinstance(source_text, str):
            raise ValueError("snapshot code cell source.text must be a string.")

        for category in _scan_source(source_text, visitor):
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
