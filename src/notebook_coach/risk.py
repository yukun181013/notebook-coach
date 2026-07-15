"""Statically detect high-risk execution patterns in notebook snapshots."""

from __future__ import annotations

import ast
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


_PATH_UNKNOWN = "unknown"
_PATH_MODULE = "module"
_PATH_CONSTRUCTOR = "constructor"
_PATH_INSTANCE = "instance"


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


class _RiskVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.categories: set[str] = set()
        self.module_aliases: dict[str, str] = {}
        self.imported_names: dict[str, str] = {}
        self.path_scopes: list[tuple[str, dict[str, str]]] = [("module", {})]

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

    def _lookup_path_name(self, name: str) -> str:
        skip_class_scopes = self.path_scopes[-1][0] == "function"
        for scope_kind, bindings in reversed(self.path_scopes):
            if skip_class_scopes and scope_kind == "class":
                continue
            if name in bindings:
                return bindings[name]
        return _PATH_UNKNOWN

    def _path_identity(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self._lookup_path_name(node.id)
        if isinstance(node, ast.Attribute) and node.attr == "Path":
            if self._path_identity(node.value) == _PATH_MODULE:
                return _PATH_CONSTRUCTOR
        if isinstance(node, ast.Call) and self._is_path_constructor(node.func):
            return _PATH_INSTANCE
        return _PATH_UNKNOWN

    def _is_path_constructor(self, node: ast.AST) -> bool:
        return self._path_identity(node) == _PATH_CONSTRUCTOR

    def _is_path_constructor_call(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and self._is_path_constructor(node.func)

    def _path_from_constructor_call(self, node: ast.AST) -> str | None:
        if not self._is_path_constructor_call(node):
            return None
        assert isinstance(node, ast.Call)
        if not node.args:
            return None
        return _literal_path(node.args[0])

    def _bind_path_name(self, name: str, identity: str) -> None:
        self.path_scopes[-1][1][name] = identity

    def _bind_path_target(self, target: ast.AST, identity: str) -> None:
        if isinstance(target, ast.Name):
            self._bind_path_name(target.id, identity)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_path_target(element, _PATH_UNKNOWN)

    def _push_path_scope(
        self, scope_kind: str, bindings: dict[str, str] | None = None
    ) -> None:
        self.path_scopes.append((scope_kind, bindings or {}))

    def _pop_path_scope(self) -> None:
        self.path_scopes.pop()

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
        if self._path_identity(node.annotation) == _PATH_CONSTRUCTOR:
            identity = _PATH_INSTANCE
        self._bind_path_target(node.target, identity)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.generic_visit(node)
        self._bind_path_target(node.target, self._path_identity(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        self._bind_path_target(node.target, _PATH_UNKNOWN)

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
        for statement in node.body:
            self.visit(statement)
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
        self.visit(node.body)
        self._pop_path_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

        self._bind_path_name(node.name, _PATH_UNKNOWN)
        self._push_path_scope("class")
        for statement in node.body:
            self.visit(statement)
        self._pop_path_scope()

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
            if (
                node.func.attr in _PATH_DELETE_METHODS
                and (
                    self._path_identity(receiver) == _PATH_INSTANCE
                    or self._path_identity(receiver) == _PATH_CONSTRUCTOR
                )
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


def _scan_source(source: str) -> set[str]:
    magic_categories = _magic_categories(source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return {
            "syntax_error",
            *magic_categories,
            *_fallback_categories(source),
        }

    visitor = _RiskVisitor()
    visitor.visit(tree)
    return visitor.categories


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

        for category in _scan_source(source_text):
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
