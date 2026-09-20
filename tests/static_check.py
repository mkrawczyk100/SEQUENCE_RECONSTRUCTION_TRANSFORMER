"""Prosty analizator statyczny: niezdefiniowane nazwy i martwe importy.

Zastępuje pyflakes, który nie jest dostępny w środowisku bez sieci.
Wykrywa najczęstszą klasę błędów w kodzie, którego nie da się
uruchomić: literówki w nazwach i brakujące importy.
"""

import ast
import builtins
import os
import sys

BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "annotations", "self", "cls",
}


class ScopeCollector(ast.NodeVisitor):
    """Zbiera nazwy definiowane w danym zakresie."""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.names.add(node.arg)
        self.generic_visit(node)

    def visit_alias(self, node):
        name = node.asname or node.name.split(".")[0]
        self.names.add(name)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)
        # nie schodzimy do wnętrza zagnieżdżonych funkcji

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.names.add(node.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.names.update(node.names)

    def visit_Nonlocal(self, node):
        self.names.update(node.names)


def collect_scope(node) -> set:
    collector = ScopeCollector()
    if isinstance(node, (ast.Module, ast.ClassDef)):
        for child in node.body:
            collector.visit(child)
    else:
        for child in node.body:
            collector.visit(child)
        for arg_group in (
            node.args.posonlyargs, node.args.args, node.args.kwonlyargs
        ):
            for argument in arg_group:
                collector.names.add(argument.arg)
        if node.args.vararg:
            collector.names.add(node.args.vararg.arg)
        if node.args.kwarg:
            collector.names.add(node.args.kwarg.arg)
    return collector.names


def used_names(node) -> set:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            base = child
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                names.add(base.id)
    return names


def check_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, path)
    problems = []

    module_scope = collect_scope(tree)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])

    all_used = used_names(tree)

    # Nazwy wymienione w __all__ są reeksportowane, więc nie są martwe.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__"
                        for t in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))):
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(
                    element.value, str
                ):
                    all_used.add(element.value)

    def walk_shallow(node):
        """Przechodzi drzewo, nie wchodząc do funkcji zagnieżdżonych."""
        stack = list(ast.iter_child_nodes(node))
        while stack:
            current = stack.pop()
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.Lambda)):
                continue
            yield current
            stack.extend(ast.iter_child_nodes(current))

    def walk_function(node, enclosing):
        """Sprawdza funkcję, uwzględniając zakresy otaczające."""
        local = collect_scope(node)
        visible = enclosing | local
        for inner in walk_shallow(node):
            if isinstance(inner, ast.Name) and isinstance(
                inner.ctx, ast.Load
            ):
                if (inner.id not in visible
                        and inner.id not in BUILTINS):
                    problems.append(
                        f"{path}:{inner.lineno}: niezdefiniowana "
                        f"nazwa '{inner.id}' w {node.name}"
                    )
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk_function(child, visible)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            walk_function(node, module_scope)
        elif isinstance(node, ast.ClassDef):
            class_scope = module_scope | collect_scope(node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef,
                                      ast.AsyncFunctionDef)):
                    walk_function(child, class_scope)

    for name in sorted(imported):
        if name not in all_used and name != "annotations":
            problems.append(f"{path}: nieużywany import '{name}'")

    return problems


def main() -> int:
    root = os.path.join(os.path.dirname(__file__), "..", "scene_rl")
    all_problems = []
    for filename in sorted(os.listdir(root)):
        if filename.endswith(".py"):
            all_problems.extend(check_file(os.path.join(root, filename)))

    if all_problems:
        for problem in all_problems:
            print(problem)
        return 1
    print("Analiza statyczna: brak zastrzeżeń")
    return 0


if __name__ == "__main__":
    sys.exit(main())
