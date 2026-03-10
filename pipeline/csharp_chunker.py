"""C#-specific Tree-sitter chunker — extracts classes, interfaces, methods."""

from dataclasses import dataclass, field

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

CS_LANGUAGE = Language(tscsharp.language())


@dataclass
class CodeChunk:
    text: str
    chunk_index: int
    namespace: str
    class_name: str
    method_name: str
    is_interface: bool
    implements_interfaces: list[str] = field(default_factory=list)


def _get_parser() -> Parser:
    return Parser(CS_LANGUAGE)


def _node_text(node, source: bytes) -> str:
    """Extract text for a node from source bytes."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _find_children_by_type(node, type_name: str) -> list:
    """Find all direct children of a given type."""
    return [c for c in node.children if c.type == type_name]


def _find_descendants_by_type(node, type_name: str) -> list:
    """Recursively find all descendants of a given type."""
    results = []
    for child in node.children:
        if child.type == type_name:
            results.append(child)
        results.extend(_find_descendants_by_type(child, type_name))
    return results


def _get_namespace(node, source: bytes) -> str:
    """Walk up or find namespace declaration."""
    for ns in _find_descendants_by_type(node, "namespace_declaration"):
        name_node = ns.child_by_field_name("name")
        if name_node:
            return _node_text(name_node, source)
    # File-scoped namespace
    for ns in _find_descendants_by_type(node, "file_scoped_namespace_declaration"):
        name_node = ns.child_by_field_name("name")
        if name_node:
            return _node_text(name_node, source)
    return ""


def _get_base_types(class_node, source: bytes) -> list[str]:
    """Extract base types (interfaces/base classes) from a class declaration."""
    bases = []
    for child in class_node.children:
        if child.type == "base_list":
            for base in child.children:
                if base.type in ("identifier", "generic_name", "qualified_name"):
                    bases.append(_node_text(base, source))
                elif base.type == "simple_base_type":
                    bases.append(_node_text(base, source))
    return bases


def _get_class_name(class_node, source: bytes) -> str:
    """Extract class/interface name."""
    name_node = class_node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node, source)
    return ""


def _get_modifiers(node, source: bytes) -> str:
    """Extract modifier keywords (public, static, async, etc.)."""
    mods = []
    for child in node.children:
        if child.type == "modifier":
            mods.append(_node_text(child, source))
    return " ".join(mods)


def _get_method_signature(method_node, source: bytes) -> str:
    """Build a method signature string."""
    mods = _get_modifiers(method_node, source)
    return_type = method_node.child_by_field_name("type")
    name = method_node.child_by_field_name("name")
    params = None
    for child in method_node.children:
        if child.type == "parameter_list":
            params = child
            break

    parts = []
    if mods:
        parts.append(mods)
    if return_type:
        parts.append(_node_text(return_type, source))
    if name:
        parts.append(_node_text(name, source))
    sig = " ".join(parts)
    if params:
        sig += _node_text(params, source)
    return sig


def _build_context_header(namespace: str, class_name: str, base_types: list[str],
                          is_interface: bool, method_sig: str = "") -> str:
    """Build the context header prepended to each chunk."""
    lines = []
    if namespace:
        lines.append(f"// Namespace: {namespace}")

    type_label = "Interface" if is_interface else "Class"
    if base_types:
        lines.append(f"// {type_label}: {class_name} : {', '.join(base_types)}")
    else:
        lines.append(f"// {type_label}: {class_name}")

    if method_sig:
        lines.append(f"// Method: {method_sig}")

    return "\n".join(lines)


def _find_constructor(class_node, source: bytes) -> str:
    """Find constructor text in a class, if present."""
    body = class_node.child_by_field_name("body")
    if not body:
        return ""
    for child in body.children:
        if child.type == "constructor_declaration":
            return _node_text(child, source)
    return ""


def _get_methods(class_node):
    """Get method and property declarations from a class body."""
    body = class_node.child_by_field_name("body")
    if not body:
        return []
    method_types = ("method_declaration", "property_declaration", "constructor_declaration")
    return [c for c in body.children if c.type in method_types]


def chunk_csharp(source_code: str, chunk_size: int = 512) -> list[CodeChunk]:
    """Parse C# source and chunk at class/method boundaries.

    Small classes (<chunk_size tokens) → one chunk per class.
    Large classes → one chunk per method with context header + constructor.
    Interfaces → always one chunk.
    """
    parser = _get_parser()
    source = source_code.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    namespace = _get_namespace(root, source)
    chunks: list[CodeChunk] = []
    index = 0

    # Find all class-like declarations
    class_types = ("class_declaration", "interface_declaration", "record_declaration",
                   "struct_declaration", "enum_declaration")

    class_nodes = []
    for ct in class_types:
        class_nodes.extend(_find_descendants_by_type(root, ct))

    if not class_nodes:
        # No classes found — treat whole file as one chunk
        text = source_code.strip()
        if text:
            chunks.append(CodeChunk(
                text=text, chunk_index=0, namespace=namespace,
                class_name="", method_name="", is_interface=False,
            ))
        return chunks

    for class_node in class_nodes:
        class_name = _get_class_name(class_node, source)
        is_interface = class_node.type == "interface_declaration"
        base_types = _get_base_types(class_node, source)
        interfaces = [b for b in base_types if b.startswith("I") and len(b) > 1]
        class_text = _node_text(class_node, source)

        # Interfaces and small classes → single chunk
        if is_interface or _estimate_tokens(class_text) <= chunk_size:
            header = _build_context_header(namespace, class_name, base_types, is_interface)
            chunks.append(CodeChunk(
                text=f"{header}\n{class_text}",
                chunk_index=index,
                namespace=namespace,
                class_name=class_name,
                method_name="",
                is_interface=is_interface,
                implements_interfaces=interfaces,
            ))
            index += 1
            continue

        # Large class → chunk per method
        constructor_text = _find_constructor(class_node, source)
        methods = _get_methods(class_node)

        if not methods:
            # No methods — emit whole class anyway
            header = _build_context_header(namespace, class_name, base_types, is_interface)
            chunks.append(CodeChunk(
                text=f"{header}\n{class_text}",
                chunk_index=index,
                namespace=namespace,
                class_name=class_name,
                method_name="",
                is_interface=is_interface,
                implements_interfaces=interfaces,
            ))
            index += 1
            continue

        for method in methods:
            method_text = _node_text(method, source)
            method_name = ""
            method_sig = ""

            if method.type == "method_declaration":
                name_node = method.child_by_field_name("name")
                method_name = _node_text(name_node, source) if name_node else ""
                method_sig = _get_method_signature(method, source)
            elif method.type == "constructor_declaration":
                method_name = "constructor"
                method_sig = class_name + "()"
            elif method.type == "property_declaration":
                name_node = method.child_by_field_name("name")
                method_name = _node_text(name_node, source) if name_node else ""
                method_sig = f"property {method_name}"

            header = _build_context_header(
                namespace, class_name, base_types, is_interface, method_sig
            )

            # Prepend constructor for non-constructor methods
            parts = [header]
            if constructor_text and method.type != "constructor_declaration":
                parts.append(constructor_text)
            parts.append(method_text)

            chunks.append(CodeChunk(
                text="\n".join(parts),
                chunk_index=index,
                namespace=namespace,
                class_name=class_name,
                method_name=method_name,
                is_interface=is_interface,
                implements_interfaces=interfaces,
            ))
            index += 1

    return chunks
