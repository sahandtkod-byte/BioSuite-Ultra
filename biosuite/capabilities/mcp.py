"""MCP catalog projection, without protocol handling or tool execution.

Scalar core results use named object fields for the future MCP adapter;
sequence statistics retain the core function's object shape.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

__all__ = ["mcp_tools", "MCP_INPUT_SCHEMAS", "MCP_OUTPUT_SCHEMAS"]

MCP_INPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "sequence.gc_content": {
        "type": "object",
        "properties": {
            "seq": {"type": "string", "description": "DNA sequence"},
        },
        "required": ["seq"],
        "additionalProperties": False,
    },
    "sequence.reverse_complement": {
        "type": "object",
        "properties": {
            "seq": {"type": "string", "description": "DNA sequence"},
        },
        "required": ["seq"],
        "additionalProperties": False,
    },
    "sequence.translate": {
        "type": "object",
        "properties": {
            "seq": {"type": "string", "description": "coding DNA sequence"},
            "frame": {"type": "integer", "enum": [1, 2, 3, -1, -2, -3], "default": 1},
            "table": {"type": "integer", "enum": [1], "default": 1},
        },
        "required": ["seq"],
        "additionalProperties": False,
    },
    "sequence.sequence_stats": {
        "type": "object",
        "properties": {
            "seq": {"type": "string", "description": "DNA sequence"},
        },
        "required": ["seq"],
        "additionalProperties": False,
    },
}

MCP_OUTPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "sequence.gc_content": {
        "type": "object",
        "properties": {"gc_percent": {"type": "number"}},
        "required": ["gc_percent"],
    },
    "sequence.reverse_complement": {
        "type": "object",
        "properties": {"reverse_complement": {"type": "string"}},
        "required": ["reverse_complement"],
    },
    "sequence.translate": {
        "type": "object",
        "properties": {"protein": {"type": "string"}},
        "required": ["protein"],
    },
    "sequence.sequence_stats": {
        "type": "object",
        "properties": {
            "length": {"type": "integer"},
            "A": {"type": "integer"},
            "T": {"type": "integer"},
            "G": {"type": "integer"},
            "C": {"type": "integer"},
            "N": {"type": "integer"},
            "AT": {"type": "number"},
            "GC": {"type": "number"},
            "other": {"type": "number"},
        },
        "required": ["length", "A", "T", "G", "C", "N", "AT", "GC", "other"],
    },
}


def mcp_tools(capabilities=None) -> List[Dict[str, Any]]:
    """Project exposed entries from the full catalog or a supplied iterable."""
    if capabilities is None:
        from .catalog import list_capabilities

        capabilities = list_capabilities()
    tools: List[Dict[str, Any]] = []
    for cap in capabilities:
        if not cap.mcp_exposed:
            continue
        tools.append(
            {
                "name": cap.id,
                "description": cap.summary,
                "inputSchema": deepcopy(MCP_INPUT_SCHEMAS[cap.id]),
                "outputSchema": deepcopy(MCP_OUTPUT_SCHEMAS[cap.id]),
            }
        )
    return tools
