"""Measure Ren's model-bound workflow result compaction."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import mcp_server  # noqa: E402
from chat_runtime import (  # noqa: E402
    MODEL_COMPILER_RESULT_MAX_CHARS,
    MODEL_TOOL_RESULT_MAX_CHARS,
    ActiveRun,
    prepare_embedded_tool_result,
)
from workflow_graph_patch import (  # noqa: E402
    ApplyGraphPatchRequest,
    GraphPatchPlan,
    graph_patch_hash,
)


def serialized_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def workflow_result(node_count: int) -> dict[str, Any]:
    nodes = [
        {
            "id": index,
            "type": "KSampler" if index % 4 == 0 else "CLIPTextEncode",
            "title": f"Representative workflow node {index}",
            "widgets_values": ["x" * 256, index, 7.5],
            "pos": [index * 20, index * 10],
        }
        for index in range(node_count)
    ]
    return {
        "workflow_identity": "benchmark-workflow",
        "graph_hash": "a" * 64,
        "workflow": {
            "nodes": nodes,
            "links": [
                [index, index, 0, index + 1, 0, "CONDITIONING"]
                for index in range(max(0, node_count - 1))
            ],
        },
    }


def compiler_result(operation_count: int) -> dict[str, Any]:
    create_nodes = [
        {
            "ref": f"node_{index}",
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "benchmark prompt " + "x" * 256},
        }
        for index in range(operation_count)
    ]
    apply_request = {
        "workflow_id": "benchmark-workflow",
        "plan": {
            "create_nodes": create_nodes,
            "update_nodes": [],
            "remove_nodes": [],
            "add_edges": [],
            "remove_edges": [],
            "attachments": [],
        },
    }
    return {
        "valid": True,
        "schema": "graph_patch.v1",
        "patch_hash": "b" * 64,
        "apply_request": apply_request,
        "expected_final": {"nodes": create_nodes, "edges": []},
        "issues": [],
        "error_count": 0,
    }


def native_compiler_result(operation_count: int) -> dict[str, Any]:
    catalog_hash = "c" * 64
    create_nodes = [
        {
            "alias": f"node_{index}",
            "node_type": "CLIPTextEncode",
            "schema_hash": "d" * 64,
            "values": {"text": "benchmark prompt " + "x" * 2_000},
        }
        for index in range(operation_count)
    ]
    plan = GraphPatchPlan.model_validate({
        "expected_workflow_identity": "benchmark-workflow",
        "expected_graph_hash": "a" * 64,
        "assertions": {"nodes": [], "edges": []},
        "create_nodes": create_nodes,
        "expected_delta": {
            "created_node_count": operation_count,
            "updated_node_count": 0,
            "removed_node_count": 0,
            "added_edge_count": 0,
            "removed_edge_count": 0,
            "final_node_count": operation_count,
            "final_edge_count": 0,
        },
    })
    request = ApplyGraphPatchRequest(
        application_id="benchmark-native-compiler",
        expected_catalog_hash=catalog_hash,
        patch_hash=graph_patch_hash(plan, catalog_hash),
        plan=plan,
    )
    return {
        "valid": True,
        "compiler_schema": "fl-mcp.workflow-refinement-compiler.v1",
        "needs_choice": False,
        "patch_hash": request.patch_hash,
        "catalog": {"hash": catalog_hash},
        "plan": plan.model_dump(mode="json"),
        "apply_request": request.model_dump(mode="json"),
        "issues": [],
        "error_count": 0,
        "warning_count": 0,
    }


def measure(
    name: str,
    source: dict[str, Any],
    tool_name: str,
    limit: int,
    iterations: int,
) -> dict[str, Any]:
    durations = []
    prepared = None
    for _ in range(iterations):
        state = ActiveRun("benchmark", "benchmark", "benchmark")
        started = time.perf_counter()
        prepared = prepare_embedded_tool_result(state, tool_name, source)
        durations.append((time.perf_counter() - started) * 1_000)
    before = serialized_chars(source)
    after = serialized_chars(prepared)
    if after > limit:
        raise RuntimeError(f"{name} exceeded its {limit}-character model-result limit")
    return {
        "case": name,
        "input_chars": before,
        "model_chars": after,
        "reduction_percent": round((1 - after / before) * 100, 2),
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(sorted(durations)[max(0, int(iterations * 0.95) - 1)], 3),
    }


def measure_native_compiler(iterations: int) -> dict[str, Any]:
    source = native_compiler_result(100)
    durations = []
    prepared = None
    prior = os.environ.get("FL_MCP_NATIVE_TURN_CONTROLS")
    os.environ["FL_MCP_NATIVE_TURN_CONTROLS"] = "1"
    try:
        for _ in range(iterations):
            context = SimpleNamespace(request_context=SimpleNamespace(
                lifespan_context={"ren_turn_state": {
                    "apply_handles": {},
                    "compile_attempts": 1,
                    "apply_attempts": 0,
                    "read_calls": {},
                    "completed_reads": set(),
                }},
            ))
            started = time.perf_counter()
            result = mcp_server._native_compiler_result(context, source)
            durations.append((time.perf_counter() - started) * 1_000)
            prepared = json.loads(result.content[0].text)
    finally:
        if prior is None:
            os.environ.pop("FL_MCP_NATIVE_TURN_CONTROLS", None)
        else:
            os.environ["FL_MCP_NATIVE_TURN_CONTROLS"] = prior
    before = serialized_chars(source)
    after = serialized_chars(prepared)
    if after > mcp_server.NATIVE_COMPILER_RESULT_MAX_CHARS:
        raise RuntimeError("native compiler result exceeded its model-result limit")
    return {
        "case": "native_compiler_100_operations",
        "input_chars": before,
        "model_chars": after,
        "reduction_percent": round((1 - after / before) * 100, 2),
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(sorted(durations)[max(0, int(iterations * 0.95) - 1)], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    cases: list[tuple[str, Callable[[], dict[str, Any]], str, int]] = [
        (
            f"workflow_{node_count}_nodes",
            lambda node_count=node_count: workflow_result(node_count),
            "workflow_get_current_json",
            MODEL_TOOL_RESULT_MAX_CHARS,
        )
        for node_count in (100, 500, 2_000)
    ]
    cases.append(
        (
            "compiler_500_operations",
            lambda: compiler_result(500),
            "compile_workflow_refinement_spec",
            MODEL_COMPILER_RESULT_MAX_CHARS,
        )
    )
    results = [
        measure(name, factory(), tool_name, limit, args.iterations)
        for name, factory, tool_name, limit in cases
    ]
    results.append(measure_native_compiler(args.iterations))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
