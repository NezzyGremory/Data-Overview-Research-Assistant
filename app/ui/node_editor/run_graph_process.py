"""Run a node graph in an isolated process and stream progress to the UI."""

from __future__ import annotations

import pickle
import struct
import sys
from pathlib import Path


def _send(output, record):
    # Serialize before writing so a serialization error cannot leave a partial
    # pickle in the progress stream and corrupt the following failure record.
    payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    output.write(struct.pack(">Q", len(payload)))
    output.write(payload)
    output.flush()


def main() -> int:
    protocol_output = sys.stdout.buffer
    # Imported analysis modules may print diagnostics; reserve stdout for the
    # pickle protocol consumed by the parent process.
    sys.stdout = sys.stderr
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    graph = None
    try:
        payload = pickle.load(sys.stdin.buffer)
        from app.ml.graph import GraphEngine

        graph = GraphEngine()
        graph_data = payload["graph"]
        for raw in graph_data.get("nodes", []):
            graph.add_node(
                raw["spec_id"],
                raw["id"],
                raw.get("parameters"),
                tuple(raw.get("position", (0, 0))),
            )
        for edge in graph_data.get("connections", []):
            graph.connect(*edge)

        def report_progress(node_id: str, status: str) -> None:
            _send(protocol_output, ("progress", node_id, status))

        output = graph.run(payload["dataframe"], report_progress)
        # The UI only renders tables and metrics. Keep fitted estimators and
        # other internal values inside the worker process.
        display_output = {
            node_id: {
                key: value
                for key, value in node_output.items()
                if key in {"table", "metrics"}
            }
            for node_id, node_output in output.items()
            if isinstance(node_output, dict)
            and any(key in node_output for key in ("table", "metrics"))
        }
        _send(protocol_output, ("finished", display_output))
        return 0
    except BaseException as exc:
        if graph is not None:
            for node_id, node in graph.nodes.items():
                if node.status == "error":
                    try:
                        _send(protocol_output, ("progress", node_id, "error"))
                    except BaseException:
                        break
                elif node.status in {"menunggu", "berjalan"}:
                    try:
                        _send(protocol_output, ("progress", node_id, "dilewati"))
                    except BaseException:
                        break
        try:
            _send(protocol_output, ("failed", str(exc) or exc.__class__.__name__))
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
