"""
SIH26054 Replan to Learn: Merkle lineage DAG (M3, F12).
Every ProvenanceManifest already carries extensions.parent_manifest_hashes
(see provenance/manifest.py ProvenanceAuditMetadata) -- this module turns
that into a traversable graph: nodes are manifest content hashes, edges run
parent -> child.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx

from replan_to_learn.provenance.hasher import compute_manifest_hash
from replan_to_learn.provenance.manifest import ProvenanceManifest


class LineageDAG:
    """
    Directed acyclic graph of artifact provenance: an edge parent_hash ->
    child_hash means the child artifact was derived from the parent.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._manifests: Dict[str, ProvenanceManifest] = {}

    def add_manifest(self, manifest: ProvenanceManifest) -> str:
        node_hash = compute_manifest_hash(manifest)
        self._manifests[node_hash] = manifest
        self.graph.add_node(
            node_hash,
            dataset_id=manifest.dataset_id,
            flight_id=manifest.flight_id,
            model_version=manifest.model_version,
        )
        for parent_hash in manifest.extensions.parent_manifest_hashes:
            self.graph.add_edge(parent_hash, node_hash)
        return node_hash

    @classmethod
    def build_from_registry(cls, registry) -> "LineageDAG":
        """registry: replan_to_learn.registry.ArtifactRegistry"""
        dag = cls()
        for manifest in registry.list_all_manifests():
            dag.add_manifest(manifest)
        return dag

    def manifest_for(self, node_hash: str) -> Optional[ProvenanceManifest]:
        return self._manifests.get(node_hash)

    def ancestors(self, node_hash: str) -> set:
        if node_hash not in self.graph:
            return set()
        return nx.ancestors(self.graph, node_hash)

    def descendants(self, node_hash: str) -> set:
        if node_hash not in self.graph:
            return set()
        return nx.descendants(self.graph, node_hash)

    def topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "hash": h,
                    "dataset_id": data.get("dataset_id"),
                    "flight_id": data.get("flight_id"),
                    "model_version": data.get("model_version"),
                }
                for h, data in self.graph.nodes(data=True)
            ],
            "edges": [{"parent": u, "child": v} for u, v in self.graph.edges()],
        }

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def from_index(cls, path: Union[str, Path]) -> "LineageDAG":
        """
        Rebuild graph structure (hashes + edges) from a persisted index.
        Node metadata is restored, but full ProvenanceManifest objects are
        not (the index only stores a summary) -- use build_from_registry
        for full manifest access.
        """
        dag = cls()
        data = json.loads(Path(path).read_text())
        for node in data.get("nodes", []):
            dag.graph.add_node(
                node["hash"],
                dataset_id=node.get("dataset_id"),
                flight_id=node.get("flight_id"),
                model_version=node.get("model_version"),
            )
        for edge in data.get("edges", []):
            dag.graph.add_edge(edge["parent"], edge["child"])
        return dag
