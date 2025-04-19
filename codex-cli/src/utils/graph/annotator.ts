// src/utils/graph/annotator.ts
import type { GraphData } from "./types";

import path from "path";

/**
 * Generates a Markdown representation of the dependency graph structure.
 * Uses relative paths for better readability in the prompt.
 * @param graph The graph data object.
 * @returns A Markdown string describing the graph.
 */
export function generateGraphMarkdown(graph: GraphData): string {
  const projectRoot = graph.metadata.projectRoot ?? process.cwd();
  const lines: Array<string> = ["## Project Dependency Graph\n"];

  const sortedNodeIds = Object.keys(graph.nodes).sort();

  for (const nodeId of sortedNodeIds) {
    const node = graph.nodes[nodeId];
    if (!node) {
      continue; // Add curly braces
    }

    const relativePath = path.relative(projectRoot, node.id);
    lines.push(`- **${relativePath}**`);

    if (node.dependencies.length > 0) {
      const deps = node.dependencies
        .map((depId) => `\`${path.relative(projectRoot, depId)}\``)
        .sort()
        .join(", ");
      lines.push(`  - Depends on: ${deps}`);
    } else {
      lines.push(`  - Depends on: (none)`);
    }

    if (node.dependents.length > 0) {
      const deps = node.dependents
        .map((depId) => `\`${path.relative(projectRoot, depId)}\``)
        .sort()
        .join(", ");
      lines.push(`  - Depended on by: ${deps}`);
    } else {
      lines.push(`  - Depended on by: (none)`);
    }
  }

  if (sortedNodeIds.length === 0) {
    lines.push("(No nodes in graph)");
  }

  return lines.join("\n");
}