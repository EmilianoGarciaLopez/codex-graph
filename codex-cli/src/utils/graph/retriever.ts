// src/utils/graph/retriever.ts
import type { GraphData, GraphNode, RelatedContextResult } from "./types";
import { log } from "../agent/log";
import path from "path";

/**
 * Retrieves related files and their relationship summaries for a given file path.
 *
 * @param targetFilePaths Absolute paths of the target files to find context for.
 * @param graphData The loaded dependency graph data.
 * @param _maxTokens Optional token budget (currently unused, but kept for potential future use).
 * @returns A promise resolving to an array of RelatedContextResult objects.
 */
export async function getRelatedContext(
  targetFilePaths: Array<string>,
  graphData: GraphData | null,
  _maxTokens?: number, // Keep param for signature consistency, but ignore for now
): Promise<Array<RelatedContextResult>> {
  if (!graphData) {
    log("Graph data is not available. Cannot retrieve related context.");
    return [];
  }

  const results: Array<RelatedContextResult> = [];
  const addedRelatedFiles = new Set<string>(); // Track added related files to avoid duplicates

  for (const targetFilePath of targetFilePaths) {
    const absoluteTargetPath = path.resolve(graphData.metadata.projectRoot ?? process.cwd(), targetFilePath);
    const node = graphData.nodes[absoluteTargetPath];

    if (!node) {
      log(`Node not found in graph for target file: ${absoluteTargetPath}`);
      continue;
    }

    // Add dependencies
    for (const depInfo of node.dependencies) {
      if (!addedRelatedFiles.has(depInfo.id)) {
        results.push({
          relatedFilename: depInfo.id,
          relationshipSummary: depInfo.summary || "Dependency relationship (no summary available).",
        });
        addedRelatedFiles.add(depInfo.id);
      }
    }

    // Add dependents
    for (const depInfo of node.dependents) {
       if (!addedRelatedFiles.has(depInfo.id)) {
         results.push({
           relatedFilename: depInfo.id,
           // The summary stored is for the source->target relationship.
           // We can reuse it or potentially generate a reverse summary if needed.
           // For now, reuse the stored summary.
           relationshipSummary: depInfo.summary || "Dependent relationship (no summary available).",
         });
         addedRelatedFiles.add(depInfo.id);
       }
    }
  }

   // Sort results alphabetically by filename for consistent output
   results.sort((a, b) => a.relatedFilename.localeCompare(b.relatedFilename));

  log(`Retrieved ${results.length} related context entries for target(s): ${targetFilePaths.join(', ')}`);
  return results;
}