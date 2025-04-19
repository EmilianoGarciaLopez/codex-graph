// src/utils/graph/retriever.ts
import type { FileContent, GraphData } from "./types";
import type { ResponseItem } from "openai/resources/responses/responses.mjs"; // Import ResponseItem

import { log } from "../agent/log";
import { approximateTokensUsed } from "../approximate-tokens-used";
import * as fs from "fs/promises";
import * as path from "path";

// Simple cache for file contents during retrieval
const retrievalCache = new Map<string, string>();

/**
 * Retrieves the content of files related to the target files based on the dependency graph.
 * @param targetFiles Absolute paths of the files being edited or focused on.
 * @param graph The pre-computed dependency graph.
 * @param maxTokens Maximum approximate tokens allowed for the related context.
 * @returns An array of FileContent objects for the related files.
 */
export async function getRelatedContext(
  targetFiles: Array<string>,
  graph: GraphData,
  maxTokens: number,
): Promise<Array<FileContent>> {
  const relatedFilePaths = new Set<string>();
  const candidates = new Set<string>();
  const targetSet = new Set(targetFiles.map((p) => path.resolve(p)));

  // 1. Gather direct dependencies and dependents
  for (const targetFile of targetSet) {
    const node = graph.nodes[targetFile];
    if (node) {
      node.dependencies.forEach((dep) => candidates.add(dep));
      node.dependents.forEach((dep) => candidates.add(dep));
    }
  }

  // 2. Iteratively add candidates while respecting token limit
  let currentTokens = 0;
  const relatedFilesContent: Array<FileContent> = [];

  // Prioritize direct neighbors
  const sortedCandidates = Array.from(candidates).sort(); // Sort for deterministic order

  for (const candidatePath of sortedCandidates) {
    // Don't include the target files themselves in the related context
    if (targetSet.has(candidatePath)) {
      continue;
    }

    try {
      let content = retrievalCache.get(candidatePath);
      if (content === undefined) {
        // eslint-disable-next-line no-await-in-loop
        content = await fs.readFile(candidatePath, "utf-8");
        retrievalCache.set(candidatePath, content); // Cache the content
      }

      // Construct a valid ResponseItem for token calculation
      const tempResponseItem: ResponseItem = {
        id: `temp-${candidatePath}`, // Add a temporary ID
        type: "message",
        role: "system", // Role doesn't matter for token count here
        content: [{ type: "input_text", text: content }],
      };

      const fileTokens = approximateTokensUsed([tempResponseItem]);

      if (currentTokens + fileTokens <= maxTokens) {
        relatedFilesContent.push({ path: candidatePath, content });
        currentTokens += fileTokens;
        relatedFilePaths.add(candidatePath);
      } else {
        // Stop adding files if we exceed the token limit
        log(
          `Token limit (${maxTokens}) reached while gathering related context. Added ${relatedFilesContent.length} files.`,
        );
        break;
      }
    } catch (error) {
      log(`Could not read related file ${candidatePath}: ${error}`);
      retrievalCache.delete(candidatePath); // Remove from cache if read failed
    }
  }

  // Clear cache after retrieval is done for this call (optional, depends on desired cache lifetime)
  // retrievalCache.clear();

  log(
    `Retrieved related context for ${targetFiles.join(
      ", ",
    )}: ${relatedFilesContent.length} files, ~${currentTokens} tokens.`,
  );
  return relatedFilesContent;
}