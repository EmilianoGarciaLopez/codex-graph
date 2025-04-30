import type { GraphNode, GraphEdge } from "./types";
import type { AppConfig } from "../utils/config";

import { loadIgnorePatterns } from "../utils/singlepass/context_files"; // Reusing ignore pattern logic
import { log } from "../utils/logger/log";
import { parseFileWithLLM } from "./llm-parser";
import {
  getGraphDir,
  loadNodes,
  loadEdges,
  loadHashes,
  saveNodes,
  saveEdges,
  saveHashes,
} from "./storage";
import crypto from "crypto";
import fs from "fs/promises";
import path from "path";

// TODO: Make configurable
const RELEVANT_EXTENSIONS = new Set([
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".py",
  ".rb",
  ".java",
  ".cs",
  ".go",
  ".php",
  ".swift",
  ".kt",
  ".rs",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
  ".m",
  ".scala",
  ".pl",
  ".pm",
  // Add more as needed
]);

// Concurrency limit for LLM parsing
const CONCURRENT_LIMIT = 5; // Adjust as needed

/**
 * Builds or updates the codebase graph by scanning files, parsing them with an LLM,
 * and saving the results.
 *
 * @param projectRoot - The absolute path to the root of the project.
 * @param config - The application configuration.
 */
export async function buildOrUpdateGraph(
  projectRoot: string,
  config: AppConfig,
): Promise<void> {
  log("Starting graph build/update process...");
  const graphDir = await getGraphDir(projectRoot);

  // Load existing data
  log("Loading existing graph data...");
  const existingNodesMap = new Map(
    (await loadNodes(graphDir)).map((n) => [n.id, n]),
  );
  const existingEdges = await loadEdges(graphDir); // Edges are harder to map efficiently, load as array
  const existingHashes = await loadHashes(graphDir);
  log(
    `Loaded ${existingNodesMap.size} nodes, ${existingEdges.length} edges, ${
      Object.keys(existingHashes).length
    } hashes.`,
  );

  // Scan for relevant source files
  log("Scanning for source files...");
  const ignorePatterns = loadIgnorePatterns(
    path.join(projectRoot, ".codexignore"),
  ); // Allow custom ignores
  const sourceFiles = await findSourceFiles(projectRoot, ignorePatterns);
  log(`Found ${sourceFiles.length} potential source files.`);

  const updatedNodesMap = new Map<string, GraphNode>(existingNodesMap);
  const updatedEdgesSet = new Set<string>(
    existingEdges.map((e) => `${e.src}->${e.dst}`),
  ); // Use a set for quick edge existence check
  let updatedEdges: GraphEdge[] = [...existingEdges]; // Start with existing edges
  const updatedHashes: Record<string, string> = { ...existingHashes };
  const nodesByFile = new Map<string, Set<string>>(); // Map file path -> Set<nodeId>

  // Pre-populate nodesByFile from existing nodes
  for (const node of existingNodesMap.values()) {
    const relativePath = node.ptr[0];
    if (!nodesByFile.has(relativePath)) {
      nodesByFile.set(relativePath, new Set());
    }
    nodesByFile.get(relativePath)!.add(node.id);
  }

  let filesProcessed = 0;
  let filesSkipped = 0;
  let filesFailed = 0;

  // --- Stage 1: Identify files needing processing ---
  const filesToProcess: {
    filePath: string;
    relativePath: string;
    fileContent: string;
  }[] = [];
  for (const filePath of sourceFiles) {
    const relativePath = path.relative(projectRoot, filePath);
    try {
      // eslint-disable-next-line no-await-in-loop
      const fileContent = await fs.readFile(filePath, "utf-8");

      // Skip empty or very short files
      if (fileContent.length < 2) {
        log(`Skipping empty file: ${relativePath}`);
        filesSkipped++;
        continue;
      }

      const currentFileHash = crypto
        .createHash("sha256")
        .update(fileContent)
        .digest("hex");

      let needsReparsing = false;
      const nodeIdsInFile = nodesByFile.get(relativePath);

      if (!nodeIdsInFile) {
        needsReparsing = true; // New file
      } else {
        let fileHashChanged = false;
        for (const nodeId of nodeIdsInFile) {
          const storedHash = existingHashes[nodeId]; // Check the hash stored from the *last* successful parse
          if (storedHash !== currentFileHash) {
            fileHashChanged = true;
            log(
              `File ${relativePath} needs reparsing because hash for node ${nodeId} changed (stored: ${storedHash?.substring(
                0,
                7,
              )}, current: ${currentFileHash.substring(0, 7)}).`,
            );
            break;
          }
        }
        needsReparsing = fileHashChanged;
      }

      if (!needsReparsing) {
        log(`Skipping unchanged file: ${relativePath}`);
        filesSkipped++;
        continue;
      }

      filesToProcess.push({ filePath, relativePath, fileContent });
    } catch (error) {
      filesFailed++; // Count failure during initial read/hash check
      log(`Failed initial check for file ${relativePath}: ${error}`);
    }
  }

  // --- Stage 2: Process identified files with concurrency limit ---
  log(
    `Processing ${filesToProcess.length} files with concurrency limit ${CONCURRENT_LIMIT}...`,
  );
  const results: Array<any> = []; // Store results from processed files
  const activePromises: Map<
    Promise<any>,
    { filePath: string; relativePath: string }
  > = new Map();
  let fileIndex = 0;

  while (fileIndex < filesToProcess.length || activePromises.size > 0) {
    // Start new tasks if below the concurrency limit and there are files left
    while (
      activePromises.size < CONCURRENT_LIMIT &&
      fileIndex < filesToProcess.length
    ) {
      const { filePath, relativePath, fileContent } =
        filesToProcess[fileIndex]!;
      fileIndex++;

      const promise = (async () => {
        try {
          log(`Processing file: ${relativePath}`);
          const nodesToRemove = nodesByFile.get(relativePath) || new Set();
          const edgesToRemoveFromFile = updatedEdges.filter(
            (edge) => nodesToRemove.has(edge.src) || nodesToRemove.has(edge.dst),
          );

          const { nodes: parsedNodes, edges: parsedEdges } =
            // eslint-disable-next-line no-await-in-loop
            await parseFileWithLLM(
              projectRoot,
              filePath,
              fileContent,
              config,
            );

          const currentFileHash = crypto
            .createHash("sha256")
            .update(fileContent)
            .digest("hex");

          return {
            relativePath,
            parsedNodes,
            parsedEdges,
            nodesToRemove,
            edgesToRemoveFromFile,
            currentFileHash,
            success: true,
          };
        } catch (error) {
          log(
            `Failed to process file ${relativePath} during LLM parsing: ${error}`,
          );
          return { relativePath, error, success: false };
        }
      })();

      activePromises.set(promise, { filePath, relativePath });

      // Handle completion or error of this specific promise
      promise
        .then((result) => {
          results.push(result);
          activePromises.delete(promise); // Remove from active pool
        })
        .catch((error) => {
          // This catch is mainly for unexpected errors not handled inside the async IIFE
          log(
            `Unexpected error processing promise for ${relativePath}: ${error}`,
          );
          results.push({ relativePath, error, success: false });
          activePromises.delete(promise); // Ensure removal even on unexpected error
        });
    }

    // If the concurrency limit is reached or all files are dispatched, wait for any promise to finish
    if (activePromises.size >= CONCURRENT_LIMIT || fileIndex === filesToProcess.length) {
      if (activePromises.size > 0) {
        // eslint-disable-next-line no-await-in-loop
        await Promise.race(activePromises.keys());
      }
    }
  }


  // --- Stage 3: Integrate results ---
  // This stage needs to be sequential to avoid race conditions when modifying shared maps/arrays
  log("Integrating results...");
  for (const result of results) {
    if (result.success) {
      const {
        relativePath,
        parsedNodes,
        parsedEdges,
        nodesToRemove,
        edgesToRemoveFromFile,
        currentFileHash,
      } = result;
      filesProcessed++;

      // 1. Remove old nodes and their hashes associated with the file
      nodesToRemove.forEach((nodeId: string) => {
        updatedNodesMap.delete(nodeId);
        delete updatedHashes[nodeId]; // Remove old hash entry
      });
      nodesByFile.set(relativePath, new Set()); // Reset nodes for this file

      // 2. Remove old edges associated with the file
      const edgesToRemoveKeys = new Set(
        edgesToRemoveFromFile.map((e: GraphEdge) => `${e.src}->${e.dst}`),
      );
      updatedEdges = updatedEdges.filter(
        (edge) => !edgesToRemoveKeys.has(`${edge.src}->${edge.dst}`),
      );
      edgesToRemoveKeys.forEach((key) => updatedEdgesSet.delete(key)); // Keep set consistent

      // 3. Add new nodes and update hashes
      for (const node of parsedNodes) {
        // Use the pre-calculated file hash for consistency check in the *next* run
        node.hash = currentFileHash; // Store the hash of the entire file content
        updatedHashes[node.id] = currentFileHash; // Update hash map for this node

        updatedNodesMap.set(node.id, node);
        // Update nodesByFile map
        if (!nodesByFile.has(relativePath)) {
          nodesByFile.set(relativePath, new Set());
        }
        nodesByFile.get(relativePath)!.add(node.id);
      }

      // 4. Add new edges
      for (const edge of parsedEdges) {
        const edgeKey = `${edge.src}->${edge.dst}`;
        if (!updatedEdgesSet.has(edgeKey)) {
          updatedEdges.push(edge);
          updatedEdgesSet.add(edgeKey);
        }
      }
    } else {
      filesFailed++; // Count failures during parallel processing
      // Error already logged within the promise catch block
    }
  }

  // --- Stage 4: Create stub nodes (unchanged from original logic) ---
  log("Creating stub nodes for missing destinations...");
  const allNodeIds = new Set(updatedNodesMap.keys());
  for (const edge of updatedEdges) {
    if (!allNodeIds.has(edge.dst)) {
      // Create a stub node if the destination doesn't exist
      const stubNode: GraphNode = {
        id: edge.dst,
        ptr: ["external", 0, 0], // Mark as external/stub
        sig: edge.dst, // Use ID as signature for stub
        doc: "External dependency or unresolved reference",
        summary: "External dependency or unresolved reference",
        tok: 0, // No body content
        tags: ["external"], // Tag as external
        hash: undefined, // No hash for stubs
      };
      updatedNodesMap.set(stubNode.id, stubNode);
      allNodeIds.add(stubNode.id); // Add to the set to avoid re-creating
      log(`Created stub node for missing destination: ${edge.dst}`);
    }
  }

  // Save updated data
  log("Saving updated graph data...");
  const finalNodes = Array.from(updatedNodesMap.values());
  await saveNodes(graphDir, finalNodes);
  await saveEdges(graphDir, updatedEdges);
  // Save the separate hashes map (only include hashes for nodes that still exist)
  const finalHashes = finalNodes.reduce(
    (acc, node) => {
      if (node.hash) {
        // Only save if hash was successfully calculated
        acc[node.id] = node.hash;
      }
      return acc;
    },
    {} as Record<string, string>,
  );
  await saveHashes(graphDir, finalHashes);

  log(
    `Graph build/update complete. Processed: ${filesProcessed}, Skipped: ${filesSkipped}, Failed: ${filesFailed}. Total nodes: ${finalNodes.length}, Total edges: ${updatedEdges.length}`,
  );
}

/**
 * Recursively finds all source files in a directory, respecting ignore patterns.
 *
 * @param dir - The directory to scan.
 * @param ignorePatterns - An array of RegExp patterns to ignore.
 * @returns A promise resolving to an array of absolute file paths.
 */
async function findSourceFiles(
  dir: string,
  ignorePatterns: RegExp[],
): Promise<string[]> {
  let files: string[] = [];
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.resolve(dir, entry.name);
      const relativePath = path.relative(process.cwd(), fullPath); // Use relative path for logging ignores

      // Check ignore patterns first
      if (ignorePatterns.some((pattern) => pattern.test(fullPath))) {
        log(`Ignoring path due to pattern: ${relativePath}`);
        continue;
      }
      // Skip common hidden / config / build directories explicitly
      const baseName = path.basename(fullPath);
      if (
        baseName.startsWith(".") ||
        [
          "node_modules",
          "dist",
          "build",
          "target",
          "venv",
          ".venv",
          "env",
          ".git",
          ".hg",
          ".svn",
        ].includes(baseName)
      ) {
        log(`Ignoring common directory: ${relativePath}`);
        continue;
      }

      if (entry.isDirectory()) {
        // Check if directory itself is ignored before recursing
        if (!ignorePatterns.some((pattern) => pattern.test(fullPath + "/"))) {
          // Check directory pattern
          // eslint-disable-next-line no-await-in-loop
          files = files.concat(
            await findSourceFiles(fullPath, ignorePatterns), // Recurse into subdirectories
          );
        } else {
          log(`Ignoring directory due to pattern: ${relativePath}/`);
        }
      } else if (
        entry.isFile() &&
        RELEVANT_EXTENSIONS.has(path.extname(entry.name).toLowerCase())
      ) {
        // File extension check is case-insensitive now
        files.push(fullPath); // Add relevant files
      }
    }
  } catch (error) {
    // Log specific errors like EACCES, but don't stop the whole scan
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((error as any).code === "EACCES") {
      log(`Permission denied accessing directory ${dir}. Skipping.`);
    } else {
      log(`Error scanning directory ${dir}: ${error}`);
    }
  }
  return files;
}
