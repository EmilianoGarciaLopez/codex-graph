import type {
  AnalyzedEdge, // Use the new edge type
  FileContent,
  GraphData,
} from "./types";


import { saveGraph } from "./storage";
import { log, isLoggingEnabled } from "../agent/log.js"; // Corrected import path
import { OPENAI_BASE_URL, OPENAI_TIMEOUT_MS } from "../config";
import {
  getFileContents,
  loadIgnorePatterns,
} from "../singlepass/context_files"; // Assuming these are made exportable
import OpenAI from "openai";
import path from "path";

// --- Updated Constants ---
// Target batch size in approximate tokens (128k)
const MAX_BATCH_TOKENS = 128_000;
// Still use a character limit to skip extremely large individual files quickly
const MAX_SINGLE_FILE_CHARS = 2_000_000; // ~500k tokens
// --- End Updated Constants ---

/**
 * Orchestrates the building of the dependency graph for a project.
 * @param projectRoot Absolute path to the project root directory.
 * @param graphStoragePath Absolute path where the graph JSON should be saved.
 * @param model The OpenAI model to use for analysis.
 * @param apiKey The OpenAI API key.
 */
export async function buildDependencyGraph(
  projectRoot: string,
  graphStoragePath: string,
  model: string,
  apiKey: string,
): Promise<void> {
  log(`Starting dependency graph build for project: ${projectRoot}`);
  const startTime = Date.now();

  const graphData: GraphData = {
    nodes: {},
    metadata: { projectRoot, buildTimestamp: startTime },
  };

  const ignorePatterns = loadIgnorePatterns(); // Load default or project-specific ignores
  const allFiles = await getProjectFiles(projectRoot, ignorePatterns);
  // --- Debug Logging ---
  log(`[Graph Debug] Found ${allFiles.length} files. First 5 paths: ${allFiles.slice(0, 5).map(f => f.path).join(', ')}`);
  // --- End Debug Logging ---
  if (allFiles.length === 0) {
    log("No source files found to build graph.");
    await saveGraph(graphStoragePath, graphData); // Save empty graph
    return;
  }
  const allFilePaths = allFiles.map((f) => f.path);
  log(`Found ${allFiles.length} files for analysis.`);

  // --- Updated File Separation Logic ---
  const normalFiles: Array<FileContent> = [];
  const largeFiles: Array<FileContent> = [];

  for (const file of allFiles) {
      // Still use character limit for skipping *individual* massive files
      const estimatedChars = file.content.length + 100;
      if (estimatedChars > MAX_SINGLE_FILE_CHARS) {
          log(`Skipping file ${file.path} as it exceeds the hard single file character limit (${estimatedChars} > ${MAX_SINGLE_FILE_CHARS})`);
          continue; // Skip entirely
      }
      // Check if file *alone* exceeds the *batch token limit* (using approximation)
      const estimatedTokens = Math.ceil(estimatedChars / 4);
      if (estimatedTokens > MAX_BATCH_TOKENS) {
          largeFiles.push(file); // Will be processed individually
      } else {
          normalFiles.push(file); // Will be batched
      }
  }

  log(`Separated files: ${normalFiles.length} normal, ${largeFiles.length} large (to process individually).`);

  // Batch the normal files using the token limit
  const normalBatches = createFileBatches(normalFiles, MAX_BATCH_TOKENS);
  log(`Created ${normalBatches.length} batches for normal files (target tokens: ${MAX_BATCH_TOKENS}).`);

  // Create single-file batches for large files
  const largeFileBatches = largeFiles.map(largeFile => [largeFile]);
  log(`Created ${largeFileBatches.length} single-file batches for large files.`);

  // Combine all batches
  const allBatches = [...normalBatches, ...largeFileBatches];
  log(`Total batches to process: ${allBatches.length}`);
  // --- End Updated File Separation Logic ---


  // --- Debug Logging ---
  allBatches.forEach((batch, index) => {
      log(`[Graph Debug] Batch ${index + 1} contains ${batch.length} files.`);
  });
  // --- End Debug Logging ---

  const allDiscoveredEdges: Array<AnalyzedEdge> = []; // Use AnalyzedEdge

  const oai = new OpenAI({
    apiKey: apiKey || undefined,
    baseURL: OPENAI_BASE_URL || undefined,
    timeout: OPENAI_TIMEOUT_MS,
  });

  for (let i = 0; i < allBatches.length; i++) {
    const batch = allBatches[i];
    if (!batch || batch.length === 0) {continue;}
    log(`Analyzing batch ${i + 1}/${allBatches.length}...`);
    try {
      // eslint-disable-next-line no-await-in-loop
      const discoveredEdges = await analyzeBatchDependencies(
        batch,
        allFilePaths,
        model,
        oai,
      );
      // --- Debug Logging ---
      log(`[Graph Debug] Batch ${i + 1} analysis returned ${discoveredEdges.length} edges.`);
      // --- End Debug Logging ---
      allDiscoveredEdges.push(...discoveredEdges);
    } catch (error) {
      log(`Error analyzing batch ${i + 1}: ${error}`);
      // Optionally, decide whether to continue or abort on error
    }
  }

  // --- Debug Logging ---
  log(`[Graph Debug] Total discovered edges before consolidation: ${allDiscoveredEdges.length}`);
  // --- End Debug Logging ---

  log(
    `Consolidating ${allDiscoveredEdges.length} discovered edges into graph...`,
  );
  consolidateGraphData(graphData, allDiscoveredEdges); // Pass AnalyzedEdge array

  const endTime = Date.now();
  log(
    `Graph build completed in ${((endTime - startTime) / 1000).toFixed(
      2,
    )} seconds. Final node count: ${Object.keys(graphData.nodes).length}`,
  );

  await saveGraph(graphStoragePath, graphData);
}

/**
 * Retrieves project files respecting ignore patterns.
 * (This might reuse or adapt logic from singlepass/context_files.ts)
 */
async function getProjectFiles(
  projectRoot: string,
  ignorePatterns: Array<RegExp>,
): Promise<Array<FileContent>> {
  // Assuming getFileContents is adapted/exported
  return getFileContents(projectRoot, ignorePatterns);
}

/**
 * Groups files into batches based on approximate token count limit.
 */
function createFileBatches(
  files: Array<FileContent>, // Receives only files that individually fit within MAX_BATCH_TOKENS
  maxBatchTokens: number, // Renamed parameter
): Array<Array<FileContent>> {
  const batches: Array<Array<FileContent>> = [];
  let currentBatch: Array<FileContent> = [];
  let currentBatchTokens = 0; // Renamed variable

  for (const file of files) {
    const fileChars = file.content.length;
    // Heuristic: Add overhead for separators and paths per file
    const overhead = 100;
    // Approximate tokens for the current file using 4 chars/token
    const estimatedTokens = Math.ceil((fileChars + overhead) / 4);

    // Check if adding this file would exceed the token limit
    if (
      currentBatch.length > 0 &&
      currentBatchTokens + estimatedTokens > maxBatchTokens
    ) {
      batches.push(currentBatch);
      currentBatch = [];
      currentBatchTokens = 0;
    }

    // Add the file (it's guaranteed to fit individually based on pre-filtering)
    currentBatch.push(file);
    currentBatchTokens += estimatedTokens;
  }

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}

/**
 * Sends a batch of files to the LLM for dependency analysis.
 */
async function analyzeBatchDependencies(
  batchFiles: Array<FileContent>,
  allFilePaths: Array<string>,
  model: string,
  oai: OpenAI,
): Promise<Array<AnalyzedEdge>> { // Return AnalyzedEdge
  // --- Use simpler text format instead of XML ---
  const batchFileContentText = batchFiles
    .map(
      (fc) => `--- File: ${fc.path} ---\n${fc.content}\n--- End File: ${fc.path} ---`
    )
    .join("\n\n");
  // --- End simpler text format ---

  const allFilesXml = `<all_project_files>${allFilePaths
    .map((p) => `<path>${p}</path>`)
    .join("")}</all_project_files>`;

  // --- Updated Prompt ---
  // Ask for relationship summary
  const prompt = `
Analyze the code for the files provided below, separated by '--- File: ...'. For *each* file in this batch, identify its relationships with other files listed in <all_project_files>. For each identified relationship (source -> target):
1.  Determine the **type**: "import" (direct import/require), "usage" (calls function/uses class defined elsewhere), or "related" (conceptual link, same feature/module).
2.  Provide a **concise summary** (max 1-2 sentences) explaining *how* the source file relates to the target file (e.g., "Imports function X", "Uses component Y for UI", "Handles data processing for Z").

Output ONLY the JSON array of dependency edges. Each edge represents 'source depends on target'. Use the format: \`[{"source": "path/to/caller.py", "target": "path/to/callee.py", "type": "import|usage|related", "summary": "Concise explanation of the relationship."}, ...]\`. Be precise and include explicit dependencies found in the code, as well as strong conceptual links. Ensure paths are absolute or relative to the project root provided in the file list.

${batchFileContentText}

${allFilesXml}
`;
  // --- End Updated Prompt ---

  if (isLoggingEnabled()) {
    log(`Sending batch analysis prompt (length: ${prompt.length})`);
  }

  try {
    const response = await oai.chat.completions.create({
      model: model,
      messages: [{ role: "user", content: prompt }],
      // response_format: { type: "json_object" }, // Keep allowing flexible output for now
    });

    const content = response.choices[0]?.message?.content;
    // --- Debug Logging ---
    log(`[Graph Debug] Raw LLM response content: ${content ?? '(empty)'}`);
    // --- End Debug Logging ---
    if (!content) {
      log("LLM response content is empty for batch analysis.");
      return [];
    }

    // --- Parsing Logic (Keep fallback) ---
    let edges: Array<AnalyzedEdge> = []; // Expect AnalyzedEdge
    try {
      // Attempt 1: Try parsing directly as JSON
      const parsedJson = JSON.parse(content);

      if (Array.isArray(parsedJson)) {
        edges = parsedJson;
      } else if (
        typeof parsedJson === "object" &&
        // eslint-disable-next-line eqeqeq
        parsedJson !== null &&
        Array.isArray(parsedJson.edges) // Check if response wrapped in { "edges": [...] }
      ) {
        edges = parsedJson.edges;
      } else {
         // If it's valid JSON but not the expected structure, try the regex fallback
         throw new Error("Parsed JSON is not the expected array or object structure.");
      }
    } catch (initialParseError) {
      // Fallback: Try extracting array from potentially unstructured text
      try {
        const jsonMatch = content.match(/(\[[\s\S]*\])/); // More general regex to find [...]
        if (jsonMatch && jsonMatch[0]) {
          edges = JSON.parse(jsonMatch[0]);
           // Additional check if the extracted part is actually an array
           if (!Array.isArray(edges)) {
             log(`Extracted JSON is not an array: ${jsonMatch[0]}`);
             return [];
           }
        } else {
          log(
            `Could not extract valid JSON edge list from LLM response: ${content}`,
          );
          return [];
        }
      } catch (fallbackParseError) {
        log(
          `[Graph Debug] Failed to parse JSON edge list from LLM response: ${fallbackParseError}\nResponse content: ${content}`,
        );
        return [];
      }
    }
    // --- End Parsing Logic ---

    // Validate the structure of the extracted edges, including the summary
    if (
      !Array.isArray(edges) ||
      !edges.every(
        (edge) =>
          typeof edge === "object" &&
          // eslint-disable-next-line eqeqeq
          edge !== null &&
          typeof edge.source === "string" &&
          typeof edge.target === "string" &&
          ["import", "usage", "related"].includes(edge.type) &&
          typeof edge.summary === "string" // Check for summary
      )
    ) {
      log(
        `Parsed edge list has invalid structure or missing summaries: ${JSON.stringify(edges)}`,
      );
      return [];
    }

    // --- Debug Logging ---
    log(`[Graph Debug] Parsed ${edges.length} edges from LLM response.`);
    // --- End Debug Logging ---
    return edges;
  } catch (error) {
    log(`LLM API call failed during batch analysis: ${error}`);
    throw error; // Re-throw to be caught by the main loop
  }
}

/**
 * Merges discovered edges into the main graph data structure, ensuring bidirectionality and storing summaries.
 */
function consolidateGraphData(
  graphData: GraphData,
  edges: Array<AnalyzedEdge>, // Accept AnalyzedEdge
): void {
  for (const edge of edges) {
    // --- Robust path handling ---
    // Ensure paths are absolute and normalized before using as keys
    let sourcePath: string;
    let targetPath: string;
    try {
        // Assume paths from LLM might be relative to project root or absolute
        sourcePath = path.resolve(graphData.metadata.projectRoot ?? process.cwd(), edge.source);
        targetPath = path.resolve(graphData.metadata.projectRoot ?? process.cwd(), edge.target);
    } catch (pathError) {
        log(`[Graph Debug] Error resolving paths for edge: ${JSON.stringify(edge)} - ${pathError}`);
        continue; // Skip edge if paths are invalid
    }
    // --- End Robust path handling ---


    // --- Debug Logging ---
    if (isLoggingEnabled()) {
        log(`[Graph Debug] Consolidating edge: ${edge.source} (${sourcePath}) -> ${edge.target} (${targetPath})`);
    }
    // --- End Debug Logging ---

    // Ensure nodes exist
    if (!graphData.nodes[sourcePath]) {
      graphData.nodes[sourcePath] = {
        id: sourcePath,
        dependencies: [],
        dependents: [],
      };
    }
    if (!graphData.nodes[targetPath]) {
      graphData.nodes[targetPath] = {
        id: targetPath,
        dependencies: [],
        dependents: [],
      };
    }

    const sourceNode = graphData.nodes[sourcePath]!;
    const targetNode = graphData.nodes[targetPath]!;

    // Add forward dependency (source -> target) with summary if not present
    if (!sourceNode.dependencies.some(dep => dep.id === targetPath)) {
      sourceNode.dependencies.push({ id: targetPath, summary: edge.summary });
    }

    // Add backward dependency (target <- source) with summary if not present
    if (!targetNode.dependents.some(dep => dep.id === sourcePath)) {
      // Note: The summary describes the source->target relationship.
      // We store the same summary on the dependent link for retrieval.
      targetNode.dependents.push({ id: sourcePath, summary: edge.summary });
    }
  }

  // Optional: Sort dependency/dependent lists for consistency
  for (const node of Object.values(graphData.nodes)) {
    node.dependencies.sort((a, b) => a.id.localeCompare(b.id));
    node.dependents.sort((a, b) => a.id.localeCompare(b.id));
  }
}
