// src/utils/graph/builder.ts
import type {
    DependencyEdge,
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
  
  const MAX_BATCH_CHARS = 100_000; // Adjust based on model and typical file sizes
  
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
  
    const batches = createFileBatches(allFiles, MAX_BATCH_CHARS);
    log(`Created ${batches.length} batches for analysis.`);
    // --- Debug Logging ---
    batches.forEach((batch, index) => {
        log(`[Graph Debug] Batch ${index + 1} contains ${batch.length} files.`);
    });
    // --- End Debug Logging ---
  
    const allDiscoveredEdges: Array<DependencyEdge> = [];
  
    const oai = new OpenAI({
      apiKey: apiKey || undefined,
      baseURL: OPENAI_BASE_URL || undefined,
      timeout: OPENAI_TIMEOUT_MS,
    });
  
    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      if (!batch || batch.length === 0) {continue;}
      log(`Analyzing batch ${i + 1}/${batches.length}...`);
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
    consolidateGraphData(graphData, allDiscoveredEdges);
  
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
   * Groups files into batches based on character count limit.
   */
  function createFileBatches(
    files: Array<FileContent>,
    maxBatchChars: number,
  ): Array<Array<FileContent>> {
    const batches: Array<Array<FileContent>> = [];
    let currentBatch: Array<FileContent> = [];
    let currentBatchChars = 0;
  
    for (const file of files) {
      const fileChars = file.content.length;
      // Heuristic: Add overhead for separators and paths per file
      const estimatedChars = fileChars + 100; // Adjusted overhead
  
      if (
        currentBatch.length > 0 &&
        currentBatchChars + estimatedChars > maxBatchChars
      ) {
        batches.push(currentBatch);
        currentBatch = [];
        currentBatchChars = 0;
      }
  
      // Handle files larger than the limit (skip or process individually if needed)
      if (estimatedChars > maxBatchChars) {
        log(
          `Skipping file ${file.path} as it exceeds the batch character limit (${estimatedChars} > ${maxBatchChars})`,
        );
        continue;
      }
  
      currentBatch.push(file);
      currentBatchChars += estimatedChars;
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
  ): Promise<Array<DependencyEdge>> {
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
    // Reference the new text format and remove the strict JSON output instruction.
    const prompt = `
  Analyze the code for the files provided below, separated by '--- File: ...'. For *each* file in this batch, identify:
  1.  **Direct Imports/Requires:** List the *exact* file paths it imports or requires, referencing only paths from the <all_project_files> list.
  2.  **Function/Class Usage:** Identify functions or classes defined in *other* files (from <all_project_files>) that are called or instantiated within this file. Map these usages back to the file path where the function/class is likely defined (must be in <all_project_files>).
  3.  **Conceptual Links:** Identify any other files from <all_project_files> that seem conceptually related (e.g., part of the same feature, module, or data flow) even without direct code references.
  
  Output ONLY the JSON array of dependency edges. Each edge represents 'source depends on target'. Use the format: \`[{"source": "path/to/caller.py", "target": "path/to/callee.py", "type": "import|usage|related"}, ...]\`. Be precise and include explicit dependencies found in the code, as well as strong conceptual links.
  
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
        // --- REMOVED response_format to allow more flexibility ---
        // response_format: { type: "json_object" },
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
      let edges: Array<DependencyEdge> = [];
      try {
        // Attempt 1: Try parsing directly as JSON (might work if model still returns pure JSON)
        const parsedJson = JSON.parse(content);
  
        if (Array.isArray(parsedJson)) {
          edges = parsedJson;
        } else if (
          typeof parsedJson === "object" &&
          // eslint-disable-next-line eqeqeq
          parsedJson !== null &&
          Array.isArray(parsedJson.edges)
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
  
      // Validate the structure of the extracted edges
      if (
        !Array.isArray(edges) ||
        !edges.every(
          (edge) =>
            typeof edge === "object" &&
            // eslint-disable-next-line eqeqeq
            edge !== null &&
            typeof edge.source === "string" &&
            typeof edge.target === "string" &&
            ["import", "usage", "related"].includes(edge.type),
        )
      ) {
        log(
          `Parsed edge list has invalid structure: ${JSON.stringify(edges)}`,
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
   * Merges discovered edges into the main graph data structure, ensuring bidirectionality.
   */
  function consolidateGraphData(
    graphData: GraphData,
    edges: Array<DependencyEdge>,
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
  
      // Add forward dependency (source -> target) if not present
      if (!sourceNode.dependencies.includes(targetPath)) {
        sourceNode.dependencies.push(targetPath);
      }
  
      // Add backward dependency (target <- source) if not present
      if (!targetNode.dependents.includes(sourcePath)) {
        targetNode.dependents.push(sourcePath);
      }
    }
  
    // Optional: Sort dependency/dependent lists for consistency
    for (const node of Object.values(graphData.nodes)) {
      node.dependencies.sort();
      node.dependents.sort();
    }
  }