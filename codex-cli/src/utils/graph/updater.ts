import type {
  AnalyzedEdge, // Use AnalyzedEdge
  FileContent,
  GraphData,
  GraphNode,
  ChangedFile,
  GraphEdgeInfo, // Import GraphEdgeInfo
} from "./types";

import { log, isLoggingEnabled } from "../agent/log";
import { OPENAI_BASE_URL, OPENAI_TIMEOUT_MS } from "../config";
import * as fs from "fs/promises";
import OpenAI from "openai";
import * as path from "path";


/**
 * Updates the dependency graph based on file changes.
 * @param changedFiles An array describing the files that changed.
 * @param currentGraph The current graph data.
 * @param model The OpenAI model to use for re-analysis.
 * @param apiKey The OpenAI API key.
 * @param allFilePaths A list of all known file paths in the project.
 * @returns The updated graph data.
 */
export async function updateGraphForChanges(
  changedFiles: Array<ChangedFile>,
  currentGraph: GraphData,
  model: string,
  apiKey: string,
  allFilePaths: Array<string>,
): Promise<GraphData> {
  log(
    `Updating graph for ${
      changedFiles.length
    } changes: ${changedFiles.map((f) => `${f.changeType}:${f.path}`)}`,
  );
  const mutableNodes = { ...currentGraph.nodes };
  const projectRoot = currentGraph.metadata.projectRoot ?? process.cwd();

  const oai = new OpenAI({
    apiKey: apiKey || undefined,
    baseURL: OPENAI_BASE_URL || undefined,
    timeout: OPENAI_TIMEOUT_MS,
  });

  // Process deletions first
  for (const change of changedFiles) {
    if (change.changeType === "delete") {
      const deletedPath = path.resolve(projectRoot, change.path);
      const nodeToDelete = mutableNodes[deletedPath];
      if (nodeToDelete) {
        log(`Removing node ${deletedPath} from graph.`);
        // Remove references from dependents' dependency lists
        for (const dependentInfo of nodeToDelete.dependents) {
          const dependentNode = mutableNodes[dependentInfo.id];
          if (dependentNode) {
            dependentNode.dependencies = dependentNode.dependencies.filter(
              (dep) => dep.id !== deletedPath,
            );
          }
        }
        // Remove references from dependencies' dependent lists
        for (const dependencyInfo of nodeToDelete.dependencies) {
          const dependencyNode = mutableNodes[dependencyInfo.id];
          if (dependencyNode) {
            dependencyNode.dependents = dependencyNode.dependents.filter(
              (dep) => dep.id !== deletedPath,
            );
          }
        }
        // Delete the node itself
        delete mutableNodes[deletedPath];
      }
    }
  }

  // Process creations and updates
  for (const change of changedFiles) {
    if (change.changeType === "create" || change.changeType === "update") {
      const filePath = path.resolve(projectRoot, change.path);
      let fileContent: FileContent | null = null;
      try {
        // eslint-disable-next-line no-await-in-loop
        const content = await fs.readFile(filePath, "utf-8");
        fileContent = { path: filePath, content };
      } catch (error) {
        log(`Could not read file ${filePath} for graph update: ${error}`);
        continue; // Skip if file can't be read
      }

      if (fileContent) {
        log(`Re-analyzing dependencies for ${change.changeType}d file: ${filePath}`);
        // eslint-disable-next-line no-await-in-loop
        const newEdges = await analyzeFileDependencies( // Expect AnalyzedEdge
          fileContent,
          allFilePaths,
          model,
          oai,
        );

        // Get old connections before updating/creating the node
        const oldNode = mutableNodes[filePath];
        const oldDependencies = new Map(oldNode?.dependencies.map(dep => [dep.id, dep.summary]));
        const oldDependents = new Map(oldNode?.dependents.map(dep => [dep.id, dep.summary]));

        // Update or create the node for the changed file
        const newNode: GraphNode = {
          id: filePath,
          dependencies: [],
          dependents: [],
        };
        mutableNodes[filePath] = newNode;

        const currentDependencies = new Map<string, GraphEdgeInfo>();
        const currentDependents = new Map<string, GraphEdgeInfo>();

        // Process new edges related to this file
        for (const edge of newEdges) {
          const sourcePath = path.resolve(projectRoot, edge.source);
          const targetPath = path.resolve(projectRoot, edge.target);

          // Ensure neighbor nodes exist
          if (!mutableNodes[sourcePath])
            {mutableNodes[sourcePath] = {
              id: sourcePath,
              dependencies: [],
              dependents: [],
            };}
          if (!mutableNodes[targetPath])
            {mutableNodes[targetPath] = {
              id: targetPath,
              dependencies: [],
              dependents: [],
            };}

          const neighborSourceNode = mutableNodes[sourcePath]!;
          const neighborTargetNode = mutableNodes[targetPath]!;

          if (sourcePath === filePath) {
            // Edge originating from the changed file (dependency)
            const depInfo: GraphEdgeInfo = { id: targetPath, summary: edge.summary };
            currentDependencies.set(targetPath, depInfo);
            // Update neighbor's dependents list
            if (!neighborTargetNode.dependents.some(dep => dep.id === filePath)) {
              neighborTargetNode.dependents.push({ id: filePath, summary: edge.summary });
            } else {
               // Update summary if edge already exists
               const existingDep = neighborTargetNode.dependents.find(dep => dep.id === filePath);
               if (existingDep) {existingDep.summary = edge.summary;}
            }

          } else if (targetPath === filePath) {
            // Edge terminating at the changed file (dependent)
            const depInfo: GraphEdgeInfo = { id: sourcePath, summary: edge.summary };
            currentDependents.set(sourcePath, depInfo);
             // Update neighbor's dependencies list
            if (!neighborSourceNode.dependencies.some(dep => dep.id === filePath)) {
              neighborSourceNode.dependencies.push({ id: filePath, summary: edge.summary });
            } else {
               // Update summary if edge already exists
               const existingDep = neighborSourceNode.dependencies.find(dep => dep.id === filePath);
               if (existingDep) {existingDep.summary = edge.summary;}
            }
          }
        }

        newNode.dependencies = Array.from(currentDependencies.values()).sort((a, b) => a.id.localeCompare(b.id));
        newNode.dependents = Array.from(currentDependents.values()).sort((a, b) => a.id.localeCompare(b.id));

        // Clean up stale references in neighbors
        // Removed dependencies: Neighbors that were depended on by the old node but not the new one
        for (const [oldDepId] of oldDependencies) {
          if (!currentDependencies.has(oldDepId)) {
            const neighbor = mutableNodes[oldDepId];
            if (neighbor) {
              neighbor.dependents = neighbor.dependents.filter(
                (dep) => dep.id !== filePath,
              );
            }
          }
        }
        // Removed dependents: Neighbors that depended on the old node but not the new one
        for (const [oldDependentId] of oldDependents) {
          if (!currentDependents.has(oldDependentId)) {
            const neighbor = mutableNodes[oldDependentId];
            if (neighbor) {
              neighbor.dependencies = neighbor.dependencies.filter(
                (dep) => dep.id !== filePath,
              );
            }
          }
        }

        // Sort neighbor lists after updates
        for (const neighborId of [...currentDependencies.keys(), ...currentDependents.keys()]) {
            const neighbor = mutableNodes[neighborId];
            if (neighbor) {
                neighbor.dependencies.sort((a, b) => a.id.localeCompare(b.id));
                neighbor.dependents.sort((a, b) => a.id.localeCompare(b.id));
            }
        }
      }
    }
  }

  // Update the graph data object
  currentGraph.nodes = mutableNodes;
  currentGraph.metadata.buildTimestamp = Date.now(); // Update timestamp
  log("Graph update complete.");
  return currentGraph;
}

/**
 * Analyzes a single file's dependencies using the LLM.
 * This is used during graph updates.
 */
async function analyzeFileDependencies(
  fileContent: FileContent,
  allFilePaths: Array<string>,
  model: string,
  oai: OpenAI,
): Promise<Array<AnalyzedEdge>> { // Return AnalyzedEdge
  const allFilesXml = `<all_project_files>${allFilePaths
    .map((p) => `<path>${p}</path>`)
    .join("")}</all_project_files>`;

  // Updated prompt to ask for summaries
  const prompt = `
Analyze the provided code file (${
    fileContent.path
  }) and its content. Based *only* on the code and the provided list of other files (${allFilePaths.join(
    ", ",
  )}) in the project (<all_project_files>), identify its relationships:
1.  **Direct Dependencies:** Files this file directly imports, calls, or relies on (type: "import" or "usage").
2.  **Potential Dependents:** Files that likely import, call, or rely on elements defined *within* this file (type: "dependent").
3.  **Related Files:** Files that are conceptually or functionally related (type: "related").

For each relationship found (source -> target), provide a **concise summary** (max 1-2 sentences) explaining *how* the files are related (e.g., "Imports function X", "Uses component Y for UI", "Handles data processing for Z").

Focus on explicit relationships inferable from the code or common programming patterns, and strong conceptual links. Output ONLY JSON in the specified format: \`[{"source": "path/to/source.py", "target": "path/to/target.py", "type": "import|usage|related|dependent", "summary": "Concise explanation."}, ...]\`. Ensure paths are absolute or relative to the project root.

<file path="${fileContent.path}">
<content><![CDATA[${fileContent.content}]]></content>
</file>

${allFilesXml}
`;

  if (isLoggingEnabled()) {
    log(`Sending single file analysis prompt (length: ${prompt.length})`);
  }

  try {
    const response = await oai.chat.completions.create({
      model: model,
      messages: [{ role: "user", content: prompt }],
      // response_format: { type: "json_object" }, // Keep allowing flexible output
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      log("LLM response content is empty for single file analysis.");
      return [];
    }

    let edges: Array<AnalyzedEdge> = []; // Expect AnalyzedEdge
    try {
      // Attempt 1: Parse directly
      const parsedJson = JSON.parse(content);
      if (Array.isArray(parsedJson)) {
          edges = parsedJson;
      } else if (typeof parsedJson === 'object' && parsedJson != null && Array.isArray(parsedJson.edges)) {
          edges = parsedJson.edges; // Handle { "edges": [...] }
      } else {
          throw new Error("Parsed JSON is not the expected array or object structure.");
      }
    } catch (initialParseError) {
      // Fallback: Extract from text
      try {
        const jsonMatch = content.match(/(\[[\s\S]*\])/);
        if (jsonMatch && jsonMatch[0]) {
          edges = JSON.parse(jsonMatch[0]);
          if (!Array.isArray(edges)) {
              log(`Extracted JSON is not an array: ${jsonMatch[0]}`);
              return [];
          }
        } else {
          log(
            `Could not extract valid JSON edge list from single file LLM response: ${content}`,
          );
          return [];
        }
      } catch (parseError) {
        log(
          `Failed to parse JSON edge list from single file LLM response: ${parseError}\nResponse content: ${content}`,
        );
        return [];
      }
    }

    // Validate structure including summary
    if (
      !Array.isArray(edges) ||
      !edges.every(
        (edge) =>
          typeof edge === "object" &&
          // eslint-disable-next-line eqeqeq
          edge !== null &&
          typeof edge.source === "string" &&
          typeof edge.target === "string" &&
          ["import", "usage", "related", "dependent"].includes(edge.type) &&
          typeof edge.summary === "string" // Check for summary
      )
    ) {
      log(
        `Parsed edge list has invalid structure or missing summaries: ${JSON.stringify(edges)}`,
      );
      return [];
    }

    log(`Received ${edges.length} edges from LLM for single file.`);
    return edges;
  } catch (error) {
    log(`LLM API call failed during single file analysis: ${error}`);
    throw error;
  }
}
