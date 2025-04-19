// src/utils/graph/updater.ts
import type {
    DependencyEdge,
    FileContent,
    GraphData,
    GraphNode,
  } from "./types";

  import { log, isLoggingEnabled } from "../agent/log";
  import { OPENAI_BASE_URL, OPENAI_TIMEOUT_MS } from "../config";
  import * as fs from "fs/promises";
  import OpenAI from "openai";
  import * as path from "path";

  type ChangedFile = {
    path: string;
    changeType: "create" | "update" | "delete";
  };

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
        if (mutableNodes[deletedPath]) {
          log(`Removing node ${deletedPath} from graph.`);
          // Remove references from dependents' dependency lists
          for (const dependentId of mutableNodes[deletedPath]!.dependents) {
            const dependentNode = mutableNodes[dependentId];
            if (dependentNode) {
              dependentNode.dependencies = dependentNode.dependencies.filter(
                (dep) => dep !== deletedPath,
              );
            }
          }
          // Remove references from dependencies' dependent lists
          for (const dependencyId of mutableNodes[deletedPath]!.dependencies) {
            const dependencyNode = mutableNodes[dependencyId];
            if (dependencyNode) {
              dependencyNode.dependents = dependencyNode.dependents.filter(
                (dep) => dep !== deletedPath,
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
          const newEdges = await analyzeFileDependencies(
            fileContent,
            allFilePaths,
            model,
            oai,
          );

          // Get old connections before updating/creating the node
          const oldNode = mutableNodes[filePath];
          const oldDependencies = new Set(oldNode?.dependencies ?? []);
          const oldDependents = new Set(oldNode?.dependents ?? []);

          // Update or create the node for the changed file
          const newNode: GraphNode = {
            id: filePath,
            dependencies: [],
            dependents: [],
          };
          mutableNodes[filePath] = newNode;

          const currentDependencies = new Set<string>();
          const currentDependents = new Set<string>();

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

            if (sourcePath === filePath) {
              // Edge originating from the changed file (dependency)
              currentDependencies.add(targetPath);
              if (!mutableNodes[targetPath]!.dependents.includes(filePath)) {
                mutableNodes[targetPath]!.dependents.push(filePath);
              }
            } else if (targetPath === filePath) {
              // Edge terminating at the changed file (dependent)
              currentDependents.add(sourcePath);
              if (!mutableNodes[sourcePath]!.dependencies.includes(filePath)) {
                mutableNodes[sourcePath]!.dependencies.push(filePath);
              }
            }
          }

          newNode.dependencies = Array.from(currentDependencies).sort();
          newNode.dependents = Array.from(currentDependents).sort();

          // Clean up stale references in neighbors
          // Removed dependencies: Neighbors that were depended on by the old node but not the new one
          for (const oldDepId of oldDependencies) {
            if (!currentDependencies.has(oldDepId)) {
              const neighbor = mutableNodes[oldDepId];
              if (neighbor) {
                neighbor.dependents = neighbor.dependents.filter(
                  (id) => id !== filePath,
                );
              }
            }
          }
          // Removed dependents: Neighbors that depended on the old node but not the new one
          for (const oldDependentId of oldDependents) {
            if (!currentDependents.has(oldDependentId)) {
              const neighbor = mutableNodes[oldDependentId];
              if (neighbor) {
                neighbor.dependencies = neighbor.dependencies.filter(
                  (id) => id !== filePath,
                );
              }
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
  ): Promise<Array<DependencyEdge>> {
    const allFilesXml = `<all_project_files>${allFilePaths
      .map((p) => `<path>${p}</path>`)
      .join("")}</all_project_files>`;

    const prompt = `
Analyze the provided code file (${
      fileContent.path
    }) and its content. Based *only* on the code and the provided list of other files (${allFilePaths.join(
      ", ",
    )}) in the project, identify:
1.  **Direct Dependencies:** A list of file paths that this file directly imports, calls, or relies on (type: "import" or "usage").
2.  **Potential Dependents:** A list of file paths that likely import, call, or rely on elements defined *within* this file (type: "dependent").
3.  **Related Files:** A list of file paths that are conceptually or functionally related (type: "related").

Focus on explicit relationships inferable from the code or common programming patterns, and strong conceptual links. Output ONLY JSON in the specified format: \`[{"source": "path/to/this/file.py", "target": "path/to/dependency.py", "type": "import|usage|related"}, {"source": "path/to/dependent.py", "target": "path/to/this/file.py", "type": "dependent"}, ...]\`.

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
        response_format: { type: "json_object" },
      });

      const content = response.choices[0]?.message?.content;
      if (!content) {
        log("LLM response content is empty for single file analysis.");
        return [];
      }

      let edges: Array<DependencyEdge> = [];
      try {
        edges = JSON.parse(content);
      } catch {
        try {
          const jsonMatch = content.match(/\[\s*{[\s\S]*?}\s*]/);
          if (jsonMatch && jsonMatch[0]) {
            edges = JSON.parse(jsonMatch[0]);
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

      // Validate structure
      if (
        !Array.isArray(edges) ||
        !edges.every(
          (edge) =>
            typeof edge === "object" &&
            // eslint-disable-next-line eqeqeq
            edge !== null &&
            typeof edge.source === "string" &&
            typeof edge.target === "string" &&
            ["import", "usage", "related", "dependent"].includes(edge.type),
        )
      ) {
        log(
          `Parsed edge list has invalid structure: ${JSON.stringify(edges)}`,
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