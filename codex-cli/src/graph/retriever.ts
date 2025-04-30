import type { GraphNode, GraphEdge } from "./types";
import type { AppConfig } from "../utils/config"; // Import AppConfig

import { approximateTokensUsed } from "../utils/approximate-tokens-used";
import { getApiKey, getBaseUrl } from "../utils/config"; // Import config utils
import { log } from "../utils/logger/log";
import { getGraphDir, loadNodes, loadEdges } from "./storage";
import OpenAI from "openai"; // Import OpenAI
import fs from "fs/promises";
import path from "path";
import { z } from "zod"; // Import Zod for parsing LLM response

// --- Zod Schema for LLM Response ---
const RelevantNodesSchema = z.object({
  relevant_node_ids: z
    .array(z.string())
    .describe(
      "An array containing the string IDs of the graph nodes deemed most relevant to the user's query.",
    ),
});

/**
 * Uses an LLM to identify relevant nodes based on a query and the graph structure.
 * Handles potential context window limitations by chunking if necessary (basic implementation).
 *
 * @param query - The natural language query from the agent.
 * @param allNodes - All nodes loaded from the graph.
 * @param allEdges - All edges loaded from the graph.
 * @param config - The application configuration (needed for LLM call).
 * @returns A promise resolving to an array of relevant node IDs.
 */
async function identifyRelevantNodesWithLLM(
  query: string,
  allNodes: GraphNode[],
  allEdges: GraphEdge[],
  config: AppConfig,
): Promise<string[]> {
  log(`LLM Identifier: Identifying relevant nodes for query: "${query}"`);
  const apiKey = getApiKey(config.provider);
  const baseURL = getBaseUrl(config.provider);
  const model = config.model; // Use the configured model

  if (!apiKey && config.provider?.toLowerCase() !== "ollama") {
    log("LLM Identifier: API key not found for provider: " + config.provider);
    return [];
  }

  const oai = new OpenAI({
    apiKey: apiKey ?? "ollama",
    baseURL,
  });

  // Format graph data for the prompt (simple JSON stringification for now)
  // Exclude potentially large 'hash' and calculated 'tok' fields from the prompt context
  const nodesForPrompt = allNodes.map(({ hash: _h, tok: _t, ...rest }) => rest);
  const edgesForPrompt = allEdges.map(({ fenced: _f, ...rest }) => rest);
  const graphDataString = JSON.stringify(
    { nodes: nodesForPrompt, edges: edgesForPrompt },
    null,
    2,
  ); // Pretty print for readability if needed

  const systemPrompt = `You are a code graph analysis assistant. Given a natural language query and a representation of a codebase graph (nodes representing functions/methods, edges representing calls/references), your task is to identify the node IDs that are most relevant to answering the query.

Consider the query's intent:
- If it asks about a specific function/class/file, prioritize nodes related to that entity.
- If it asks about relationships (e.g., "who calls X?"), prioritize the source/destination nodes of relevant edges.
- If it asks about concepts (e.g., "show me authentication code"), prioritize nodes whose summaries, docs, or tags match the concept.
- **Crucially, consider the dependencies:** Include nodes that are directly called by (downstream) or directly call (upstream) the primary relevant nodes, as they provide essential context for understanding impact and flow.

Input:
- User Query: The natural language question about the codebase.
- Codebase Graph: A JSON object with "nodes" and "edges" arrays. Each node has an 'id', 'sig', 'doc', 'summary', 'tags', etc. Each edge has 'src', 'dst', 'why', 'kind'.

Output: Respond ONLY with a valid JSON object containing a single key: "relevant_node_ids". The value should be an array of strings, where each string is the ID of a relevant node from the provided graph data. Return an empty array if no nodes seem relevant.

Example Output:
{
  "relevant_node_ids": ["src/utils/auth.ts:authenticateUser", "src/api/login.ts:handleLoginRequest", "src/db/user.ts:findUserById"]
}`;

  const userPrompt = `User Query: ${query}\n\nCodebase Graph:\n\`\`\`json\n${graphDataString}\n\`\`\`\n\nIdentify the relevant node IDs based on the query and graph, including direct upstream and downstream dependencies. Respond only with the specified JSON format.`;

  // TODO: Implement proper chunking if graphDataString exceeds context limit.
  // For now, assume it fits or truncate (which is not ideal).
  // const maxTokens = maxTokensForModel(model);
  // const promptTokens = approximateTokensUsed([...]); // Estimate prompt tokens
  // if (promptTokens > maxTokens * 0.8) { // Leave some room for response
  //   log.warn("LLM Identifier: Graph data might exceed context window. Chunking not yet implemented.");
  //   // Implement chunking logic here
  // }

  try {
    log(`LLM Identifier: Sending request to ${model}...`);
    const response = await oai.chat.completions.create({
      model: model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      response_format: { type: "json_object" },
    });

    const jsonContent = response.choices[0]?.message?.content;
    if (!jsonContent) {
      log("LLM Identifier: Received empty response.");
      return [];
    }

    // Parse and validate the response using Zod
    const parsedResult = RelevantNodesSchema.safeParse(JSON.parse(jsonContent));

    if (!parsedResult.success) {
      log(
        `LLM Identifier: Failed to parse or validate LLM response JSON: ${parsedResult.error.message}`,
      );
      log(`LLM Identifier: Raw response content: ${jsonContent}`);
      return [];
    }

    const relevantIds = parsedResult.data.relevant_node_ids;
    log(`LLM Identifier: Identified ${relevantIds.length} relevant nodes.`);
    // Optional: Filter IDs to ensure they actually exist in the graph
    const existingNodeIds = new Set(allNodes.map((n) => n.id));
    const validIds = relevantIds.filter((id) => existingNodeIds.has(id));
    if (validIds.length !== relevantIds.length) {
      log(
        `LLM Identifier: Filtered out ${
          relevantIds.length - validIds.length
        } non-existent node IDs returned by LLM.`,
      );
    }
    return validIds;
  } catch (error) {
    log(`LLM Identifier: Error during LLM call: ${error}`);
    return [];
  }
}

/**
 * Retrieves relevant code context from the graph based on LLM-identified nodes.
 * **Does not include the actual code content of the nodes.** Only includes metadata
 * for the identified nodes and edges *between* those nodes.
 *
 * @param projectRoot - The absolute path to the project root.
 * @param _seedNodeIds - Ignored. Relevant nodes are determined by the LLM based on the query.
 * @param query - The natural language query from the agent.
 * @param budgetTokens - The maximum number of tokens allowed for the context.
 * @param config - The application configuration (needed for LLM call).
 * @returns A promise resolving to a string containing the formatted context (nodes and edges only).
 */
export async function getContext(
  projectRoot: string,
  _seedNodeIds: string[], // This parameter is now ignored
  query: string | undefined,
  budgetTokens: number,
  config: AppConfig, // Added config parameter
): Promise<string> {
  log(
    `Graph Retriever: Getting context (nodes/edges only). Query: "${
      query || "N/A"
    }", Budget: ${budgetTokens}`,
  );
  const graphDir = await getGraphDir(projectRoot);

  // Load graph data
  const allNodes = await loadNodes(graphDir);
  const allEdges = await loadEdges(graphDir);

  if (allNodes.length === 0) {
    log("Graph Retriever: No nodes found in graph.");
    return "// No graph data found. Run 'codex index' first.";
  }

  let relevantNodeIds: string[] = [];
  if (query) {
    // Use LLM to identify relevant nodes based on the query
    relevantNodeIds = await identifyRelevantNodesWithLLM(
      query,
      allNodes,
      allEdges,
      config,
    );
    log(
      `Graph Retriever: Using ${relevantNodeIds.length} nodes identified by LLM.`,
    );
    if (relevantNodeIds.length === 0) {
      log(
        "Graph Retriever: LLM identified no relevant nodes for the query. Returning empty context.",
      );
      return "// LLM found no relevant nodes for the query.";
    }
  } else {
    log(
      "Graph Retriever: No query provided to graph tool. Returning empty context.",
    );
    return "// No query provided to the graph tool.";
  }

  // --- Start of context retrieval logic using identified relevantNodeIds ---

  const nodesMap = new Map<string, GraphNode>(allNodes.map((n) => [n.id, n]));
  const relevantNodeIdSet = new Set(relevantNodeIds);

  let promptString =
    "// Relevant graph context (nodes and edges only) based on the query:\n\n";
  let currentTokenCount = approximateTokensUsed([
    { type: "message", role: "system", content: [{ type: "input_text", text: promptString }] },
  ]);

  // Process only the nodes identified by the LLM
  log(
    `Graph Retriever: Processing metadata for ${relevantNodeIds.length} relevant nodes.`,
  );

  for (const nodeId of relevantNodeIds) {
    const node = nodesMap.get(nodeId);
    if (!node) continue; // Should not happen if LLM returns valid IDs

    const nodeHeader = `// Node: ${node.id}\n// Path: ${node.ptr[0]}\n// Lines: ${node.ptr[1]}-${node.ptr[2]}\n// Signature: ${node.sig}\n// Doc: ${node.doc}\n// Summary: ${node.summary}\n// Tags: [${(node.tags ?? []).join(', ')}]\n\n`;
    const headerTokens = approximateTokensUsed([
      { type: "message", role: "system", content: [{ type: "input_text", text: nodeHeader }] },
    ]);

    if (currentTokenCount + headerTokens <= budgetTokens) {
      promptString += nodeHeader;
      currentTokenCount += headerTokens;
    } else {
      log(
        `Graph Retriever: Budget limit reached trying to add header for node ${node.id}. Stopping node processing.`,
      );
      break; // Stop adding nodes if budget exceeded
    }
  }

  // Add edges *between* the relevant nodes
  log(
    `Graph Retriever: Adding edges between relevant nodes. Current tokens: ${currentTokenCount}`,
  );
  for (const edge of allEdges) {
    // Only include edges where BOTH source and destination are in the relevant set
    if (relevantNodeIdSet.has(edge.src) && relevantNodeIdSet.has(edge.dst)) {
      const edgeInfo = `// Edge: ${edge.src} -> ${edge.dst} (Kind: ${
        edge.kind || "call"
      })\n// Why: ${edge.why}\n\n`;
      const edgeTokens = approximateTokensUsed([
        { type: "message", role: "system", content: [{ type: "input_text", text: edgeInfo }] },
      ]);

      if (currentTokenCount + edgeTokens <= budgetTokens) {
        promptString += edgeInfo;
        currentTokenCount += edgeTokens;
      } else {
        log(
          `Graph Retriever: Budget limit reached while trying to add edge info ${edge.src}->${edge.dst}. Stopping edge processing.`,
        );
        break; // Stop adding edges if budget exceeded
      }
    }
  }

  if (
    promptString.trim() ===
    "// Relevant graph context (nodes and edges only) based on the query:"
  ) {
    promptString +=
      "// No specific context found within the budget for the identified relevant nodes.";
  }

  log(
    `Graph Retriever: Final context size: ${currentTokenCount} tokens (approx). Returning context string length: ${promptString.length}`,
  );
  return promptString;
}