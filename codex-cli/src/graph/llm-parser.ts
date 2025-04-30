import type { GraphNode, GraphEdge } from "./types";
import type { AppConfig } from "../utils/config";

import { approximateTokensUsed } from "../utils/approximate-tokens-used";
import { getApiKey, getBaseUrl } from "../utils/config";
import { log } from "../utils/logger/log";
import OpenAI, { APIError } from "openai"; // Import APIError
import path from "path";
import { setTimeout } from "timers/promises"; // Import setTimeout for async delay

interface LLMParseResult {
  nodes: Array<Omit<GraphNode, "tok" | "hash">>; // LLM provides structure, 'tok'/'hash' calculated later
  edges: Array<Omit<GraphEdge, "fenced">>; // LLM provides structure, 'fenced' defaults later
}

// --- Rate Limit Handling Constants ---
const MAX_RETRIES = 5;
const INITIAL_BACKOFF_MS = 1000; // 1 second
const MAX_BACKOFF_MS = 30000; // 30 seconds
const JITTER_FACTOR = 0.5; // Apply up to 50% jitter

/**
 * Uses an LLM to parse a file's content and extract graph nodes (functions/methods)
 * and edges (calls/references). Includes rate limit handling with exponential backoff.
 *
 * **Uses gpt-4.1 specifically for this parsing task.**
 *
 * @param projectRoot - Absolute path to the project root.
 * @param filePath - Absolute path to the file being parsed.
 * @param fileContent - The content of the file.
 * @param config - The application configuration (used for API key/base URL).
 * @returns A promise resolving to the extracted nodes and edges.
 */
export async function parseFileWithLLM(
  projectRoot: string,
  filePath: string,
  fileContent: string,
  config: AppConfig,
): Promise<{ nodes: Array<GraphNode>; edges: Array<GraphEdge> }> {
  const relativePath = path.relative(projectRoot, filePath);
  const apiKey = getApiKey(config.provider);
  const baseURL = getBaseUrl(config.provider);
  const model = "gpt-4.1"; // Hardcode the model for graph parsing

  if (!apiKey && config.provider?.toLowerCase() !== "ollama") {
    log("LLM Parser: API key not found for provider: " + config.provider);
    return { nodes: [], edges: [] }; // Cannot proceed without API key (except Ollama)
  }

  const oai = new OpenAI({
    apiKey: apiKey ?? "ollama", // Use a dummy key for Ollama if none provided
    baseURL,
  });

  const systemPrompt = `
You are an expert code analysis tool. Your task is to parse the provided code file content and extract structured information about functions/methods and their relationships (calls) using a **strict and consistent identification scheme**.

Input: You will receive the relative file path and the full content of a source code file.

Output: Respond ONLY with a valid JSON object containing two keys: "nodes" and "edges".

**Node ID Format Rules (CRITICAL):**
- For functions/methods defined **inside a class**, the ID MUST be: \`"relative/path/to/file:ClassName.methodName"\`.
- For functions defined **outside any class** (top-level), the ID MUST be: \`"relative/path/to/file:functionName"\`.
- **Consistency is paramount:** The \`src\` and \`dst\` IDs used in the "edges" array MUST exactly match the format of the \`id\` generated for the corresponding nodes.

**Output Structure:**
- "nodes": An array of objects, where each object represents a function or method found in the file. Each node object must have the following keys:
    - "id": string - A unique identifier following the **strict format rules** defined above.
        - Example (Class Method): \`"src/utils/parser.ts:Parser.parseToken"\`
        - Example (Top-Level Function): \`"src/utils/helpers.ts:calculateTotal"\`
    - "ptr": [string, number, number] - An array containing the relative file path (string), the starting line number (number), and the ending line number (number) of the function/method definition. Use 1-based indexing for line numbers. The file path MUST match the one used in the 'id'.
    - "sig": string - A concise one-line signature of the function/method (e.g., "function calculateTotal(cart): number").
    - "doc": string - The first line of the function/method's docstring, or an empty string if none exists.
    - "summary": string - A brief (max 30 words) summary of the function's purpose.
    - "tags": string[] (optional) - An array of 1-3 concise conceptual tags describing the function's domain or purpose (e.g., ["ui", "state-management"], ["parsing", "utils"], ["authentication", "api"]). Consider the file path and surrounding code for context. Reuse existing relevant tags identified within this file before creating new ones.
- "edges": An array of objects, where each object represents a call or reference from one function/method (source) to another (destination). Each edge object must have the following keys:
    - "src": string - The "id" of the source node, **matching the canonical ID format**.
    - "dst": string - The "id" of the destination node, **matching the canonical ID format**. This might be in a different file. Use your knowledge of imports/modules to determine the destination ID format ("relative/path/to/dst/file:functionName" or "relative/path/to/dst/file:ClassName.methodName"). Ensure the relative path format is consistent with node 'id's (relative to project root). If the destination is a standard library or external dependency, use a conventional name (e.g., "node:fs.readFileSync", "react:useState").
    - "why": string - A brief explanation (max 50 words) of why the source function calls the destination function.
    - "kind": string (optional) - A tag indicating the type of relationship (e.g., "call", "import", "reference"). Default to "call" if unsure.

Constraints:
- Extract information for ALL functions/methods defined at the top level of the file or class. Do NOT generate nodes for helper functions defined *inside* the body of another function or method.
- Line numbers in "ptr" should be 1-based and inclusive.
- Ensure all IDs ("id", "src", "dst") strictly follow the specified canonical format relative to the project root. Resolve import paths accurately relative to the project root.
- If a function calls something you cannot resolve to a specific function ID (e.g., a complex dynamic call), omit the edge.
- Provide accurate line ranges.
- Summaries, 'why' explanations, and tags should be concise and informative.
- Output ONLY the JSON object. Do not include any introductory text, explanations, or markdown formatting around the JSON.
`;

  const userPrompt = `
File Path: ${relativePath}

File Content:
\`\`\`
${fileContent}
\`\`\`

Extract nodes and edges according to the specified JSON format.
`;

  let response: OpenAI.Chat.Completions.ChatCompletion | null = null;
  let lastError: unknown = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      log(
        `LLM Parser: Sending ${relativePath} to ${model} for parsing (Attempt ${
          attempt + 1
        }/${MAX_RETRIES}).`,
      );
      // eslint-disable-next-line no-await-in-loop
      response = await oai.chat.completions.create({
        model: model, // Use the hardcoded model here
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        response_format: { type: "json_object" },
      });
      // If successful, break the loop
      break;
    } catch (error) {
      lastError = error; // Store the last error
      const isRateLimitError =
        error instanceof APIError && error.status === 429;

      if (isRateLimitError && attempt < MAX_RETRIES - 1) {
        // Calculate backoff time with jitter
        const backoff = Math.min(
          INITIAL_BACKOFF_MS * 2 ** attempt,
          MAX_BACKOFF_MS,
        );
        const jitter = backoff * JITTER_FACTOR * (Math.random() - 0.5); // +/- 50% jitter
        const delay = Math.max(0, Math.round(backoff + jitter)); // Ensure non-negative delay

        log(
          `LLM Parser: Rate limit hit for ${relativePath}. Retrying in ${delay}ms (Attempt ${
            attempt + 2
          }/${MAX_RETRIES})...`,
        );
        // eslint-disable-next-line no-await-in-loop
        await setTimeout(delay);
      } else {
        // Non-retryable error or max retries reached
        log(
          `LLM Parser: Failed to parse file ${relativePath} after ${
            attempt + 1
          } attempts. Error: ${error}`,
        );
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        if ((error as any).response) {
          log(
            `LLM Parser: API Response Error: ${JSON.stringify(
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (error as any).response.data,
            )}`,
          );
        }
        return { nodes: [], edges: [] }; // Return empty on failure
      }
    }
  }

  // If response is still null after retries, it means all attempts failed
  if (!response) {
    log(
      `LLM Parser: All retry attempts failed for ${relativePath}. Last error: ${lastError}`,
    );
    return { nodes: [], edges: [] };
  }

  // --- Process successful response ---
  try {
    const jsonContent = response.choices[0]?.message?.content;
    if (!jsonContent) {
      log(`LLM Parser: Received empty response content for ${relativePath}`);
      return { nodes: [], edges: [] };
    }

    const parsedResult = JSON.parse(jsonContent) as LLMParseResult;

    // Validate the basic structure
    if (
      !parsedResult ||
      !Array.isArray(parsedResult.nodes) ||
      !Array.isArray(parsedResult.edges)
    ) {
      log(
        `LLM Parser: Invalid JSON structure received for ${relativePath}. Content: ${jsonContent}`,
      );
      return { nodes: [], edges: [] };
    }

    // Post-process nodes to calculate token counts
    const processedNodes: GraphNode[] = parsedResult.nodes
      .map((node) => {
        // Basic validation
        if (
          !node.id ||
          !Array.isArray(node.ptr) ||
          node.ptr.length !== 3 ||
          typeof node.ptr[0] !== "string" ||
          typeof node.ptr[1] !== "number" ||
          typeof node.ptr[2] !== "number"
        ) {
          log(
            `LLM Parser: Invalid node structure for ID ${node.id} in ${relativePath}`,
          );
          return null; // Skip invalid nodes
        }
        const [_, startLine, endLine] = node.ptr;
        // Ensure start/end lines are valid
        if (startLine < 1 || endLine < startLine) {
          log(
            `LLM Parser: Invalid line numbers for node ${node.id} in ${relativePath}: [${startLine}, ${endLine}]`,
          );
          return null;
        }

        const bodyLines = fileContent.split("\n").slice(startLine - 1, endLine);
        const body = bodyLines.join("\n");
        const tok = approximateTokensUsed([
          {
            type: "message",
            role: "user", // Role doesn't matter for token count here
            content: [{ type: "input_text", text: body }],
            id: "temp-id", // Adding required id property
          },
        ]);

        // Ensure ptr[0] uses the relative path provided by the LLM initially
        node.ptr[0] = relativePath;

        return {
          ...node,
          tok,
          tags: node.tags ?? [], // Ensure tags array exists
          // hash will be added in the indexing workflow
        };
      })
      .filter((node): node is GraphNode => node !== null); // Filter out invalid nodes

    // Post-process edges to add default 'fenced' property
    const processedEdges: Array<GraphEdge> = parsedResult.edges
      .map((edge) => {
        // Basic validation
        if (!edge.src || !edge.dst || !edge.why) {
          log(
            `LLM Parser: Invalid edge structure received in ${relativePath}: src=${edge.src}, dst=${edge.dst}`,
          );
          return null; // Skip invalid edges
        }
        return {
          ...edge,
          fenced: false,
        };
      })
      .filter((edge): edge is GraphEdge => edge !== null); // Filter out invalid edges

    log(
      `LLM Parser: Successfully parsed ${relativePath}. Found ${processedNodes.length} nodes and ${processedEdges.length} edges.`,
    );
    return { nodes: processedNodes, edges: processedEdges };
  } catch (parseError) {
    log(
      `LLM Parser: Error parsing JSON response for ${relativePath}: ${parseError}`,
    );
    return { nodes: [], edges: [] };
  }
}