/**
 * Represents a node in the codebase graph, typically a function or method.
 */
export interface GraphNode {
  /** Fully-qualified identifier (e.g., "path/to/file:functionName") */
  id: string;
  /** Pointer to the source code location [relativeFilePath, startLine, endLine] */
  ptr: [string, number, number];
  /** One-line signature of the function/method */
  sig: string;
  /** First line of the docstring, if available */
  doc: string;
  /** LLM-generated summary (up to 30 words) */
  summary: string;
  /** Approximate token count of the function body */
  tok: number;
  /** Optional hash of the function body content for change detection */
  hash?: string;
  /** Optional array of conceptual tags assigned by the LLM */
  tags?: string[];
}

/**
 * Represents an edge in the codebase graph, indicating a relationship (e.g., call, import)
 * between two nodes.
 */
export interface GraphEdge {
  /** ID of the source node */
  src: string;
  /** ID of the destination node */
  dst: string;
  /** LLM-generated explanation of the relationship (up to 50 words) */
  why: string;
  /** Optional tag describing the kind of relationship (e.g., "call", "import") */
  kind?: string;
  /** Flag to temporarily hide this edge during retrieval */
  fenced: boolean;
}