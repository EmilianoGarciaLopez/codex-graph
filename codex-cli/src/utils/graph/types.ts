// src/utils/graph/types.ts

/**
 * Represents a relationship link in the graph, including a summary.
 */
export interface GraphEdgeInfo {
  /** Absolute path of the related file (target for dependency, source for dependent). */
  id: string;
  /** Concise summary of how the files are related. */
  summary: string;
}

/**
 * Represents a node in the dependency graph, typically a file.
 */
export interface GraphNode {
  /** Absolute path of the file. */
  id: string;
  /** List of files this node depends on, with relationship summaries. */
  dependencies: Array<GraphEdgeInfo>;
  /** List of files that depend on this node, with relationship summaries. */
  dependents: Array<GraphEdgeInfo>;
}

/**
 * Represents the entire dependency graph data structure.
 */
export interface GraphData {
  nodes: { [id: string]: GraphNode };
  metadata: {
    projectRoot?: string;
    buildTimestamp?: number;
    // Add other metadata as needed
  };
}

/**
 * Represents a directed edge with summary identified during analysis.
 */
export interface AnalyzedEdge {
  /** Absolute path of the file with the dependency (source). */
  source: string;
  /** Absolute path of the file being depended upon (target). */
  target: string;
  /** Type of dependency identified by the LLM. */
  type: "import" | "usage" | "related" | "dependent"; // 'dependent' used in single file analysis
  /** Concise summary of the relationship. */
  summary: string;
}

/**
 * Represents file contents with absolute path.
 * Re-exported here for convenience within the graph module.
 */
export interface FileContent {
  path: string;
  content: string;
}

/**
 * Represents a file change event used for graph updates.
 */
export type ChangedFile = {
    path: string;
    changeType: "create" | "update" | "delete";
};

/**
 * Represents the output structure for the getRelatedContext tool.
 */
export interface RelatedContextResult {
    relatedFilename: string;
    relationshipSummary: string;
}