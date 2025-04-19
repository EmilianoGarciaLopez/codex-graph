// src/utils/graph/types.ts
/**
 * Represents a node in the dependency graph, typically a file.
 */
export interface GraphNode {
    /** Absolute path of the file. */
    id: string;
    /** List of absolute file paths this node depends on. */
    dependencies: Array<string>;
    /** List of absolute file paths that depend on this node. */
    dependents: Array<string>;
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
     * Represents a directed edge in the dependency graph during analysis.
     */
    export interface DependencyEdge {
    /** Absolute path of the file with the dependency (source). */
    source: string;
    /** Absolute path of the file being depended upon (target). */
    target: string;
    /** Type of dependency identified by the LLM. */
    type: "import" | "usage" | "related" | "dependent";
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