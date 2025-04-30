import type { GraphNode, GraphEdge } from "./types";

import { log } from "../utils/logger/log";
import fs from "fs/promises";
import path from "path";

const GRAPH_DIR_NAME = ".codex/graph";
const NODES_FILE = "nodes.json";
const EDGES_FILE = "edges.json";
const HASHES_FILE = "hashes.json"; // Store content hashes separately

/**
 * Returns the absolute path to the graph storage directory for a given project root.
 * Ensures the directory exists.
 * @param projectRoot - The root directory of the project.
 * @returns The absolute path to the graph storage directory.
 */
export async function getGraphDir(projectRoot: string): Promise<string> {
  const graphDir = path.resolve(projectRoot, GRAPH_DIR_NAME);
  try {
    await fs.mkdir(graphDir, { recursive: true });
  } catch (error) {
    log(`Error creating graph directory ${graphDir}: ${error}`);
    throw new Error(`Failed to create graph directory: ${graphDir}`);
  }
  return graphDir;
}

/**
 * Loads graph nodes from nodes.json.
 * @param graphDir - The absolute path to the graph storage directory.
 * @returns An array of GraphNode objects.
 */
export async function loadNodes(graphDir: string): Promise<GraphNode[]> {
  const filePath = path.join(graphDir, NODES_FILE);
  try {
    const data = await fs.readFile(filePath, "utf-8");
    // Ensure nodes have the optional 'tags' field initialized if missing
    const nodes = JSON.parse(data) as Omit<GraphNode, 'hash'>[];
    return nodes.map(node => ({ ...node, tags: node.tags ?? [] }));
  } catch (error) {
    // If file not found or JSON parse error, return empty array
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((error as any).code === "ENOENT") {
      return [];
    }
    log(`Error loading nodes from ${filePath}: ${error}`);
    return []; // Return empty on other errors too, treat as non-existent
  }
}

/**
 * Loads graph edges from edges.json.
 * @param graphDir - The absolute path to the graph storage directory.
 * @returns An array of GraphEdge objects.
 */
export async function loadEdges(graphDir: string): Promise<GraphEdge[]> {
  const filePath = path.join(graphDir, EDGES_FILE);
  try {
    const data = await fs.readFile(filePath, "utf-8");
    return JSON.parse(data) as GraphEdge[];
  } catch (error) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((error as any).code === "ENOENT") {
      return [];
    }
    log(`Error loading edges from ${filePath}: ${error}`);
    return [];
  }
}

/**
 * Loads content hashes from hashes.json.
 * @param graphDir - The absolute path to the graph storage directory.
 * @returns A record mapping node IDs to content hashes.
 */
export async function loadHashes(
  graphDir: string,
): Promise<Record<string, string>> {
  const filePath = path.join(graphDir, HASHES_FILE);
  try {
    const data = await fs.readFile(filePath, "utf-8");
    return JSON.parse(data) as Record<string, string>;
  } catch (error) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((error as any).code === "ENOENT") {
      return {};
    }
    log(`Error loading hashes from ${filePath}: ${error}`);
    return {};
  }
}

/**
 * Saves graph nodes to nodes.json.
 * @param graphDir - The absolute path to the graph storage directory.
 * @param nodes - The array of GraphNode objects to save.
 */
export async function saveNodes(
  graphDir: string,
  nodes: GraphNode[],
): Promise<void> {
  const filePath = path.join(graphDir, NODES_FILE);
  try {
    // Exclude hash from saved nodes.json, keep it separate
    const nodesToSave = nodes.map(({ hash: _hash, ...rest }) => rest);
    await fs.writeFile(filePath, JSON.stringify(nodesToSave, null, 2), "utf-8");
  } catch (error) {
    log(`Error saving nodes to ${filePath}: ${error}`);
    throw new Error(`Failed to save nodes: ${filePath}`);
  }
}

/**
 * Saves graph edges to edges.json.
 * @param graphDir - The absolute path to the graph storage directory.
 * @param edges - The array of GraphEdge objects to save.
 */
export async function saveEdges(
  graphDir: string,
  edges: GraphEdge[],
): Promise<void> {
  const filePath = path.join(graphDir, EDGES_FILE);
  try {
    await fs.writeFile(filePath, JSON.stringify(edges, null, 2), "utf-8");
  } catch (error) {
    log(`Error saving edges to ${filePath}: ${error}`);
    throw new Error(`Failed to save edges: ${filePath}`);
  }
}

/**
 * Saves content hashes to hashes.json.
 * @param graphDir - The absolute path to the graph storage directory.
 * @param hashes - A record mapping node IDs to content hashes.
 */
export async function saveHashes(
  graphDir: string,
  hashes: Record<string, string>,
): Promise<void> {
  const filePath = path.join(graphDir, HASHES_FILE);
  try {
    await fs.writeFile(filePath, JSON.stringify(hashes, null, 2), "utf-8");
  } catch (error) {
    log(`Error saving hashes to ${filePath}: ${error}`);
    throw new Error(`Failed to save hashes: ${filePath}`);
  }
}