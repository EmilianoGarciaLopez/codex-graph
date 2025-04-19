// src/utils/graph/storage.ts
import type { GraphData } from "./types";

import { log } from "../agent/log";
import * as fsSync from "fs";
import * as fs from "fs/promises";
import * as path from "path";

/**
 * Loads the dependency graph from a JSON file.
 * @param graphPath Absolute path to the graph JSON file.
 * @returns The loaded GraphData object, or null if the file doesn't exist or is invalid.
 */
export async function loadGraph(
  graphPath: string,
): Promise<GraphData | null> {
  try {
    if (!fsSync.existsSync(graphPath)) {
      log(`Graph file not found at ${graphPath}`);
      return null;
    }
    const rawData = await fs.readFile(graphPath, "utf-8");
    const graphData = JSON.parse(rawData) as GraphData;
    // Basic validation
    if (
      typeof graphData !== "object" ||
      graphData == null || // Use loose equality
      typeof graphData.nodes !== "object"
    ) {
      log(`Invalid graph data format in ${graphPath}`);
      return null;
    }
    log(`Graph loaded successfully from ${graphPath}`);
    return graphData;
  } catch (error) {
    log(`Error loading graph from ${graphPath}: ${error}`);
    return null;
  }
}

/**
 * Saves the dependency graph data to a JSON file.
 * @param graphPath Absolute path to the graph JSON file.
 * @param graphData The graph data object to save.
 */
export async function saveGraph(
  graphPath: string,
  graphData: GraphData,
): Promise<void> {
  try {
    const dir = path.dirname(graphPath);
    // Ensure the directory exists
    if (!fsSync.existsSync(dir)) {
      await fs.mkdir(dir, { recursive: true });
    }
    // Update timestamp before saving
    graphData.metadata.buildTimestamp = Date.now();
    const jsonData = JSON.stringify(graphData, null, 2);
    await fs.writeFile(graphPath, jsonData, "utf-8");
    log(`Graph saved successfully to ${graphPath}`);
  } catch (error) {
    log(`Error saving graph to ${graphPath}: ${error}`);
    // Decide if we should re-throw or handle silently
  }
}