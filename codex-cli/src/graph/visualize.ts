import type { GraphNode, GraphEdge } from "./types";

import { log } from "../utils/logger/log";
import { getGraphDir, loadNodes, loadEdges } from "./storage";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url"; // Import necessary modules for ESM __dirname equivalent
import { dirname } from "path"; // Import necessary modules for ESM __dirname equivalent

// Define the structure vis-network expects
interface VisNode {
  id: string;
  label: string;
  title?: string; // Tooltip
  group?: string; // Group by file path or type (e.g., 'external')
  color?: any; // Optional: for specific node styling
  shape?: string; // Optional: for specific node styling
}

interface VisEdge {
  from: string;
  to: string;
  title?: string; // Tooltip (use 'why')
  arrows?: string; // Default arrows
}

// ESM equivalent for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Generates a static HTML file visualizing the codebase graph.
 *
 * @param projectRoot - The absolute path to the project root.
 * @param outputHtmlPath - The absolute path where the HTML file should be saved.
 */
export async function generateGraphVisualization(
  projectRoot: string,
  outputHtmlPath: string,
): Promise<void> {
  log(`Generating graph visualization for project: ${projectRoot}`);
  const graphDir = await getGraphDir(projectRoot);

  // Load graph data
  log("Loading graph data...");
  const nodes = await loadNodes(graphDir);
  const edges = await loadEdges(graphDir);

  if (nodes.length === 0) {
    log("No graph data found. Cannot generate visualization.");
    // eslint-disable-next-line no-console
    console.error("Error: No graph data found. Run 'codex index' first.");
    return;
  }
  log(`Loaded ${nodes.length} nodes and ${edges.length} edges.`);

  // Transform data for vis-network
  log("Transforming data for visualization...");
  const visNodes: VisNode[] = nodes.map((node) => {
    // Extract function/method name for label, fallback to full ID
    const label = node.id.includes(":")
      ? node.id.substring(node.id.lastIndexOf(":") + 1)
      : node.id;
    // Construct tooltip including tags if available
    let title = `ID: ${node.id}\nPath: ${node.ptr[0]}\nLines: ${
      node.ptr[1]
    }-${node.ptr[2]}\nTokens: ${node.tok}\nSignature: ${
      node.sig
    }\nDoc: ${node.doc}\nSummary: ${node.summary}`;
    if (node.tags && node.tags.length > 0) {
        title += `\nTags: [${node.tags.join(', ')}]`;
    }
    const group = node.ptr[0]; // Group nodes by file path initially

    return {
      id: node.id,
      label: label,
      title: title,
      group: group,
    };
  });

  // Assign external nodes to a specific group
  visNodes.forEach(visNode => {
      const originalNode = nodes.find(n => n.id === visNode.id);
      if (originalNode && (originalNode.ptr[0] === "external" || originalNode.tags?.includes("external"))) {
          visNode.group = "external"; // Assign to 'external' group
          // Optionally adjust shape/color directly if not using group styling extensively
          // visNode.color = { background: '#cccccc', border: '#aaaaaa' };
          // visNode.shape = 'box';
      }
  });


  const visEdges: VisEdge[] = edges
    .filter(
      (edge) =>
        nodes.some((n) => n.id === edge.src) &&
        nodes.some((n) => n.id === edge.dst),
    ) // Ensure both source and destination nodes exist
    .map((edge) => ({
      from: edge.src,
      to: edge.dst,
      title: edge.why, // Use 'why' for tooltip
      arrows: "to",
    }));

  // Load HTML template
  log("Loading HTML template...");
  // Resolve template path relative to the *built* script's directory
  const templatePath = path.resolve(
    __dirname, // This will be 'dist/' or similar when running the built code
    "../src/graph/visualization/template.html", // Adjust path relative to dist/
  );
  let templateContent: string;
  try {
    templateContent = await fs.readFile(templatePath, "utf-8");
  } catch (error) {
    log(`Error loading template file ${templatePath}: ${error}`);
    throw new Error(`Could not load visualization template: ${templatePath}`);
  }

  // Inject data into template
  log("Injecting graph data into template...");
  const nodesJson = JSON.stringify(visNodes);
  const edgesJson = JSON.stringify(visEdges);

  // Define vis-network options, including group styling
  const options = {
      nodes: {
          shape: 'dot',
          size: 10,
          font: {
              size: 12,
              face: 'Tahoma',
              color: '#333'
          },
          borderWidth: 1,
          shadow: true
      },
      edges: {
          width: 1,
          shadow: true,
          color: { inherit: 'from' },
          smooth: {
              type: 'continuous'
          }
      },
      physics: {
          enabled: true,
          barnesHut: {
              gravitationalConstant: -2000,
              centralGravity: 0.3,
              springLength: 95,
              springConstant: 0.04,
              damping: 0.09,
              avoidOverlap: 0
          },
          maxVelocity: 50,
          minVelocity: 0.1,
          solver: 'barnesHut',
          stabilization: {
              enabled: true,
              iterations: 1000,
              updateInterval: 25,
              onlyDynamicEdges: false,
              fit: true
          },
          timestep: 0.5,
          adaptiveTimestep: true
      },
      interaction: {
          dragNodes: true,
          dragView: true,
          hideEdgesOnDrag: false,
          hideNodesOnDrag: false,
          zoomView: true,
          navigationButtons: true, // Add navigation buttons
          keyboard: true // Enable keyboard navigation
      },
      groups: {
          external: { // Define styling for the 'external' group
              color: { background: '#f0f0f0', border: '#cccccc' },
              shape: 'ellipse', // Or 'box', 'database', etc.
              font: { color: '#888888', size: 10 },
              borderWidth: 1,
              // You can add more specific styling here
          }
          // Add other group styles if needed
      }
  };
  const optionsJson = JSON.stringify(options, null, 2); // Pretty print options JSON

  const finalHtml = templateContent
    .replace("/* NODES_JSON_PLACEHOLDER */", nodesJson)
    .replace("/* EDGES_JSON_PLACEHOLDER */", edgesJson)
    .replace("/* OPTIONS_JSON_PLACEHOLDER */", optionsJson); // Inject options

  // Write output file
  log(`Writing visualization to: ${outputHtmlPath}`);
  try {
    await fs.writeFile(outputHtmlPath, finalHtml, "utf-8");
    // eslint-disable-next-line no-console
    console.log(`Graph visualization saved to: ${outputHtmlPath}`);
  } catch (error) {
    log(`Error writing HTML file ${outputHtmlPath}: ${error}`);
    throw new Error(`Failed to save visualization HTML: ${outputHtmlPath}`);
  }

  log("Graph visualization generation complete.");
}