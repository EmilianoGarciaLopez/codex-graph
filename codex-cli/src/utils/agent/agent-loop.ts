// src/utils/agent/agent-loop.ts
import type { ReviewDecision } from "./review.js";
import type { ApplyPatchCommand, ApprovalPolicy } from "../../approvals.js";
import type { AppConfig } from "../config.js";
import type { GraphData, FileContent, ChangedFile } from "../graph/types"; // Import graph types
import type { ResponseEvent } from "../responses.js";
import type {
  ResponseFunctionToolCallItem,
  ResponseInputItem,
  ResponseItem,
  ResponseCreateParams,
  FunctionTool,
} from "openai/resources/responses/responses.mjs";
import type { Reasoning } from "openai/resources.mjs";


import { OPENAI_TIMEOUT_MS, getApiKey, getBaseUrl } from "../config.js";
import { log } from "../logger/log.js";
import { log, isLoggingEnabled } from "./log.js";
import { parseApplyPatch } from "../../parse-apply-patch"; // Import patch parser
import { OPENAI_BASE_URL, OPENAI_TIMEOUT_MS } from "../config.js";
import { parseToolCallArguments } from "../parsers.js";
import { responsesCreateViaChatCompletions } from "../responses.js";
import {
  ORIGIN,
  CLI_VERSION,
  getSessionId,
  setCurrentModel,
  setSessionId,
} from "../session.js";
import { handleExecCommand } from "./handle-exec-command.js";
// import { generateGraphMarkdown } from "../graph/annotator"; // Removed unused import
import { getRelatedContext } from "../graph/retriever"; // Import graph retriever
import { loadGraph, saveGraph } from "../graph/storage"; // Import graph storage
import { updateGraphForChanges } from "../graph/updater"; // Import graph updater
import { getFileContents, loadIgnorePatterns } from "../singlepass/context_files"; // Import file utils
import { randomUUID } from "node:crypto";
import OpenAI, { APIConnectionTimeoutError } from "openai";
import path from "path"; // Import path

// Wait time before retrying after rate limit errors (ms).
const RATE_LIMIT_RETRY_WAIT_MS = parseInt(
  process.env["OPENAI_RATE_LIMIT_RETRY_WAIT_MS"] || "2500",
  10,
);

// Max tokens for related context injection (currently unused by getRelatedContext)
// const MAX_GRAPH_CONTEXT_TOKENS = 10000; // Adjust as needed

export type CommandConfirmation = {
  review: ReviewDecision;
  applyPatch?: ApplyPatchCommand | undefined;
  customDenyMessage?: string;
  explanation?: string;
};

const alreadyProcessedResponses = new Set();

type AgentLoopParams = {
  model: string;
  provider?: string;
  config?: AppConfig;
  instructions?: string;
  approvalPolicy: ApprovalPolicy;
  /**
   * Whether the model responses should be stored on the server side (allows
   * using `previous_response_id` to provide conversational context). Defaults
   * to `true` to preserve the current behaviour. When set to `false` the agent
   * will instead send the *full* conversation context as the `input` payload
   * on every request and omit the `previous_response_id` parameter.
   */
  disableResponseStorage?: boolean;
  onItem: (item: ResponseItem) => void;
  onLoading: (loading: boolean) => void;

  /** Extra writable roots to use with sandbox execution. */
  additionalWritableRoots: ReadonlyArray<string>;

  /** Called when the command is not auto-approved to request explicit user review. */
  getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  onLastResponseId: (lastResponseId: string) => void;
};

const shellTool: FunctionTool = {
  type: "function",
  name: "shell",
  description: "Runs a shell command, and returns its output.",
  strict: false,
  parameters: {
    type: "object",
    properties: {
      command: { type: "array", items: { type: "string" } },
      workdir: {
        type: "string",
        description: "The working directory for the command.",
      },
      timeout: {
        type: "number",
        description:
          "The maximum time to wait for the command to complete in milliseconds.",
      },
    },
    required: ["command"],
    additionalProperties: false,
  },
};

export class AgentLoop {
  private model: string;
  private provider: string;
  private instructions?: string;
  private approvalPolicy: ApprovalPolicy;
  private config: AppConfig;
  private additionalWritableRoots: ReadonlyArray<string>;

  // Graph related properties
  private graphMode: boolean = false;
  private graphData: GraphData | null = null;
  // private graphAnnotation: string | null = null; // Removed unused variable
  private graphPath: string = "";
  private allFilePaths: Array<string> = []; // Store all project file paths for updater

  private oai: OpenAI;

  private onItem: (item: ResponseItem) => void;
  private onLoading: (loading: boolean) => void;
  private getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  private onLastResponseId: (lastResponseId: string) => void;

  private currentStream: unknown | null = null;
  private generation = 0;
  private execAbortController: AbortController | null = null;
  private canceled = false;

  /**
   * Local conversation transcript used when `disableResponseStorage === true`. Holds
   * all non‑system items exchanged so far so we can provide full context on
   * every request.
   */
  private transcript: Array<ResponseInputItem> = [];
  /** Function calls that were emitted by the model but never answered because
   *  the user cancelled the run.  We keep the `call_id`s around so the *next*
   *  request can send a dummy `function_call_output` that satisfies the
   *  contract and prevents the
   *    400 | No tool output found for function call …
   *  error from OpenAI. */
  private pendingAborts: Set<string> = new Set();
  private terminated = false;
  private readonly hardAbort = new AbortController();

  public cancel(): void {
    if (this.terminated) {
      return;
    }
    this.currentStream = null;
    log(
      `AgentLoop.cancel() invoked – currentStream=${Boolean(
        this.currentStream,
      )} execAbortController=${Boolean(this.execAbortController)} generation=${
        this.generation
      }`,
    );
    (
      this.currentStream as { controller?: { abort?: () => void } } | null
    )?.controller?.abort?.();

    this.canceled = true;
    this.execAbortController?.abort();
    this.execAbortController = new AbortController();
    if (isLoggingEnabled()) {
      log("AgentLoop.cancel(): execAbortController.abort() called");
    }

    // NOTE: We intentionally do *not* clear `lastResponseId` here.  If the
    // stream produced a `function_call` before the user cancelled, OpenAI now
    // expects a corresponding `function_call_output` that must reference that
    // very same response ID.  We therefore keep the ID around so the
    // follow‑up request can still satisfy the contract.

    // If we have *not* seen any function_call IDs yet there is nothing that
    // needs to be satisfied in a follow‑up request.  In that case we clear
    // the stored lastResponseId so a subsequent run starts a clean turn.
    if (this.pendingAborts.size === 0) {
      try {
        this.onLastResponseId("");
      } catch {
        /* ignore */
      }
    }
    this.onLoading(false);
    this.generation += 1;
    log(`AgentLoop.cancel(): generation bumped to ${this.generation}`);
  }

  public terminate(): void {
    if (this.terminated) {
      return;
    }
    this.terminated = true;
    this.hardAbort.abort();
    this.cancel();
  }

  public sessionId: string;

  constructor({
    model,
    provider = "openai",
    instructions,
    approvalPolicy,
    disableResponseStorage,
    // `config` used to be required.  Some unit‑tests (and potentially other
    // callers) instantiate `AgentLoop` without passing it, so we make it
    // optional and fall back to sensible defaults.  This keeps the public
    // surface backwards‑compatible and prevents runtime errors like
    // "Cannot read properties of undefined (reading 'apiKey')" when accessing
    // `config.apiKey` below.
    config,
    onItem,
    onLoading,
    getCommandConfirmation,
    onLastResponseId,
    additionalWritableRoots,
  }: AgentLoopParams) {
    this.model = model;
    this.provider = provider;
    this.instructions = instructions;
    this.approvalPolicy = approvalPolicy;

    // If no `config` has been provided we derive a minimal stub so that the
    // rest of the implementation can rely on `this.config` always being a
    // defined object.  We purposefully copy over the `model` and
    // `instructions` that have already been passed explicitly so that
    // downstream consumers (e.g. telemetry) still observe the correct values.
    this.config = config ?? {
      model,
      instructions: instructions ?? "",
    };
    this.additionalWritableRoots = additionalWritableRoots;
    this.onItem = onItem;
    this.onLoading = onLoading;
    this.getCommandConfirmation = getCommandConfirmation;
    this.onLastResponseId = onLastResponseId;

    this.disableResponseStorage = disableResponseStorage ?? false;
    this.sessionId = getSessionId() || randomUUID().replaceAll("-", "");

    // --- Graph Mode Initialization ---
    this.graphMode = config.graphMode ?? false;
    if (this.graphMode) {
      const projectRoot = process.cwd();
      this.graphPath = path.join(projectRoot, ".codex", "dependency_graph.json");
      this.initializeGraphData().catch((err) => {
        log(`Error initializing graph data: ${err}`);
        // Optionally disable graph mode if init fails
        // this.graphMode = false;
      });
    }
    // --- End Graph Mode Initialization ---

    const timeoutMs = OPENAI_TIMEOUT_MS;
    const apiKey = getApiKey(this.provider);
    const baseURL = getBaseUrl(this.provider);

    this.oai = new OpenAI({
      ...(apiKey ? { apiKey } : {}),
      baseURL,
      defaultHeaders: {
        originator: ORIGIN,
        version: CLI_VERSION,
        session_id: this.sessionId,
      },
      ...(timeoutMs !== undefined ? { timeout: timeoutMs } : {}),
    });

    setSessionId(this.sessionId);
    setCurrentModel(this.model);

    this.hardAbort.signal.addEventListener(
      "abort",
      () => this.execAbortController?.abort(),
      { once: true },
    );
  }

  // --- Graph Initialization Helper ---
  private async initializeGraphData(): Promise<void> {
    log("Initializing graph data...");
    this.graphData = await loadGraph(this.graphPath);
    if (this.graphData) {
      // this.graphAnnotation = generateGraphMarkdown(this.graphData); // Keep generating annotation if needed elsewhere
      log("Graph loaded.");
      // Load all file paths for the updater
      const ignorePatterns = loadIgnorePatterns();
      const allFiles = await getProjectFiles(
        process.cwd(),
        ignorePatterns,
      );
      this.allFilePaths = allFiles.map((f) => f.path);
      log(`Loaded ${this.allFilePaths.length} file paths for graph updates.`);
    } else {
      log("Graph data not found or invalid.");
      // Graph build should have happened in cli.tsx if file didn't exist
    }
  }
  // --- End Graph Initialization Helper ---

  // --- New getRelatedContext Tool Implementation ---
  private async toolGetRelatedContext(args: { filename: string }): Promise<Array<RelatedContextResult>> {
      if (!this.graphMode || !this.graphData) {
          log("Graph mode disabled or graph not loaded. Cannot get related context.");
          throw new Error("Graph mode is not enabled or graph data is unavailable.");
      }
      if (!args || typeof args.filename !== 'string') {
          throw new Error("Invalid arguments: filename is required and must be a string.");
      }
      log(`Tool call: getRelatedContext for filename: ${args.filename}`);
      // Pass only the single filename to the retriever
      return getRelatedContext([args.filename], this.graphData);
  }
  // --- End New getRelatedContext Tool Implementation ---


  private async handleFunctionCall(
    item: ResponseFunctionToolCallItem,
    _currentItems: Array<ResponseItem>, // Keep receiving current items, though not used here
  ): Promise<Array<ResponseInputItem>> {
    if (this.canceled) {
      return [];
    }

    // Access name and arguments directly from the item
    const name: string | undefined = item.name;
    const rawArguments: string | undefined = item.arguments;
    const callId: string = (item as { call_id?: string }).call_id ?? (item as { id: string }).id;

    const args = parseToolCallArguments(rawArguments ?? "{}");
    log(
      `handleFunctionCall(): name=${
        name ?? "undefined"
      } callId=${callId} args=${rawArguments}`,
    );

    // Default output item in case of errors or unknown function
    const outputItem: ResponseInputItem.FunctionCallOutput = {
      type: "function_call_output",
      call_id: callId,
      output: "no function found", // Default message
    };
    const additionalItems: Array<ResponseInputItem> = [];

    try {
        // --- Tool Dispatching ---
        if (name === "getRelatedContext") {
            const args = JSON.parse(rawArguments ?? "{}") as { filename: string };
            const relatedContextResult = await this.toolGetRelatedContext(args);
            // Wrap the actual result in the shell output format
            outputItem.output = JSON.stringify({
                output: JSON.stringify(relatedContextResult), // Stringify the actual result array
                metadata: { exit_code: 0, duration_seconds: 0 } // Provide dummy metadata
            });
        } else if (name === "container.exec" || name === "shell") {
            const args = parseToolCallArguments(rawArguments ?? "{}");
            if (args == null) {
                outputItem.output = `invalid arguments for ${name}: ${rawArguments}`;
            } else {
                const {
                    outputText,
                    metadata,
                    additionalItems: additionalItemsFromExec,
                } = await handleExecCommand(
                    args,
                    this.config,
                    this.approvalPolicy,
                    this.additionalWritableRoots,
                    this.getCommandConfirmation,
                    this.execAbortController?.signal,
                );
                // Shell commands already return the correct format
                outputItem.output = JSON.stringify({ output: outputText, metadata });

                if (additionalItemsFromExec) {
                    additionalItems.push(...additionalItemsFromExec);
                }

                // --- Graph Update Logic (for shell/exec) ---
                const isApplyPatch = args.cmd[0] === "apply_patch";
                if (
                    isApplyPatch &&
                    this.graphMode &&
                    this.graphData &&
                    metadata['exit_code'] === 0 // Use bracket notation
                ) {
                    const patchText = args.cmd[1];
                    if (patchText) {
                        const parsedOps = parseApplyPatch(patchText);
                        if (parsedOps) {
                            const changes: Array<ChangedFile> = parsedOps.map((op) => ({
                                path: op.path,
                                changeType: op.type as "create" | "update" | "delete", // Cast type
                            }));
                            log(`Triggering graph update for ${changes.length} changes from apply_patch.`);
                            try {
                                // eslint-disable-next-line no-await-in-loop
                                const updatedGraph = await updateGraphForChanges(
                                    changes,
                                    this.graphData,
                                    this.model,
                                    this.config.apiKey ?? "",
                                    this.allFilePaths,
                                );
                                this.graphData = updatedGraph;
                                // this.graphAnnotation = generateGraphMarkdown(this.graphData); // Regenerate if needed
                                // eslint-disable-next-line no-await-in-loop
                                await saveGraph(this.graphPath, this.graphData);
                                log("Graph updated and saved successfully after apply_patch.");
                            } catch (error) {
                                log(`Error updating graph after apply_patch: ${error}`);
                            }
                        } else {
                            log("Could not parse patch text for graph update.");
                        }
                    } else {
                        log("Patch text not found in apply_patch command for graph update.");
                    }
                }
                // --- End Graph Update Logic ---
            }
        } else {
            log(`Unknown function call name: ${name}`);
            // Wrap unknown function output as well for safety, though ideally model shouldn't call unknown functions
             outputItem.output = JSON.stringify({
                output: `Unknown function name: ${name}`,
                metadata: { exit_code: 1, duration_seconds: 0 }
            });
        }
        // --- End Tool Dispatching ---

    } catch (error) {
        log(`Error handling function call ${name}: ${error}`);
        // Format error output consistently
        outputItem.output = JSON.stringify({
            output: `Error executing tool ${name}: ${error instanceof Error ? error.message : String(error)}`,
            metadata: { exit_code: 1, duration_seconds: 0 }
        });
    }

    return [outputItem, ...additionalItems];
  }


  public async run(
    input: Array<ResponseInputItem>,
    previousResponseId: string = "",
    currentItems: Array<ResponseItem> = [], // Receive current items
  ): Promise<void> {
    try {
      if (this.terminated) {
        throw new Error("AgentLoop has been terminated");
      }
      const thinkingStart = Date.now();
      const thisGeneration = ++this.generation;

      this.canceled = false;
      this.currentStream = null;
      this.execAbortController = new AbortController();
      log(
        `AgentLoop.run(): new execAbortController created (${this.execAbortController.signal}) for generation ${this.generation}`,
      );
      // NOTE: We no longer (re‑)attach an `abort` listener to `hardAbort` here.
      // A single listener that forwards the `abort` to the current
      // `execAbortController` is installed once in the constructor. Re‑adding a
      // new listener on every `run()` caused the same `AbortSignal` instance to
      // accumulate listeners which in turn triggered Node's
      // `MaxListenersExceededWarning` after ten invocations.

      // Track the response ID from the last *stored* response so we can use
      // `previous_response_id` when `disableResponseStorage` is enabled.  When storage
      // is disabled we deliberately ignore the caller‑supplied value because
      // the backend will not retain any state that could be referenced.
      // If the backend stores conversation state (`disableResponseStorage === false`) we
      // forward the caller‑supplied `previousResponseId` so that the model sees the
      // full context.  When storage is disabled we *must not* send any ID because the
      // server no longer retains the referenced response.
      let lastResponseId: string = this.disableResponseStorage
        ? ""
        : previousResponseId;

      // If there are unresolved function calls from a previously cancelled run
      // we have to emit dummy tool outputs so that the API no longer expects
      // them.  We prepend them to the user‑supplied input so they appear
      // first in the conversation turn.
      const abortOutputs: Array<ResponseInputItem> = [];
      if (this.pendingAborts.size > 0) {
        for (const id of this.pendingAborts) {
          abortOutputs.push({
            type: "function_call_output",
            call_id: id,
            output: JSON.stringify({
              output: "aborted",
              metadata: { exit_code: 1, duration_seconds: 0 },
            }),
          } as ResponseInputItem.FunctionCallOutput);
        }
        this.pendingAborts.clear();
      }

      // Build the input list for this turn. When responses are stored on the
      // server we can simply send the *delta* (the new user input as well as
      // any pending abort outputs) and rely on `previous_response_id` for
      // context.  When storage is disabled the server has no memory of the
      // conversation, so we must include the *entire* transcript (minus system
      // messages) on every call.

      let turnInput: Array<ResponseInputItem> = [];
      // Keeps track of how many items in `turnInput` stem from the existing
      // transcript so we can avoid re‑emitting them to the UI. Only used when
      // `disableResponseStorage === true`.
      let transcriptPrefixLen = 0;

      const stripInternalFields = (
        item: ResponseInputItem,
      ): ResponseInputItem => {
        // Clone shallowly and remove fields that are not part of the public
        // schema expected by the OpenAI Responses API.
        // We shallow‑clone the item so that subsequent mutations (deleting
        // internal fields) do not affect the original object which may still
        // be referenced elsewhere (e.g. UI components).
        const clean = { ...item } as Record<string, unknown>;
        delete clean["duration_ms"];
        // Remove OpenAI-assigned identifiers and transient status so the
        // backend does not reject items that were never persisted because we
        // use `store: false`.
        delete clean["id"];
        delete clean["status"];
        return clean as unknown as ResponseInputItem;
      };

      if (this.disableResponseStorage) {
        // Remember where the existing transcript ends – everything after this
        // index in the upcoming `turnInput` list will be *new* for this turn
        // and therefore needs to be surfaced to the UI.
        transcriptPrefixLen = this.transcript.length;

        // Ensure the transcript is up‑to‑date with the latest user input so
        // that subsequent iterations see a complete history.
        // `turnInput` is still empty at this point (it will be filled later).
        // We need to look at the *input* items the user just supplied.
        this.transcript.push(...filterToApiMessages(input));

        turnInput = [...this.transcript, ...abortOutputs].map(
          stripInternalFields,
        );
      } else {
        turnInput = [...abortOutputs, ...input].map(stripInternalFields);
      }

      this.onLoading(true);

      const staged: Array<ResponseItem | undefined> = [];
      const stageItem = (item: ResponseItem) => {
        if (thisGeneration !== this.generation) {
          return;
        }
        const idx = staged.push(item) - 1;
        setTimeout(() => {
          if (
            thisGeneration === this.generation &&
            !this.canceled &&
            !this.hardAbort.signal.aborted
          ) {
            this.onItem(item);
            staged[idx] = undefined;

            // When we operate without server‑side storage we keep our own
            // transcript so we can provide full context on subsequent calls.
            if (this.disableResponseStorage) {
              // Exclude system messages from transcript as they do not form
              // part of the assistant/user dialogue that the model needs.
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const role = (item as any).role;
              if (role !== "system") {
                // Clone the item to avoid mutating the object that is also
                // rendered in the UI. We need to strip auxiliary metadata
                // such as `duration_ms` which is not part of the Responses
                // API schema and therefore causes a 400 error when included
                // in subsequent requests whose context is sent verbatim.

                // Skip items that we have already inserted earlier or that the
                // model does not need to see again in the next turn.
                //   • function_call   – superseded by the forthcoming
                //     function_call_output.
                //   • reasoning       – internal only, never sent back.
                //   • user messages   – we added these to the transcript when
                //     building the first turnInput; stageItem would add a
                //     duplicate.
                if (
                  (item as ResponseInputItem).type === "function_call" ||
                  (item as ResponseInputItem).type === "reasoning" ||
                  ((item as ResponseInputItem).type === "message" &&
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    (item as any).role === "user")
                ) {
                  return;
                }

                const clone: ResponseInputItem = {
                  ...(item as unknown as ResponseInputItem),
                } as ResponseInputItem;
                // The `duration_ms` field is only added to reasoning items to
                // show elapsed time in the UI. It must not be forwarded back
                // to the server.
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                delete (clone as any).duration_ms;

                this.transcript.push(clone);
              }
            }
          }
        }, 10);
      };

      let currentTurnInput = turnInput; // Use a mutable variable for the loop

      while (currentTurnInput.length > 0) {
        if (this.canceled || this.hardAbort.signal.aborted) {
          this.onLoading(false);
          return;
        }
        // send request to openAI
        // Only surface the *new* input items to the UI – replaying the entire
        // transcript would duplicate messages that have already been shown in
        // earlier turns.
        // `turnInput` holds the *new* items that will be sent to the API in
        // this iteration.  Surface exactly these to the UI so that we do not
        // re‑emit messages from previous turns (which would duplicate user
        // prompts) and so that freshly generated `function_call_output`s are
        // shown immediately.
        // Figure out what subset of `turnInput` constitutes *new* information
        // for the UI so that we don’t spam the interface with repeats of the
        // entire transcript on every iteration when response storage is
        // disabled.
        const deltaInput = this.disableResponseStorage
          ? turnInput.slice(transcriptPrefixLen)
          : [...turnInput];
        for (const item of deltaInput) {
          stageItem(item as ResponseItem);
        }

        let stream;
        const MAX_RETRIES = 5;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
          try {
            let reasoning: Reasoning | undefined;
            if (this.model.startsWith("o")) {
              reasoning = { effort: "high" };
              if (this.model === "o3" || this.model === "o4-mini") {
                // @ts-expect-error waiting for API type update
                reasoning.summary = "auto";
              }
            }

            // --- Keep Graph Annotation Injection (Optional) ---
            // Decide if you still want the static graph overview in the system prompt
            const baseInstructions = this.instructions ?? ""; // Use const
            // Example: Conditionally add graph annotation if desired
            // if (this.graphMode && this.graphAnnotation) {
            //   baseInstructions = `${this.graphAnnotation}\n\n---\n\n${baseInstructions}`;
            //   log("Prepended graph annotation to instructions.");
            // }
            // --- End Graph Annotation Injection ---

            const mergedInstructions = [prefix, baseInstructions]
              .filter(Boolean)
              .join("\n");

            const responseCall =
              !this.config.provider ||
              this.config.provider?.toLowerCase() === "openai"
                ? (params: ResponseCreateParams) =>
                    this.oai.responses.create(params)
                : (params: ResponseCreateParams) =>
                    responsesCreateViaChatCompletions(
                      this.oai,
                      params as ResponseCreateParams & { stream: true },
                    );
            log(
              `instructions (length ${mergedInstructions.length}): ${mergedInstructions}`,
            );

            // eslint-disable-next-line no-await-in-loop
            stream = await responseCall({
              model: this.model,
              instructions: mergedInstructions,
              input: turnInput,
              stream: true,
              parallel_tool_calls: false,
              reasoning,
              ...(this.config.flexMode ? { service_tier: "flex" } : {}),
              ...(this.disableResponseStorage
                ? { store: false }
                : {
                    store: true,
                    previous_response_id: lastResponseId || undefined,
                  }),
              tools: [shellTool],
              // Explicitly tell the model it is allowed to pick whatever
              // tool it deems appropriate.  Omitting this sometimes leads to
              // the model ignoring the available tools and responding with
              // plain text instead (resulting in a missing tool‑call).
              tool_choice: "auto",
            });
            break; // Success, exit retry loop
          } catch (error) {
            // ... (existing error handling logic remains the same) ...
            const isTimeout = error instanceof APIConnectionTimeoutError;
            const ApiConnErrCtor = (OpenAI as typeof OpenAI).APIConnectionError as
              | (new (...args: Array<unknown>) => Error)
              | undefined;
            const isConnectionError = ApiConnErrCtor
              ? error instanceof ApiConnErrCtor
              : false;
            const errCtx = error as Record<string, unknown>;
            const status =
              errCtx['status'] ?? errCtx['httpStatus'] ?? errCtx['statusCode'];
            const isServerError =
              typeof status === "number" && status >= 500;
            if (
              (isTimeout || isServerError || isConnectionError) &&
              attempt < MAX_RETRIES
            ) {
              log(
                `OpenAI request failed (attempt ${attempt}/${MAX_RETRIES}), retrying...`,
              );
              // eslint-disable-next-line no-await-in-loop
              await new Promise((resolve) => setTimeout(resolve, 1000 * attempt)); // Simple backoff
              continue;
            }

            const isTooManyTokensError =
              (errCtx['param'] === "max_tokens" ||
                (typeof errCtx['message'] === "string" &&
                  /max_tokens is too large/i.test(errCtx['message']))) &&
              errCtx['type'] === "invalid_request_error";

            if (isTooManyTokensError) {
              this.onItem({
                id: `error-${Date.now()}`,
                type: "message",
                role: "system",
                content: [
                  {
                    type: "input_text",
                    text: "⚠️  The current request exceeds the maximum context length supported by the chosen model. Please shorten the conversation, run /clear, or switch to a model with a larger context window and try again.",
                  },
                ],
              });
              this.onLoading(false);
              return;
            }

            const isRateLimit =
              status === 429 ||
              errCtx['code'] === "rate_limit_exceeded" ||
              errCtx['type'] === "rate_limit_exceeded" ||
              /rate limit/i.test((errCtx['message'] as string) ?? "");
            if (isRateLimit) {
              if (attempt < MAX_RETRIES) {
                let delayMs = RATE_LIMIT_RETRY_WAIT_MS * 2 ** (attempt - 1);
                const msg = (errCtx['message'] as string) ?? "";
                const m = /(?:retry|try) again in ([\d.]+)s/i.exec(msg);
                if (m && m[1]) {
                  const suggested = parseFloat(m[1]) * 1000;
                  if (!Number.isNaN(suggested)) {
                    delayMs = suggested;
                  }
                }
                log(
                  `OpenAI rate limit exceeded (attempt ${attempt}/${MAX_RETRIES}), retrying in ${Math.round(
                    delayMs,
                  )} ms...`,
                );
                // eslint-disable-next-line no-await-in-loop
                await new Promise((resolve) => setTimeout(resolve, delayMs));
                continue;
              } else {
                const errorDetails = [
                  `Status: ${status || "unknown"}`,
                  `Code: ${errCtx['code'] || "unknown"}`,
                  `Type: ${errCtx['type'] || "unknown"}`,
                  `Message: ${errCtx['message'] || "unknown"}`,
                ].join(", ");
                this.onItem({
                  id: `error-${Date.now()}`,
                  type: "message",
                  role: "system",
                  content: [
                    {
                      type: "input_text",
                      text: `⚠️  Rate limit reached. Error details: ${errorDetails}. Please try again later.`,
                    },
                  ],
                });
                this.onLoading(false);
                return;
              }
            }

            const isClientError =
              (typeof status === "number" &&
                status >= 400 &&
                status < 500 &&
                status !== 429) ||
              errCtx['code'] === "invalid_request_error" ||
              errCtx['type'] === "invalid_request_error";
            if (isClientError) {
              this.onItem({
                id: `error-${Date.now()}`,
                type: "message",
                role: "system",
                content: [
                  {
                    type: "input_text",
                    text: (() => {
                      const reqId =
                        (
                          errCtx as Partial<{
                            request_id?: string;
                            requestId?: string;
                          }>
                        )?.request_id ??
                        (
                          errCtx as Partial<{
                            request_id?: string;
                            requestId?: string;
                          }>
                        )?.requestId;
                      const errorDetails = [
                        `Status: ${status || "unknown"}`,
                        `Code: ${errCtx['code'] || "unknown"}`,
                        `Type: ${errCtx['type'] || "unknown"}`,
                        `Message: ${errCtx['message'] || "unknown"}`,
                      ].join(", ");
                      return `⚠️  OpenAI rejected the request${
                        reqId ? ` (request ID: ${reqId})` : ""
                      }. Error details: ${errorDetails}. Please verify your settings and try again.`;
                    })(),
                  },
                ],
              });
              this.onLoading(false);
              return;
            }
            throw error; // Re-throw unhandled errors
          }
        }

        if (this.canceled || this.hardAbort.signal.aborted) {
          try {
            (
              stream as { controller?: { abort?: () => void } }
            )?.controller?.abort?.();
          } catch {
            /* ignore */
          }
          this.onLoading(false);
          return;
        }

        this.currentStream = stream;

        if (!stream) {
          this.onLoading(false);
          log("AgentLoop.run(): stream is undefined after retries");
          return;
        }

        const MAX_STREAM_RETRIES = 5;
        let streamRetryAttempt = 0;

        // eslint-disable-next-line no-constant-condition
        while (true) {
          try {
            let newTurnInput: Array<ResponseInputItem> = [];

            // eslint-disable-next-line no-await-in-loop
            for await (const event of stream as AsyncIterable<ResponseEvent>) {
              log(`AgentLoop.run(): response event ${event.type}`);

              // process and surface each item (no-op until we can depend on streaming events)
              if (event.type === "response.output_item.done") {
                const item = event.item;
                // 1) if it's a reasoning item, annotate it
                type ReasoningItem = { type?: string; duration_ms?: number };
                const maybeReasoning = item as ReasoningItem;
                if (maybeReasoning.type === "reasoning") {
                  maybeReasoning.duration_ms = Date.now() - thinkingStart;
                }
                if (item.type === "function_call") {
                  // Track outstanding tool call so we can abort later if needed.
                  // The item comes from the streaming response, therefore it has
                  // either `id` (chat) or `call_id` (responses) – we normalise
                  // by reading both.
                  const callId =
                    (item as { call_id?: string; id?: string }).call_id ??
                    (item as { id?: string }).id;
                  if (callId) {
                    this.pendingAborts.add(callId);
                  }
                } else {
                  stageItem(item as ResponseItem);
                }
              }

              if (event.type === "response.completed") {
                if (thisGeneration === this.generation && !this.canceled) {
                  for (const item of event.response.output) {
                    stageItem(item as ResponseItem);
                  }
                }
                if (
                  event.response.status === "completed" ||
                  (event.response.status as unknown as string) ===
                    "requires_action"
                ) {
                  // TODO: remove this once we can depend on streaming events
                  newTurnInput = await this.processEventsWithoutStreaming(
                    event.response.output,
                    stageItem,
                  );

                  // When we do not use server‑side storage we maintain our
                  // own transcript so that *future* turns still contain full
                  // conversational context. However, whether we advance to
                  // another loop iteration should depend solely on the
                  // presence of *new* input items (i.e. items that were not
                  // part of the previous request). Re‑sending the transcript
                  // by itself would create an infinite request loop because
                  // `turnInput.length` would never reach zero.

                  if (this.disableResponseStorage) {
                    // 1) Append the freshly emitted output to our local
                    //    transcript (minus non‑message items the model does
                    //    not need to see again).
                    const cleaned = filterToApiMessages(
                      event.response.output.map(stripInternalFields),
                    );
                    this.transcript.push(...cleaned);

                    // 2) Determine the *delta* (newTurnInput) that must be
                    //    sent in the next iteration. If there is none we can
                    //    safely terminate the loop – the transcript alone
                    //    does not constitute new information for the
                    //    assistant to act upon.

                    const delta = filterToApiMessages(
                      newTurnInput.map(stripInternalFields),
                    );

                    if (delta.length === 0) {
                      // No new input => end conversation.
                      newTurnInput = [];
                    } else {
                      // Re‑send full transcript *plus* the new delta so the
                      // stateless backend receives complete context.
                      newTurnInput = [...this.transcript, ...delta];
                      // The prefix ends at the current transcript length –
                      // everything after this index is new for the next
                      // iteration.
                      transcriptPrefixLen = this.transcript.length;
                    }
                  }
                }
                lastResponseId = event.response.id;
                this.onLastResponseId(event.response.id);
              }
            }

            // Set after we have consumed all stream events in case the stream wasn't
            // complete or we missed events for whatever reason. That way, we will set
            // the next turn to an empty array to prevent an infinite loop.
            // And don't update the turn input too early otherwise we won't have the
            // current turn inputs available for retries.
            turnInput = newTurnInput;

            // Stream finished successfully – leave the retry loop.
            break;
          } catch (err: unknown) {
            const isRateLimitError = (e: unknown): boolean => {
              if (!e || typeof e !== "object") {
                return false;
              }
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const ex: any = e;
              return (
                ex.status === 429 ||
                ex.code === "rate_limit_exceeded" ||
                ex.type === "rate_limit_exceeded"
              );
            };

            if (
              isRateLimitError(err) &&
              streamRetryAttempt < MAX_STREAM_RETRIES
            ) {
              streamRetryAttempt += 1;

              const waitMs =
                RATE_LIMIT_RETRY_WAIT_MS * 2 ** (streamRetryAttempt - 1);
              log(
                `OpenAI stream rate‑limited – retry ${streamRetryAttempt}/${MAX_STREAM_RETRIES} in ${waitMs} ms`,
              );

              // Give the server a breather before retrying.
              // eslint-disable-next-line no-await-in-loop
              await new Promise((res) => setTimeout(res, waitMs));

              // Re‑create the stream with the *same* parameters.
              let reasoning: Reasoning | undefined;
              if (this.model.startsWith("o")) {
                reasoning = { effort: "high" };
                if (this.model === "o3" || this.model === "o4-mini") {
                  reasoning.summary = "auto";
                }
              }

              const mergedInstructions = [prefix, this.instructions]
                .filter(Boolean)
                .join("\n");

              const responseCall =
                !this.config.provider ||
                this.config.provider?.toLowerCase() === "openai"
                  ? (params: ResponseCreateParams) =>
                      this.oai.responses.create(params)
                  : (params: ResponseCreateParams) =>
                      responsesCreateViaChatCompletions(
                        this.oai,
                        params as ResponseCreateParams & { stream: true },
                      );

              log(
                "agentLoop.run(): responseCall(1): turnInput: " +
                  JSON.stringify(turnInput),
              );
              // eslint-disable-next-line no-await-in-loop
              stream = await responseCall({
                model: this.model,
                instructions: mergedInstructions,
                input: turnInput,
                stream: true,
                parallel_tool_calls: false,
                reasoning,
                ...(this.config.flexMode ? { service_tier: "flex" } : {}),
                ...(this.disableResponseStorage
                  ? { store: false }
                  : {
                      store: true,
                      previous_response_id: lastResponseId || undefined,
                    }),
                tools: [shellTool],
                tool_choice: "auto",
              });

              this.currentStream = stream;
              // Continue to outer while to consume new stream.
              continue;
            }

            // Gracefully handle an abort triggered via `cancel()` so that the
            // consumer does not see an unhandled exception.
            if (err instanceof Error && err.name === "AbortError") {
              if (!this.canceled) {
                // It was aborted for some other reason; surface the error.
                throw err;
              }
              this.onLoading(false);
              return;
            }
            // Suppress internal stack on JSON parse failures
            if (err instanceof SyntaxError) {
              this.onItem({
                id: `error-${Date.now()}`,
                type: "message",
                role: "system",
                content: [
                  {
                    type: "input_text",
                    text: "⚠️ Failed to parse streaming response (invalid JSON). Please `/clear` to reset.",
                  },
                ],
              });
              this.onLoading(false);
              return;
            }
            // Handle OpenAI API quota errors
            if (
              err instanceof Error &&
              (err as { code?: string }).code === "insufficient_quota"
            ) {
              this.onItem({
                id: `error-${Date.now()}`,
                type: "message",
                role: "system",
                content: [
                  {
                    type: "input_text",
                    text: "⚠️ Insufficient quota. Please check your billing details and retry.",
                  },
                ],
              });
              this.onLoading(false);
              return;
            }
            throw err;
          } finally {
            this.currentStream = null;
          }
        } // end while retry loop

        log(
          `Next turn inputs (${currentTurnInput.length}) - ${currentTurnInput
            .map((i) => i.type)
            .join(", ")}`,
        );
      } // End while(currentTurnInput.length > 0)

      const flush = () => {
        if (
          !this.canceled &&
          !this.hardAbort.signal.aborted &&
          thisGeneration === this.generation
        ) {
          for (const item of staged) {
            if (item) {
              this.onItem(item);
            }
          }
        }
        this.pendingAborts.clear();
        this.onLoading(false);
      };

      setTimeout(flush, 30);
    } catch (err) {
      // ... (existing outer error handling logic) ...
      const isPrematureClose =
        err instanceof Error &&
        // eslint-disable-next-line
        ((err as unknown as Record<string, unknown>)['code'] === "ERR_STREAM_PREMATURE_CLOSE" ||
          err.message?.includes("Premature close"));

      if (isPrematureClose) {
        try {
          this.onItem({
            id: `error-${Date.now()}`,
            type: "message",
            role: "system",
            content: [
              {
                type: "input_text",
                text: "⚠️  Connection closed prematurely while waiting for the model. Please try again.",
              },
            ],
          });
        } catch {
          /* no-op – emitting the error message is best‑effort */
        }
        this.onLoading(false);
        return;
      }

      const NETWORK_ERRNOS = new Set([
        "ECONNRESET",
        "ECONNREFUSED",
        "EPIPE",
        "ENOTFOUND",
        "ETIMEDOUT",
        "EAI_AGAIN",
      ]);

      const isNetworkOrServerError = (() => {
        if (!err || typeof err !== "object") {
          return false;
        }
        const e = err as unknown as Record<string, unknown>;
        const ApiConnErrCtor = (OpenAI as typeof OpenAI).APIConnectionError as
          | (new (...args: Array<unknown>) => Error)
          | undefined;
        if (ApiConnErrCtor && e instanceof ApiConnErrCtor) {
          return true;
        }
        if (typeof e['code'] === "string" && NETWORK_ERRNOS.has(e['code'])) {
          return true;
        }
        if (
          e['cause'] &&
          typeof e['cause'] === "object" &&
          NETWORK_ERRNOS.has((e['cause'] as { code?: string }).code ?? "")
        ) {
          return true;
        }
        if (typeof e['status'] === "number" && e['status'] >= 500) {
          return true;
        }
        if (
          typeof e['message'] === "string" &&
          /network|socket|stream/i.test(e['message'])
        ) {
          return true;
        }
        return false;
      })();

      if (isNetworkOrServerError) {
        try {
          const msgText =
            "⚠️  Network error while contacting OpenAI. Please check your connection and try again.";
          this.onItem({
            id: `error-${Date.now()}`,
            type: "message",
            role: "system",
            content: [
              {
                type: "input_text",
                text: msgText,
              },
            ],
          });
        } catch {
          /* best‑effort */
        }
        this.onLoading(false);
        return;
      }

      const isInvalidRequestError = () => {
        if (!err || typeof err !== "object") {
          return false;
        }
        const e = err as unknown as Record<string, unknown>;
        if (
          e['type'] === "invalid_request_error" &&
          e['code'] === "model_not_found"
        ) {
          return true;
        }
        if (
          e['cause'] &&
          (e['cause'] as Record<string, unknown>)['type'] === "invalid_request_error" &&
          (e['cause'] as Record<string, unknown>)['code'] === "model_not_found"
        ) {
          return true;
        }
        return false;
      };

      if (isInvalidRequestError()) {
        try {
          const e = err as unknown as Record<string, unknown>;
          const reqId =
            e['request_id'] ??
            (e['cause'] && (e['cause'] as Record<string, unknown>)['request_id']) ??
            (e['cause'] && (e['cause'] as Record<string, unknown>)['requestId']);
          const errorDetails = [
            `Status: ${
              e['status'] || (e['cause'] && (e['cause'] as Record<string, unknown>)['status']) || "unknown"
            }`,
            `Code: ${e['code'] || (e['cause'] && (e['cause'] as Record<string, unknown>)['code']) || "unknown"}`,
            `Type: ${e['type'] || (e['cause'] && (e['cause'] as Record<string, unknown>)['type']) || "unknown"}`,
            `Message: ${
              e['message'] || (e['cause'] && (e['cause'] as Record<string, unknown>)['message']) || "unknown"
            }`,
          ].join(", ");
          const msgText = `⚠️  OpenAI rejected the request${
            reqId ? ` (request ID: ${reqId})` : ""
          }. Error details: ${errorDetails}. Please verify your settings and try again.`;
          this.onItem({
            id: `error-${Date.now()}`,
            type: "message",
            role: "system",
            content: [
              {
                type: "input_text",
                text: msgText,
              },
            ],
          });
        } catch {
          /* best-effort */
        }
        this.onLoading(false);
        return;
      }

      throw err; // Re-throw unhandled errors
    }
  }

  private async processEventsWithoutStreaming(
    output: Array<ResponseInputItem>,
    emitItem: (item: ResponseItem) => void,
    currentItems: Array<ResponseItem>, // Receive current items
  ): Promise<Array<ResponseInputItem>> {
    if (this.canceled) {
      return [];
    }
    const turnInput: Array<ResponseInputItem> = [];
    for (const item of output) {
      if (item.type === "function_call") {
        // Ensure item has an id before checking the set
        const itemId = (item as { id?: string }).id;
        if (itemId && alreadyProcessedResponses.has(itemId)) {
          continue;
        }
        if (itemId) {
            alreadyProcessedResponses.add(itemId);
        }
        // eslint-disable-next-line no-await-in-loop
        const result = await this.handleFunctionCall(item as ResponseFunctionToolCallItem, currentItems); // Pass current items
        turnInput.push(...result);
      }
      emitItem(item as ResponseItem);
    }
    return turnInput;
  }
}

// Removed unused helper function extractFilePaths

// Helper to get project files (adapt from singlepass or create new)
async function getProjectFiles(
  projectRoot: string,
  ignorePatterns: Array<RegExp>,
): Promise<Array<FileContent>> {
  // Implementation reused or adapted from src/utils/singlepass/context_files.ts
  return getFileContents(projectRoot, ignorePatterns);
}

const prefix = `You are operating as and within the Codex CLI, a terminal-based agentic coding assistant built by OpenAI. It wraps OpenAI models to enable natural language interaction with a local codebase. You are expected to be precise, safe, and helpful.

You can:
- Receive user prompts, project context, and files.
- Stream responses and emit function calls (e.g., shell commands, code edits, context retrieval).
- Apply patches, run commands, and manage user approvals based on policy.
- Work inside a sandboxed, git-backed workspace with rollback support.
- Log telemetry so sessions can be replayed or inspected later.
- More details on your functionality are available at \`codex --help\`

The Codex CLI is open-sourced. Don't confuse yourself with the old Codex language model built by OpenAI many moons ago (this is understandably top of mind for you!). Within this context, Codex refers to the open-source agentic coding interface.

You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. If you are not sure about file content or codebase structure pertaining to the user's request, use your tools (like \`getRelatedContext\` or \`shell\` with commands like \`cat\`, \`ls\`, \`grep\`) to read files and gather the relevant information: do NOT guess or make up an answer.

Please resolve the user's task by editing and testing the code files in your current code execution session. You are a deployed coding agent. Your session allows for you to modify and run code. The repo(s) are already cloned in your working directory, and you must fully solve the problem for your answer to be considered correct.

You MUST adhere to the following criteria when executing the task:
- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- User instructions may overwrite the *CODING GUIDELINES* section in this developer message.
- Use \`apply_patch\` via the \`shell\` tool to edit files: {"cmd":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file.py\\n@@ def example():\\n-  pass\\n+  return 123\\n*** End Patch"]}
- If completing the user's task requires writing or modifying files:
    - Your code and final answer should follow these *CODING GUIDELINES*:
        - Fix the problem at the root cause rather than applying surface-level patches, when possible.
        - Avoid unneeded complexity in your solution.
            - Ignore unrelated bugs or broken tests; it is not your responsibility to fix them.
        - Update documentation as necessary.
        - Keep changes consistent with the style of the existing codebase. Changes should be minimal and focused on the task.
            - Use \`git log\` and \`git blame\` to search the history of the codebase if additional context is required; internet access is disabled.
        - NEVER add copyright or license headers unless specifically requested.
        - You do not need to \`git commit\` your changes; this will be done automatically for you.
        - If there is a .pre-commit-config.yaml, use \`pre-commit run --files ...\` to check that your changes pass the pre-commit checks. However, do not fix pre-existing errors on lines you didn't touch.
            - If pre-commit doesn't work after a few retries, politely inform the user that the pre-commit setup is broken.
        - Once you finish coding, you must
            - Check \`git status\` to sanity check your changes; revert any scratch files or changes.
            - Remove all inline comments you added as much as possible, even if they look normal. Check using \`git diff\`. Inline comments must be generally avoided, unless active maintainers of the repo, after long careful study of the code and the issue, will still misinterpret the code without the comments.
            - Check if you accidentally add copyright or license headers. If so, remove them.
            - Try to run pre-commit if it is available.
            - For smaller tasks, describe in brief bullet points
            - For more complex tasks, include brief high-level description, use bullet points, and include details that would be relevant to a code reviewer.
- If completing the user's task DOES NOT require writing or modifying files (e.g., the user asks a question about the code base):
    - Respond in a friendly tone as a remote teammate, who is knowledgeable, capable and eager to help with coding.
- When your task involves writing or modifying files:
    - Do NOT tell the user to "save the file" or "copy the code into a file" if you already created or modified the file using \`apply_patch\`. Instead, reference the file as already saved.
    - Do NOT show the full contents of large files you have already written, unless the user explicitly asks for them.`;

function filterToApiMessages(
  items: Array<ResponseInputItem>,
): Array<ResponseInputItem> {
  return items.filter((it) => {
    if (it.type === "message" && it.role === "system") {
      return false;
    }
    if (it.type === "reasoning") {
      return false;
    }
    return true;
  });
}
