// Pi extension: `take_action(action, argument)` forwards `{action}: {argument}` to the environment
// server as one `step` over the unix socket named in AGENT_WIKI_ENV_SOCKET. Node built-ins only,
// so the file loads from the session scratch dir, which has no node_modules.
import net from "node:net";

type Reply = { observation?: string; error?: string; [key: string]: unknown };

function step(socketPath: string, input: string): Promise<Reply> {
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		const client = net.createConnection(socketPath, () => {
			client.write(`${JSON.stringify({ op: "step", input })}\n`);
		});
		client.on("data", (chunk: Buffer) => {
			chunks.push(chunk);
			const text = Buffer.concat(chunks).toString("utf8");
			const nl = text.indexOf("\n");
			if (nl !== -1) {
				client.end();
				try {
					resolve(JSON.parse(text.slice(0, nl)) as Reply);
				} catch (err) {
					reject(err);
				}
			}
		});
		client.on("error", reject);
		client.on("end", () => {
			if (Buffer.concat(chunks).toString("utf8").indexOf("\n") === -1) {
				reject(new Error(`environment server closed without replying (${socketPath})`));
			}
		});
	});
}

export default function (pi: any) {
	const socketPath = process.env.AGENT_WIKI_ENV_SOCKET;
	if (!socketPath) {
		return;
	}
	// Queued once: a call after the end is answered with the marker too.
	let citationQueued = false;
	// From the turn the citation steer reaches the model, every tool is blocked, so its
	// answer is the JSON object and nothing reads the corpus after the episode.
	let citationTurn = false;
	// One take_action per turn: a blocked call never reaches the server, so the model sees each
	// observation before it sends the next command, as the baseline loop does.
	let callsThisTurn = 0;
	pi.on("turn_start", () => {
		callsThisTurn = 0;
		citationTurn = citationQueued;
	});
	pi.on("tool_call", (event: { toolName?: string }) => {
		if (citationTurn) {
			return { block: true, reason: "The episode is over. Answer with the JSON object only." };
		}
		if (event?.toolName !== "take_action") {
			return undefined;
		}
		callsThisTurn += 1;
		if (callsThisTurn > 1) {
			return { block: true, reason: "one take_action per turn: read this observation before sending the next action" };
		}
		return undefined;
	});
	pi.registerTool({
		name: "take_action",
		label: "take_action",
		description:
			"Take one action in the crafting environment and return its observation. " +
			"move: transfer a specific quantity of an item from one slot to another. " +
			"smelt: smelt an item in a furnace and move the output to a specific slot. " +
			"search: search for recipes to craft a specific item. " +
			"impossible: stop the task if it is certain that it is impossible with the given inventory.",
		parameters: {
			type: "object",
			properties: {
				action: { type: "string", description: "The action name: move, smelt, search or impossible." },
				argument: {
					type: "string",
					description:
						"The rest of the action: `from [Source] to [Target] with quantity N` for move and smelt, " +
						"`<recipe name>` for search, `<reason>` for impossible.",
				},
			},
			required: ["action", "argument"],
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: Record<string, unknown>) {
			const input = `${String(params?.action ?? "")}: ${String(params?.argument ?? "")}`;
			const reply = await step(socketPath, input);
			if (reply.error) {
				throw new Error(reply.error);
			}
			// The observation alone: Pi persists a result's `details` into the turn log, which
			// becomes a wiki raw step, so nothing the agent must not see may ride along.
			const obs = String(reply.observation ?? "");
			const citation = process.env.AGENT_WIKI_CITATION_PROMPT;
			if (citation && !citationQueued && (obs === "Episode over" || obs.endsWith("\n\nEpisode over"))) {
				citationQueued = true;
				pi.sendUserMessage(citation, { deliverAs: "steer" });
			}
			return { content: [{ type: "text", text: obs }] };
		},
	});
}
