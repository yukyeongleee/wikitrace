// Pi extension: `take_action(action, argument)` for tau3 retail forwards `{"action", "argument"}` as
// one JSON string, one `step` over the unix socket named in AGENT_WIKI_ENV_SOCKET. Node built-ins
// only, so the file loads from the session scratch dir, which has no node_modules.
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
	// Queued once: a call after the end is answered with the marker too. The runner sends the
	// prompt itself when a message ended the episode, and this marks it queued all the same.
	let citationQueued = false;
	// From the turn the citation prompt reaches the model, every tool is blocked, so its
	// answer is the JSON object and nothing reads the corpus after the episode.
	let citationTurn = false;
	// One take_action per turn: a blocked call never reaches the server, so the model sees each
	// result before it sends the next action, as the reference agent kept one tool call per message.
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
			return { block: true, reason: "one take_action per turn: read this result before sending the next action" };
		}
		return undefined;
	});
	pi.on("input", (event: { text?: string }) => {
		const citation = process.env.AGENT_WIKI_CITATION_PROMPT;
		if (citation && event?.text === citation) {
			citationQueued = true;
		}
		return undefined;
	});
	pi.registerTool({
		name: "take_action",
		label: "take_action",
		description:
			"Take one action in the customer service environment and return its result. " +
			"`action` is the name of one of the retail tools listed in the system prompt, and `argument` " +
			"is that tool's parameters as one JSON object. The result is the tool's output.",
		parameters: {
			type: "object",
			properties: {
				action: { type: "string", description: "A retail tool name." },
				// A string passes too, so it reaches the server and counts as an error there.
				argument: {
					anyOf: [{ type: "string" }, { type: "object", additionalProperties: true }],
					description: "The tool's parameters as one JSON object (`{}` for a tool without parameters).",
				},
			},
			required: ["action", "argument"],
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: Record<string, unknown>) {
			// One JSON string on the wire: the socket protocol carries one step string, and the
			// server parses `argument` back as an object or text.
			const input = JSON.stringify({ action: String(params?.action ?? ""), argument: params?.argument ?? "" });
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
