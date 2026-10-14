// Pi extension: `exec(code)` forwards one Python snippet to the environment server as a
// `step` over the unix socket named in AGENT_WIKI_ENV_SOCKET. Node built-ins only, so the
// file loads from the session scratch dir, which has no node_modules.
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
	// One exec per turn: a blocked call never reaches the server, so the model sees each
	// output before it sends the next chunk, as the baseline loop does.
	let callsThisTurn = 0;
	pi.on("turn_start", () => {
		callsThisTurn = 0;
		citationTurn = citationQueued;
	});
	pi.on("tool_call", (event: { toolName?: string }) => {
		if (citationTurn) {
			return { block: true, reason: "The episode is over. Answer with the JSON object only." };
		}
		if (event?.toolName !== "exec") {
			return undefined;
		}
		callsThisTurn += 1;
		if (callsThisTurn > 1) {
			return { block: true, reason: "one exec per turn: read this output before sending the next chunk" };
		}
		return undefined;
	});
	pi.registerTool({
		name: "exec",
		label: "exec",
		description: "Run Python in the task's shell and return its output.",
		parameters: {
			type: "object",
			properties: { code: { type: "string", description: "Python source to run; state persists across calls." } },
			required: ["code"],
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: Record<string, unknown>) {
			const reply = await step(socketPath, String(params?.code ?? ""));
			if (reply.error) {
				throw new Error(reply.error);
			}
			// The output alone: Pi persists a result's `details` into the turn log, which
			// becomes a wiki raw step, so nothing the agent must not see may ride along.
			const out = String(reply.observation ?? "");
			const citation = process.env.AGENT_WIKI_CITATION_PROMPT;
			if (citation && !citationQueued && (out === "Episode over" || out.endsWith("\n\nEpisode over"))) {
				citationQueued = true;
				pi.sendUserMessage(citation, { deliverAs: "steer" });
			}
			return { content: [{ type: "text", text: out }] };
		},
	});
}
