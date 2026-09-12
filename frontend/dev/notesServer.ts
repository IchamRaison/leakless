import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

/** Review notes pinned on the running site. Dev server only; never part of the build. */
export type SiteNote = {
  id: string;
  number: number;
  author: string;
  text: string;
  status: "open" | "resolved";
  createdAt: string;
  anchor: {
    selector: string;
    section: string;
    element: string;
    snippet: string;
    offsetX: number;
    offsetY: number;
    viewportWidth: number;
    hash: string;
  };
};

const limit = (value: unknown, max: number) =>
  typeof value === "string" ? value.trim().slice(0, max) : "";
const ratio = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : 0;

export async function readNotes(file: string): Promise<SiteNote[]> {
  try {
    const data = JSON.parse(await readFile(file, "utf8"));
    return Array.isArray(data.notes) ? data.notes : [];
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function writeNotes(file: string, notes: SiteNote[]) {
  await mkdir(dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.tmp`;
  await writeFile(
    temp,
    `${JSON.stringify({ version: 1, notes }, null, 2)}\n`,
    "utf8",
  );
  await rename(temp, file);
}

/** Validates client input; server owns id, number, status and timestamp. */
export function createNote(body: unknown, existing: SiteNote[]): SiteNote {
  const input = (body ?? {}) as Record<string, unknown>;
  const anchor = (input.anchor ?? {}) as Record<string, unknown>;
  const text = limit(input.text, 2000);
  const selector = limit(anchor.selector, 1000);
  if (!text) throw new Error("Note text is required.");
  if (!selector) throw new Error("Note anchor is required.");
  return {
    id: randomUUID(),
    number: Math.max(0, ...existing.map((note) => note.number)) + 1,
    author: limit(input.author, 60) || "anonymous",
    text,
    status: "open",
    createdAt: new Date().toISOString(),
    anchor: {
      selector,
      section: limit(anchor.section, 120),
      element: limit(anchor.element, 60),
      snippet: limit(anchor.snippet, 160),
      offsetX: ratio(anchor.offsetX),
      offsetY: ratio(anchor.offsetY),
      viewportWidth: Math.round(Number(anchor.viewportWidth) || 0),
      hash: limit(anchor.hash, 60),
    },
  };
}

async function body(request: IncomingMessage) {
  let raw = "";
  for await (const chunk of request) {
    raw += chunk;
    if (raw.length > 32_000) throw new Error("Payload too large.");
  }
  return raw ? JSON.parse(raw) : {};
}

function send(response: ServerResponse, status: number, payload: unknown) {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(payload));
}

export function siteNotes(file: string): Plugin {
  // Serialise writes so two reviewers saving at once never lose a note.
  let queue = Promise.resolve();
  const exclusive = <T>(task: () => Promise<T>) => {
    const run = queue.then(task);
    queue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  };
  return {
    name: "leakless-site-notes",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__notes", async (request, response) => {
        try {
          const id = (request.url ?? "/").replace(/^\/+|\?.*$/g, "");
          if (request.method === "GET" && !id)
            return send(response, 200, { notes: await readNotes(file) });
          if (request.method === "POST" && !id) {
            const input = await body(request);
            const note = await exclusive(async () => {
              const notes = await readNotes(file);
              const created = createNote(input, notes);
              await writeNotes(file, [...notes, created]);
              return created;
            });
            return send(response, 201, note);
          }
          if (
            (request.method === "PATCH" || request.method === "DELETE") &&
            id
          ) {
            const input = request.method === "PATCH" ? await body(request) : {};
            const result = await exclusive(async () => {
              const notes = await readNotes(file);
              const index = notes.findIndex((note) => note.id === id);
              if (index < 0) return null;
              if (request.method === "DELETE") notes.splice(index, 1);
              else
                notes[index] = {
                  ...notes[index],
                  status: input.status === "resolved" ? "resolved" : "open",
                };
              await writeNotes(file, notes);
              return request.method === "DELETE" ? {} : notes[index];
            });
            return result
              ? send(response, 200, result)
              : send(response, 404, { error: "Note not found." });
          }
          send(response, 405, { error: "Method not allowed." });
        } catch (error) {
          send(response, 400, { error: (error as Error).message });
        }
      });
    },
  };
}
