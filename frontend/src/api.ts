import { z } from "zod";

export async function api<T>(
  path: string,
  schema: z.ZodType<T>,
  options: RequestInit = {},
): Promise<T> {
  const timeout = AbortSignal.timeout(20_000);
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeout])
    : timeout;
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { ...options, signal });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new Error(
      timeout.aborted
        ? "The server is taking too long to respond. Try again."
        : "API unreachable. Check that the local server is running.",
    );
  }
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      body?.error?.message ?? `Server error (${response.status}).`,
    );
  const result = schema.safeParse(body);
  if (!result.success)
    throw new Error("Server response does not match this interface version.");
  return result.data;
}

export function formatTime(seconds: number): string {
  return `${seconds.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} s`;
}
