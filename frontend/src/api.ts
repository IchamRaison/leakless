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
      body?.error?.message ??
        // A gateway error without an API body means the dev proxy could not reach the API.
        ([502, 503, 504].includes(response.status)
          ? "API unreachable. Check that the local server is running, then retry."
          : `Server error (${response.status}).`),
    );
  const result = schema.safeParse(body);
  if (!result.success)
    throw new Error("Server response does not match this interface version.");
  return result.data;
}

export function formatTime(seconds: number): string {
  return `${seconds.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} s`;
}
