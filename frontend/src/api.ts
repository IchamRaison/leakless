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
        ? "Le serveur met trop de temps à répondre. Réessayez."
        : "API inaccessible. Vérifiez que le serveur local est lancé.",
    );
  }
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      body?.error?.message ?? `Erreur du serveur (${response.status}).`,
    );
  const result = schema.safeParse(body);
  if (!result.success)
    throw new Error(
      "Réponse du serveur incompatible avec le studio. Vérifiez les versions.",
    );
  return result.data;
}

export function formatTime(seconds: number): string {
  return `${seconds.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} s`;
}
