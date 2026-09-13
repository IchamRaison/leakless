// Authoritative incident and alert state, owned by the backend (/incidents). The browser only
// reads it and asks to inject or resolve; persistence and Telegram dispatch happen on the server.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { api } from "../api";

const incidentSchema = z.object({
  incident_id: z.string(),
  source: z.string(),
  strongest_sensor: z.number().int().min(1).max(3),
  zone: z.string(),
  position: z.object({ x: z.number(), y: z.number() }),
  started_at: z.number(),
  status: z.enum([
    "ANOMALY_PENDING",
    "ALERT_DISPATCHING",
    "ALERT_SENT",
    "ALERT_FAILED",
    "RESOLVED",
  ]),
  confirmed_at: z.number().nullable(),
  alert_sent_at: z.number().nullable(),
  resolved_at: z.number().nullable(),
  transport: z.string().nullable(),
  delivery_status: z.string(),
  telegram_message_id: z.string().nullable(),
  failure_reason: z.string().nullable(),
});
export type Incident = z.infer<typeof incidentSchema>;

export const snapshotSchema = z.object({
  server_time: z.number(),
  persistence_seconds: z.number(),
  transport: z.object({ name: z.string(), configured: z.boolean() }),
  active: incidentSchema.nullable(),
  history: z.array(incidentSchema),
});
export type IncidentSnapshot = z.infer<typeof snapshotSchema>;

export const POLL_MS = 1000;

export function useIncidents() {
  const [snapshot, setSnapshot] = useState<IncidentSnapshot | null>(null);
  const [error, setError] = useState("");
  // Server clock minus browser clock, so persistence counts in server time.
  const [offsetMs, setOffsetMs] = useState(0);
  const [busy, setBusy] = useState(false);
  // Every request gets an increasing number; a response older than the last applied one is
  // dropped, so a slow poll can never overwrite the newer result of an inject or resolve.
  const sequence = useRef(0);
  const applied = useRef(0);
  const polling = useRef<AbortController | null>(null);
  const mutating = useRef(0);

  const accept = useCallback(
    (next: IncidentSnapshot, requestNo: number, sentAtMs: number) => {
      if (requestNo <= applied.current) return;
      applied.current = requestNo;
      setSnapshot(next);
      // Compared with the send time: the offset may include the request's one-way latency
      // (milliseconds on a LAN), but a response that waits in the browser never skews it.
      setOffsetMs(next.server_time * 1000 - sentAtMs);
      setError("");
    },
    [],
  );

  useEffect(() => {
    let alive = true;
    const poll = () => {
      // One poll at a time, and none while an inject/resolve/retry is in flight.
      if (polling.current || mutating.current) return;
      const controller = new AbortController();
      polling.current = controller;
      const requestNo = ++sequence.current;
      const sentAtMs = Date.now();
      api("/incidents/current", snapshotSchema, { signal: controller.signal })
        .then((next) => alive && accept(next, requestNo, sentAtMs))
        .catch((reason: Error) => {
          if (alive && !controller.signal.aborted) setError(reason.message);
        })
        .finally(() => {
          if (polling.current === controller) polling.current = null;
        });
    };
    poll();
    const timer = window.setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
      polling.current?.abort();
    };
  }, [accept]);

  const post = async (path: string, body?: unknown) => {
    // A mutation supersedes any in-flight poll.
    polling.current?.abort();
    polling.current = null;
    const requestNo = ++sequence.current;
    const sentAtMs = Date.now();
    mutating.current += 1;
    setBusy(true);
    try {
      accept(
        await api(path, snapshotSchema, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: body === undefined ? undefined : JSON.stringify(body),
        }),
        requestNo,
        sentAtMs,
      );
    } catch {
      // A refused action (e.g. retry after the server already moved on) is not an outage:
      // the next poll shows the authoritative state.
    } finally {
      mutating.current -= 1;
      setBusy(false);
    }
  };

  return {
    snapshot,
    error,
    busy,
    offsetMs,
    inject: (strongestSensor: number, position: { x: number; y: number }) =>
      post("/incidents", {
        strongest_sensor: strongestSensor,
        position,
      }),
    resolve: (incidentId: string) => post(`/incidents/${incidentId}/resolve`),
    retry: (incidentId: string) => post(`/incidents/${incidentId}/retry`),
  };
}
