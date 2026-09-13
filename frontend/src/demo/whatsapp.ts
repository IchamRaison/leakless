// WhatsApp alert for the illustrative simulation: a wa.me link the operator confirms.
// No API, no automatic sending.
import type { SimSensor } from "./simulation";

/** Every alert this demo can open must say it is a demo or a simulation. */
export function assertDemoAlert(message: string) {
  if (!/\b(demo|simulated)\b/i.test(message))
    throw new Error("Refused: an alert must be marked as demo or simulated.");
  return message;
}

export function buildAlertMessage(sensor: Pick<SimSensor, "name" | "zone">) {
  return assertDemoAlert(
    [
      "LeakLess demo alert",
      `Simulated anomaly detected near ${sensor.name}.`,
      `Location: Building demo network · ${sensor.zone}.`,
      "Action: inspection recommended.",
    ].join("\n"),
  );
}

/** Optional recipient from VITE_ALERT_WHATSAPP, digits only (international format). */
export function alertRecipient(value = import.meta.env.VITE_ALERT_WHATSAPP) {
  const digits = typeof value === "string" ? value.replace(/\D/g, "") : "";
  return digits.length >= 8 ? digits : "";
}

export function whatsappLink(message: string, recipient = alertRecipient()) {
  const text = encodeURIComponent(assertDemoAlert(message));
  return `https://wa.me/${recipient}?text=${text}`;
}
