function stringify(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function pick(...values) {
  for (const value of values) {
    if (value === null || value === undefined) continue;
    const text = String(value).trim();
    if (text) return text;
  }
  return "";
}

export default function runtimeEvent(ctx) {
  const payload = ctx.payload || {};
  const service = payload.service || "unknown-service";
  const severity = payload.severity || "info";
  const summary = payload.summary || payload.message || "Runtime event received.";
  const details = stringify(payload.details || payload.context || payload.data);
  const lastGoodAt = pick(payload.lastGoodAt, payload.last_good_at, payload.lastHealthyAt, payload.last_healthy_at);
  const suggestedAction = pick(payload.suggestedAction, payload.suggested_action, payload.nextStep, payload.next_step);

  const lines = [
    `Runtime event for ${service}`,
    `severity=${severity}`,
    `summary=${summary}`,
    lastGoodAt ? `last_good_at=${lastGoodAt}` : "",
    suggestedAction ? `suggested_action=${suggestedAction}` : "",
    details ? `details=${details}` : "",
    "",
    "Assess operational urgency. Include the likely cause, the first verification step, whether /triage should be run immediately, and the next concrete action. Keep the reply concise and operator-focused."
  ].filter(Boolean);

  return {
    kind: "agent",
    message: lines.join("\n"),
    thinking: severity === "critical" || severity === "error" ? "medium" : "low"
  };
}
