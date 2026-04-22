function stringify(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export default function runtimeEvent(ctx) {
  const payload = ctx.payload || {};
  const service = payload.service || "unknown-service";
  const severity = payload.severity || "info";
  const summary = payload.summary || payload.message || "Runtime event received.";
  const details = stringify(payload.details || payload.context || payload.data);

  const lines = [
    `Runtime event for ${service}`,
    `severity=${severity}`,
    `summary=${summary}`,
    details ? `details=${details}` : "",
    "",
    "Assess operational urgency, mention the next command or check if needed, and keep the reply concise."
  ].filter(Boolean);

  return {
    kind: "agent",
    message: lines.join("\n"),
    thinking: severity === "critical" || severity === "error" ? "medium" : "low"
  };
}
