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

function isTruthySignal(value) {
  if (value === true) return true;
  if (typeof value !== "string") return false;
  return ["1", "true", "yes", "y"].includes(value.trim().toLowerCase());
}

function hasExplicitSmokeLabel(value) {
  if (value === null || value === undefined) return false;
  const text = String(value).trim().toLowerCase();
  if (!text) return false;
  return [
    "smoke",
    "smoke_test",
    "smoke-test",
    "synthetic",
    "synthetic_test",
    "synthetic-test",
    "synthetic_validation",
    "synthetic-validation"
  ].includes(text) || /\bsmoke[-_ ]test\b/.test(text);
}

function looksLikeSmokeTest(payload, detailsText) {
  const details = payload.details && typeof payload.details === "object" ? payload.details : {};
  const flagSignals = [
    payload.synthetic,
    payload.smoke_test,
    payload.smokeTest,
    payload.is_synthetic,
    details.synthetic,
    details.smoke_test,
    details.smokeTest,
    details.is_synthetic
  ];
  if (flagSignals.some(isTruthySignal)) return true;

  const labelSignals = [
    payload.event_type,
    payload.type,
    payload.category,
    payload.source,
    details.event_type,
    details.type,
    details.category,
    details.source,
    details.model_id,
    details.model_name
  ];
  if (labelSignals.some(hasExplicitSmokeLabel)) return true;

  const pathSignals = [
    details.report_path,
    details.reportPath,
    payload.report_path,
    payload.reportPath
  ].map((value) => String(value || ""));
  if (pathSignals.some((text) => text.startsWith("/tmp/") || text.includes("/tmp/"))) return true;

  const explicitTextSignals = [
    payload.summary,
    payload.message,
    detailsText
  ].map((value) => String(value || "").toLowerCase());
  return explicitTextSignals.some((text) =>
    /\btemporary smoke[-_ ]test\b/.test(text) ||
    /\bsynthetic validation payload\b/.test(text)
  );
}

export default function runtimeEvent(ctx) {
  const payload = ctx.payload || {};
  const service = payload.service || "unknown-service";
  const severity = payload.severity || "info";
  const summary = payload.summary || payload.message || "Runtime event received.";
  const details = stringify(payload.details || payload.context || payload.data);
  const lastGoodAt = pick(payload.lastGoodAt, payload.last_good_at, payload.lastHealthyAt, payload.last_healthy_at);
  const suggestedAction = pick(payload.suggestedAction, payload.suggested_action, payload.nextStep, payload.next_step);
  const smokeTest = looksLikeSmokeTest(payload, details);
  const policy = payload.operatorAssistantPolicy || payload.operator_assistant_policy || {};
  const criticalRoute = pick(policy.critical_alert_route, "#all-beatwallstreet");

  const lines = [
    `Runtime event for ${service}`,
    `severity=${severity}`,
    `summary=${summary}`,
    smokeTest ? "smoke_test=true" : "",
    lastGoodAt ? `last_good_at=${lastGoodAt}` : "",
    suggestedAction ? `suggested_action=${suggestedAction}` : "",
    details ? `details=${details}` : "",
    "",
    smokeTest
      ? "This is a smoke-test or synthetic validation payload. Say explicitly that no trading or configuration action should be taken from this event alone."
      : "",
    "Operator assistant policy: default to fast, read-only, artifact-based analysis. Do not run broad validation, training, backtests, code patches, or service restarts from Slack. If code changes are needed, propose a Codex /goal instead of editing. For urgent runtime issues, summarize the situation and recommend exact operator commands. Critical alerts should be called out clearly for " + criticalRoute + ".",
    "Assess operational urgency. Include the likely cause, the first verification step, whether /triage should be run immediately, and the next concrete action. Keep the reply concise and operator-focused."
  ].filter(Boolean);

  return {
    kind: "agent",
    message: lines.join("\n"),
    thinking: severity === "critical" || severity === "error" ? "medium" : "low"
  };
}
