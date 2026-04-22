function normalizeList(value) {
  if (!Array.isArray(value)) return "";
  const items = value.map((item) => String(item).trim()).filter(Boolean);
  return items.length > 0 ? items.join(", ") : "";
}

export default function codingTask(ctx) {
  const payload = ctx.payload || {};
  const taskId = payload.taskId || "task";
  const objective = payload.objective || payload.message || "No objective provided.";
  const files = normalizeList(payload.files);
  const branch = payload.branch ? String(payload.branch).trim() : "";
  const acceptance = normalizeList(payload.acceptanceCriteria);

  const lines = [
    `Coding task ${taskId}`,
    `objective=${objective}`,
    files ? `files=${files}` : "",
    branch ? `branch=${branch}` : "",
    acceptance ? `acceptance=${acceptance}` : "",
    "",
    "Work in the repo at /home/aiuser/ai-trading-bot. Prefer apply_patch for file edits, validate the result, and summarize what changed."
  ].filter(Boolean);

  return {
    kind: "agent",
    message: lines.join("\n"),
    thinking: "high"
  };
}
