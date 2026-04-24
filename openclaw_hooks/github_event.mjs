function truncate(value, max = 1200) {
  const text = typeof value === "string" ? value.trim() : "";
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max - 3)}...` : text;
}

function normalizeList(value) {
  if (!Array.isArray(value)) return "";
  const items = value.map((item) => {
    if (typeof item === "string") return item.trim();
    if (item && typeof item.filename === "string") return item.filename.trim();
    return "";
  }).filter(Boolean);
  return items.length > 0 ? items.join(", ") : "";
}

function collectCommitFiles(commits) {
  if (!Array.isArray(commits)) return "";
  const files = [];
  for (const commit of commits) {
    files.push(...(Array.isArray(commit?.added) ? commit.added : []));
    files.push(...(Array.isArray(commit?.modified) ? commit.modified : []));
    files.push(...(Array.isArray(commit?.removed) ? commit.removed : []));
  }
  return normalizeList(files);
}

export default function githubEvent(ctx) {
  const event = ctx.headers["x-github-event"] || "unknown";
  const payload = ctx.payload || {};
  const repo = payload.repository?.full_name || payload.repository?.name || "unknown-repo";
  const action = payload.action ? ` action=${payload.action}` : "";
  const ref = payload.ref || payload.pull_request?.head?.ref || payload.issue?.html_url || "";
  const actor = payload.sender?.login || payload.pusher?.name || "unknown";
  const title = payload.pull_request?.title || payload.issue?.title || payload.head_commit?.message || payload.comment?.body || "";
  const files = collectCommitFiles(payload.commits)
    || normalizeList(payload.pull_request?.files);

  const lines = [
    `GitHub webhook for ${repo}`,
    `event=${event}${action}`,
    `actor=${actor}`,
    ref ? `ref=${ref}` : "",
    title ? `title=${truncate(title, 300)}` : "",
    files ? `files=${truncate(files, 300)}` : "",
    "",
    "Review the payload impact on ai-trading-bot. Summarize what happened, whether follow-up is needed, any repo/runtime risk, and whether this should change deployment or service behavior."
  ].filter(Boolean);

  return {
    kind: "agent",
    message: lines.join("\n"),
    thinking: "medium"
  };
}
