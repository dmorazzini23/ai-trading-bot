function truncate(value, max = 1200) {
  const text = typeof value === "string" ? value.trim() : "";
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max - 3)}...` : text;
}

export default function githubEvent(ctx) {
  const event = ctx.headers["x-github-event"] || "unknown";
  const payload = ctx.payload || {};
  const repo = payload.repository?.full_name || payload.repository?.name || "unknown-repo";
  const action = payload.action ? ` action=${payload.action}` : "";
  const ref = payload.ref || payload.pull_request?.head?.ref || payload.issue?.html_url || "";
  const actor = payload.sender?.login || payload.pusher?.name || "unknown";
  const title = payload.pull_request?.title || payload.issue?.title || payload.head_commit?.message || payload.comment?.body || "";

  const lines = [
    `GitHub webhook for ${repo}`,
    `event=${event}${action}`,
    `actor=${actor}`,
    ref ? `ref=${ref}` : "",
    title ? `title=${truncate(title, 300)}` : "",
    "",
    "Review the payload impact on ai-trading-bot. Summarize what happened, whether follow-up is needed, and any repo/runtime risk."
  ].filter(Boolean);

  return {
    kind: "agent",
    message: lines.join("\n"),
    thinking: "medium"
  };
}
