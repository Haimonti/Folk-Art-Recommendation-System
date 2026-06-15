const SID_KEY = "gemi_sid";
const UID_KEY = "gemi_uid";

function randomId(n = 8): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    const arr = new Uint8Array(n);
    crypto.getRandomValues(arr);
    for (let i = 0; i < n; i++) out += chars[arr[i] % chars.length];
  } else {
    for (let i = 0; i < n; i++) out += chars[Math.floor(Math.random() * chars.length)];
  }
  return out;
}

export function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let sid = window.localStorage.getItem(SID_KEY);
  if (!sid) {
    sid = randomId(8);
    window.localStorage.setItem(SID_KEY, sid);
  }
  return sid;
}

export function getUserId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(UID_KEY);
}

export function setUserId(userId: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(UID_KEY, userId);
}

function post(path: string, body: Record<string, unknown>): void {
  if (typeof window === "undefined") return;
  try {
    fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      keepalive: true,
    }).catch(() => {});
  } catch {
    return;
  }
}

export function logEvent(
  eventType: string,
  opts: { panelIndex?: number; panelId?: string; value?: number; context?: Record<string, unknown> } = {}
): void {
  post("/api/events", {
    session_id: getSessionId(),
    user_id: getUserId(),
    event_type: eventType,
    panel_index: opts.panelIndex ?? null,
    panel_id: opts.panelId ?? null,
    value: opts.value ?? null,
    context: opts.context ?? null,
  });
}

export function logRecommendation(opts: {
  modelName?: string;
  seedPanelIndices?: number[];
  servedPanelIndices?: number[];
  queryText?: string;
}): void {
  post("/api/recommendations/log", {
    session_id: getSessionId(),
    user_id: getUserId(),
    model_name: opts.modelName ?? null,
    seed_panel_indices: opts.seedPanelIndices ?? null,
    served_panel_indices: opts.servedPanelIndices ?? null,
    query_text: opts.queryText ?? null,
  });
}

export async function registerUser(handle: string): Promise<string | null> {
  if (typeof window === "undefined") return null;
  try {
    const res = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ handle }),
    });
    const data = await res.json();
    if (data && data.user_id) {
      setUserId(data.user_id as string);
      return data.user_id as string;
    }
    return null;
  } catch {
    return null;
  }
}
