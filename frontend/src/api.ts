// All backend calls live here; components never fetch directly.
// Dev requests go to /api/* and Vite proxies them to the FastAPI server (see vite.config.ts).
const BASE = "/api";

export interface Stimulus {
  id: string;
  domain: string;
  path: string;
  caption: string;
  tags: string[];
}

export interface Pair {
  a: Stimulus;
  b: Stimulus;
}

export interface Triad {
  a: Stimulus;
  b: Stimulus;
  c: Stimulus;
}

export interface Progress {
  n_choices: number;
  holdout_accuracy: number | null;
  done: boolean;
  reason: "converged" | "budget" | "collecting";
}

export interface FitResult {
  n_choices: number;
  holdout_accuracy: number | null;
  kernel_path: string | null;
}

/** An accuracy with its evidence: held-out sample size and Wilson 95% interval. */
export interface Metric {
  accuracy: number;
  n_test: number;
  low: number;
  high: number;
}

export interface ValidationResult {
  holdout: Metric | null;
  holdout_exemplar: Metric | null;
  best_model: string;
  cross_domain: Record<string, Metric>;
  ablation: Record<string, { target_only: Metric; all_domains: Metric }>;
  n_choices: number;
  per_domain_counts: Record<string, number>;
}

/** Session provenance for the Portrait footnote — confidence stays secondary. */
export interface MetaResult {
  provenance_banner: string;
  embedding_backend: string;
  n_choices?: number;
}

export interface ReflectResult {
  hero_ids: string[];
  top_tags: string[];
  summary: string;
}

export interface RecordIn {
  chosen_id: string;
  rejected_id: string;
  domain: string;
  kind: string;
  weight?: number;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${url} failed: ${r.status}`);
  return r.json();
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(`${BASE}${url}`);
  if (!r.ok) throw new Error(`GET ${url} failed: ${r.status}`);
  return r.json();
}

export const api = {
  meta: () => get<MetaResult>("/meta"),
  listStimuli: () => get<Stimulus[]>("/stimuli"),
  createSession: (person_id = "anon") =>
    post<{ session_id: string }>("/sessions", { person_id }),
  grid: (sessionId: string, k = 9, domain?: string) => {
    const params = new URLSearchParams({ k: String(k) });
    if (domain) params.set("domain", domain);
    return get<{ items: Stimulus[] }>(`/sessions/${sessionId}/grid?${params}`);
  },
  nextPair: (sessionId: string, domain?: string) =>
    get<Pair>(`/sessions/${sessionId}/next-pair${domain ? `?domain=${domain}` : ""}`),
  nextTriad: (sessionId: string, domain?: string) =>
    get<Triad>(`/sessions/${sessionId}/next-triad${domain ? `?domain=${domain}` : ""}`),
  postChoice: (sessionId: string, chosen_id: string, rejected_id: string, domain: string) =>
    post<{ ok: boolean }>("/choices", { session_id: sessionId, chosen_id, rejected_id, domain }),
  postBatch: (sessionId: string, records: RecordIn[]) =>
    post<{ ok: boolean; n: number }>("/choices/batch", { session_id: sessionId, records }),
  postTriad: (sessionId: string, odd_id: string, other_ids: string[], domain: string) =>
    post<{ ok: boolean }>("/triads", { session_id: sessionId, odd_id, other_ids, domain }),
  progress: (sessionId: string) => get<Progress>(`/sessions/${sessionId}/progress`),
  reflect: (sessionId: string) => get<ReflectResult>(`/sessions/${sessionId}/reflect`),
  fit: (sessionId: string) => post<FitResult>(`/sessions/${sessionId}/fit`, {}),
  validate: (sessionId: string) => get<ValidationResult>(`/sessions/${sessionId}/validate`),
};

/** Stimulus image URL (backend serves data/stimuli/* statically). */
export function stimulusImageUrl(s: Stimulus): string {
  return `${BASE}/${s.path}`;
}
