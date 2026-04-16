/**
 * Older aliases (e.g. gemini-1.5-flash) often 404 on v1beta. Prefer current Flash variants.
 * Order: try preferred first, then fallbacks only on 404.
 */
export const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";

export const GEMINI_MODEL_FALLBACKS = [
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite",
] as const;

export function modelCandidates(preferred: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of [preferred, ...GEMINI_MODEL_FALLBACKS]) {
    if (!seen.has(m)) {
      seen.add(m);
      out.push(m);
    }
  }
  return out;
}

export function isModelNotFoundError(err: unknown): boolean {
  const s = err instanceof Error ? err.message : String(err);
  return (
    s.includes("404") ||
    /\bnot found\b/i.test(s) ||
    s.includes("is not found for API version")
  );
}
