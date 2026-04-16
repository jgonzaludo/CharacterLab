/**
 * Gemini sometimes wraps JSON in markdown fences; strip and parse safely.
 */
export function parseJsonFromModelText<T>(text: string): T {
  let raw = text.trim();
  const fence = raw.match(/^```(?:json)?\s*([\s\S]*?)```$/m);
  if (fence) {
    raw = fence[1].trim();
  }
  return JSON.parse(raw) as T;
}
