import { GoogleGenerativeAI } from "@google/generative-ai";
import type { FacialTakeSummary, RehearsalFeedbackResult, VocalMetrics } from "../types";
import { formatFacialSummaryForPrompt } from "./facialPrompt";
import { isModelNotFoundError, isRetryableGeminiError, modelCandidates } from "./geminiModels";
import { parseJsonFromModelText } from "./parseJsonFromModel";

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result as string;
      const base64 = dataUrl.split(",")[1];
      if (!base64) {
        reject(new Error("Could not read recording as base64"));
        return;
      }
      resolve(base64);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function guidanceInstructions(slider: number): string {
  if (slider < 34) {
    return (
      "Lean exploratory: open-ended, interpretive language; invite curiosity; avoid prescriptive blocking; " +
      "suggest possibilities rather than fixed choices."
    );
  }
  if (slider > 66) {
    return (
      "Lean directive: specific, actionable adjustments; concrete line-level or breath-level ideas; " +
      "clear next steps the actor can try immediately."
    );
  }
  return (
    "Balance exploratory and directive: mix interpretive framing with a few concrete experiments."
  );
}

const JSON_SHAPE = `{
  "howItMayRead": "string — one short paragraph on how this take might land for a listener, without judging quality",
  "interpretationNotes": "string — how the script context and intended direction relate to what you notice",
  "dimensionNotes": {
    "intensity": "string — optional, tie to intensity metric + what you hear",
    "restraint": "string — optional",
    "tension": "string — optional"
  },
  "suggestions": [
    {
      "id": "unique-id-string",
      "title": "short label",
      "body": "1–3 sentences exploratory feedback",
      "reasoning": "what you noticed in the voice and/or optional on-camera cues (no clinical claims)",
      "tryThis": "one concrete experiment for the next take"
    }
  ],
  "approximateTranscript": "string — best-effort transcript or empty if unclear"
}`;

async function generateRehearsalFeedbackForModel(params: {
  apiKey: string;
  modelName: string;
  script: string;
  intendedDirection: string;
  metrics: VocalMetrics;
  facialSummary: FacialTakeSummary | null;
  guidanceSlider: number;
  recordingMimeType: string;
  recordingBase64: string;
}): Promise<RehearsalFeedbackResult> {
  const {
    apiKey,
    modelName,
    script,
    intendedDirection,
    metrics,
    facialSummary,
    guidanceSlider,
    recordingMimeType,
    recordingBase64,
  } = params;

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    generationConfig: {
      responseMimeType: "application/json",
    },
  });

  const systemPreamble = `You are a collaborative rehearsal partner for actors practicing alone. You are NOT an evaluator, judge, or casting director.

Rules:
- Do not assign scores, grades, or "good/bad" judgments.
- Avoid definitive clinical or psychological diagnoses.
- Describe how the performance may *read* to a listener and offer exploratory alternatives.
- Ground suggestions in audible cues (pace, breath, emphasis, clarity) and script context.
- You may reference approximate on-camera face cues when provided; they are ML heuristics, not truth—prioritize what you hear.
- When emotion timestamps (seconds from start of the audio) are provided, relate them to what you hear at those moments in the recording (e.g. line, breath, emphasis)—do not invent precise word-level alignment if uncertain.
- If voice metrics look unreliable, say so lightly and rely more on listening.
- Output valid JSON only, matching this shape exactly:
${JSON_SHAPE}`;

  const userText = `${systemPreamble}

Script / beat:
"""
${script.trim() || "(empty — infer general rehearsal)"}
"""

Intended emotional direction (optional):
"""
${intendedDirection.trim() || "(none specified)"}
"""

Heuristic voice metrics (0–1, approximate): intensity=${metrics.intensity.toFixed(2)}, restraint=${metrics.restraint.toFixed(2)}, tension=${metrics.tension.toFixed(2)}. Duration ~ ${metrics.durationSec.toFixed(1)}s. MetricsApproximate=${metrics.metricsApproximate}.

Approximate facial / on-camera cues over this take (when available):
${formatFacialSummaryForPrompt(facialSummary)}

Guidance style (${guidanceSlider}/100, 0=exploratory, 100=directive):
${guidanceInstructions(guidanceSlider)}

Listen to the attached recording and respond.`;

  if (import.meta.env.DEV) {
    console.log(
      "[CharacterLab] Gemini text prompt (audio sent separately as inlineData):\n\n",
      userText,
    );
  }

  const result = await model.generateContent([
    {
      inlineData: {
        mimeType: recordingMimeType,
        data: recordingBase64,
      },
    },
    { text: userText },
  ]);

  const response = result.response;
  const text = response.text();
  let parsed: RehearsalFeedbackResult;
  try {
    parsed = parseJsonFromModelText<RehearsalFeedbackResult>(text);
  } catch {
    throw new Error(
      "The model returned text that was not valid JSON. Try again or switch VITE_GEMINI_MODEL.",
    );
  }

  if (!parsed.suggestions?.length) {
    parsed.suggestions = [];
  }

  parsed.suggestions = parsed.suggestions.map((s, i) => ({
    ...s,
    id: s.id || `suggestion-${i}-${globalThis.crypto?.randomUUID?.().slice(0, 8) ?? String(i)}`,
  }));

  return parsed;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function generateWithRetry(
  params: Parameters<typeof generateRehearsalFeedbackForModel>[0],
): Promise<RehearsalFeedbackResult> {
  const maxAttempts = 3;
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await generateRehearsalFeedbackForModel(params);
    } catch (e) {
      lastError = e;
      if (!isRetryableGeminiError(e) || attempt === maxAttempts) {
        throw e;
      }
      const jitter = Math.floor(Math.random() * 200);
      const backoffMs = 500 * 2 ** (attempt - 1) + jitter;
      await sleep(backoffMs);
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Gemini request failed.");
}

export async function generateRehearsalFeedback(params: {
  apiKey: string;
  modelName: string;
  script: string;
  intendedDirection: string;
  metrics: VocalMetrics;
  facialSummary: FacialTakeSummary | null;
  guidanceSlider: number;
  recordingMimeType: string;
  recordingBase64: string;
}): Promise<RehearsalFeedbackResult> {
  const candidates = modelCandidates(params.modelName);
  let lastError: unknown;
  for (const name of candidates) {
    try {
      return await generateWithRetry({ ...params, modelName: name });
    } catch (e) {
      lastError = e;
      if (isModelNotFoundError(e) || isRetryableGeminiError(e)) continue;
      throw e;
    }
  }
  if (isRetryableGeminiError(lastError)) {
    throw new Error(
      "Gemini is currently under high demand. We retried and tried fallback models, but none were available. Please try again in a moment.",
    );
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(
        `No working Gemini model in: ${candidates.join(", ")}. Check VITE_GEMINI_MODEL and API access.`,
      );
}

export async function generateRehearsalFeedbackFromBlob(
  blob: Blob,
  params: Omit<
    Parameters<typeof generateRehearsalFeedback>[0],
    "recordingMimeType" | "recordingBase64"
  >,
): Promise<RehearsalFeedbackResult> {
  const mimeType = blob.type || "audio/webm";
  const base64 = await blobToBase64(blob);
  return generateRehearsalFeedback({
    ...params,
    recordingMimeType: mimeType,
    recordingBase64: base64,
  });
}
