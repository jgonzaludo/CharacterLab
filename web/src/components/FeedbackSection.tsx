import type {
  RehearsalFeedbackResult,
  SuggestionRating,
} from "../types";

type Props = {
  guidanceSlider: number;
  onGuidanceChange: (v: number) => void;
  analyzing: boolean;
  error: string | null;
  feedback: RehearsalFeedbackResult | null;
  ratings: Record<string, SuggestionRating>;
  onRate: (suggestionId: string, rating: Exclude<SuggestionRating, null>) => void;
};

export function FeedbackSection({
  guidanceSlider,
  onGuidanceChange,
  analyzing,
  error,
  feedback,
  ratings,
  onRate,
}: Props) {
  return (
    <section className="panel feedback-panel">
      <div className="field">
        <div className="field-label-row">
          <span className="field-label">Guidance style</span>
          <span className="field-hint">
            {guidanceSlider < 34
              ? "More exploratory"
              : guidanceSlider > 66
                ? "More directive"
                : "Balanced"}
          </span>
        </div>
        <div className="slider-row">
          <span className="slider-end">Exploratory</span>
          <input
            type="range"
            min={0}
            max={100}
            value={guidanceSlider}
            onChange={(e) => onGuidanceChange(Number(e.target.value))}
            className="guidance-slider"
            aria-label="Shift between exploratory and directive guidance"
          />
          <span className="slider-end">Directive</span>
        </div>
        <p className="slider-caption">
          Exploratory responses stay open-ended and interpretive; directive responses emphasize
          concrete, actionable experiments.
        </p>
      </div>

      {analyzing ? (
        <p className="status-line">Listening and drafting partner notes…</p>
      ) : null}
      {error ? (
        <div className="banner error" role="alert">
          {error}
        </div>
      ) : null}

      {feedback ? (
        <div className="feedback-body">
          <article className="feedback-block">
            <h3 className="feedback-heading">How this take may read</h3>
            <p className="feedback-text">{feedback.howItMayRead}</p>
          </article>

          <article className="feedback-block">
            <h3 className="feedback-heading">Script &amp; intention</h3>
            <p className="feedback-text">{feedback.interpretationNotes}</p>
          </article>

          {feedback.dimensionNotes &&
          (feedback.dimensionNotes.intensity ||
            feedback.dimensionNotes.restraint ||
            feedback.dimensionNotes.tension) ? (
            <article className="feedback-block">
              <h3 className="feedback-heading">Dimensions (voice)</h3>
              <ul className="dimension-list">
                {feedback.dimensionNotes.intensity ? (
                  <li>
                    <strong>Intensity:</strong> {feedback.dimensionNotes.intensity}
                  </li>
                ) : null}
                {feedback.dimensionNotes.restraint ? (
                  <li>
                    <strong>Restraint:</strong> {feedback.dimensionNotes.restraint}
                  </li>
                ) : null}
                {feedback.dimensionNotes.tension ? (
                  <li>
                    <strong>Tension:</strong> {feedback.dimensionNotes.tension}
                  </li>
                ) : null}
              </ul>
            </article>
          ) : null}

          {feedback.approximateTranscript ? (
            <article className="feedback-block transcript-block">
              <h3 className="feedback-heading">Approximate transcript</h3>
              <p className="feedback-text mono">{feedback.approximateTranscript}</p>
            </article>
          ) : null}

          <h3 className="feedback-heading suggestions-title">Directions to try</h3>
          <ul className="suggestion-list">
            {feedback.suggestions.map((s) => (
              <li key={s.id} className="suggestion-card">
                <h4 className="suggestion-title">{s.title}</h4>
                <p className="feedback-text">{s.body}</p>
                <p className="suggestion-reason">
                  <strong>Signals:</strong> {s.reasoning}
                </p>
                <p className="suggestion-try">
                  <strong>Try:</strong> {s.tryThis}
                </p>
                <div className="rating-row" role="group" aria-label="Was this suggestion helpful?">
                  <span className="rating-label">Helpful?</span>
                  <button
                    type="button"
                    className={`rate-btn ${ratings[s.id] === "up" ? "active" : ""}`}
                    onClick={() => onRate(s.id, "up")}
                    aria-pressed={ratings[s.id] === "up"}
                  >
                    Yes
                  </button>
                  <button
                    type="button"
                    className={`rate-btn ${ratings[s.id] === "down" ? "active" : ""}`}
                    onClick={() => onRate(s.id, "down")}
                    aria-pressed={ratings[s.id] === "down"}
                  >
                    Not really
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : !analyzing ? (
        <p className="placeholder-hint">
          Record a take to get exploratory notes grounded in your voice and script.
        </p>
      ) : null}
    </section>
  );
}
