type Props = {
  script: string;
  intendedDirection: string;
  onScriptChange: (v: string) => void;
  onIntendedChange: (v: string) => void;
};

export function ScriptPanel({
  script,
  intendedDirection,
  onScriptChange,
  onIntendedChange,
}: Props) {
  return (
    <section className="panel script-panel">
      <label className="field">
        <span className="field-label">Script or beat</span>
        <textarea
          className="textarea"
          value={script}
          onChange={(e) => onScriptChange(e.target.value)}
          placeholder="Paste a short line, beat, or monologue you’re working on…"
          rows={8}
        />
      </label>
      <label className="field">
        <span className="field-label">Intended emotional direction (optional)</span>
        <textarea
          className="textarea textarea-sm"
          value={intendedDirection}
          onChange={(e) => onIntendedChange(e.target.value)}
          placeholder="e.g. controlled grief, barely contained anger, warm invitation…"
          rows={3}
        />
      </label>
    </section>
  );
}
