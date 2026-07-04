/** Mid-session interlude: the AI names a pattern discovered from choices; the user ratifies.
 * Words are AI-authored, user-confirmed — never free-text taste input. */
export function ReflectBack({
  summary,
  heroSrcs,
  onConfirm,
}: {
  summary: string;
  heroSrcs: string[];
  onConfirm: (ok: boolean) => void;
}) {
  return (
    <div className="wrap reflect-back">
      <p className="reflect-summary">{summary}</p>
      <div className="reflect-heroes">
        {heroSrcs.map((src, i) => (
          <div key={`${src}-${i}`} className="reflect-hero">
            <img src={src} alt="" onError={(e) => (e.currentTarget.style.display = "none")} />
          </div>
        ))}
      </div>
      <p className="muted small reflect-prompt">Does this feel like you?</p>
      <div className="reflect-actions">
        <button type="button" className="primary" onClick={() => onConfirm(true)}>
          Yes, that's me
        </button>
        <button type="button" className="reflect-secondary" onClick={() => onConfirm(false)}>
          Not quite
        </button>
      </div>
    </div>
  );
}
