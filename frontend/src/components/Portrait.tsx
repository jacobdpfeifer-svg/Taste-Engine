import type { MetaResult, Metric, ValidationResult } from "../api";

const DEFAULT_PROVENANCE =
  "Measured on your own held-out choices, not asserted. A preference model, not a diagnosis of you.";

/** End-of-session portrait: taste as images and words up front; confidence in a small footnote. */
export function Portrait({
  result,
  heroSrcs,
  topTags,
  meta,
}: {
  result: ValidationResult;
  heroSrcs: string[];
  topTags: string[];
  meta: MetaResult;
}) {
  const banner = meta.provenance_banner || DEFAULT_PROVENANCE;

  return (
    <div className="wrap portrait">
      <h1 className="portrait-headline">Here's the taste I'm seeing</h1>

      {heroSrcs.length > 0 && (
        <div className="portrait-heroes">
          {heroSrcs.map((src, i) => (
            <div key={`${src}-${i}`} className="portrait-hero">
              <img src={src} alt="" onError={(e) => (e.currentTarget.style.display = "none")} />
            </div>
          ))}
        </div>
      )}

      {topTags.length > 0 ? (
        <p className="portrait-tags">
          {topTags.map((tag, i) => (
            <span key={tag}>
              {i > 0 && <span className="portrait-tag-sep"> · </span>}
              <span className="portrait-tag">{tag}</span>
            </span>
          ))}
        </p>
      ) : (
        <p className="muted portrait-tags-empty">Still forming — not enough signal to name it yet.</p>
      )}

      <p className="portrait-footnote">
        {fmtHoldout(result.holdout)}
        {result.holdout != null && " · "}
        {banner}
        {meta.n_choices != null && meta.n_choices > 0 && ` · ${meta.n_choices} choices`}
      </p>
    </div>
  );
}

function fmtHoldout(h: Metric | null): string {
  if (h == null) return "Hold-out: not enough choices yet";
  return `Hold-out ${(h.accuracy * 100).toFixed(0)}% (95% CI ${(h.low * 100).toFixed(0)}–${(h.high * 100).toFixed(0)}%, n=${h.n_test})`;
}
