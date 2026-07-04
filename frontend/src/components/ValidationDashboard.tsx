import type { Metric, ValidationResult } from "../api";

const CHANCE = 0.5;

// The three falsifiable tests (docs/SPEC.md), reported with error bars. Every number carries
// its held-out sample size and a Wilson 95% interval; verdicts are only issued when the
// interval clears chance. Confidence, not certainty.
export function ValidationDashboard({ result }: { result: ValidationResult }) {
  const h = result.holdout;
  const verdict = h == null ? "none" : h.low > CHANCE ? "pass" : h.high < CHANCE ? "fail" : "inconclusive";

  return (
    <div className="wrap">
      <h1>Your taste — validation</h1>
      <p className="muted">
        Fitted on {result.n_choices} choices ({perDomain(result.per_domain_counts)}). Does the
        model predict your held-out choices better than a coin flip? That is the whole question
        this MVP answers.
      </p>

      <h2>1. Hold-out accuracy</h2>
      <div className="metric">
        <span>Predicting your unseen choices</span>
        <strong className={verdict === "pass" ? "good" : verdict === "inconclusive" ? "" : "bad"}>
          {fmtAcc(h)}
        </strong>
        <span className="muted">{fmtCi(h)} · chance = 0.50</span>
      </div>
      {verdict === "pass" && (
        <p className="muted">
          The interval clears chance: your choices are predictable from geometry. Phase 1's
          question, answered on your own held-out choices.
        </p>
      )}
      {verdict === "inconclusive" && (
        <p className="warn">
          Inconclusive: the confidence interval still includes the coin flip. This usually means
          not enough choices yet, or the embedding can't see what you were choosing on
          (placeholder images + stub embeddings can't capture real visual taste — use real
          photos with EMBEDDING_BACKEND=openclip).
        </p>
      )}
      {verdict === "fail" && (
        <p className="warn">
          Below chance with the interval clear of 0.50. Honest read: on this stimulus bank and
          embedding, the premise did not hold. Stop and investigate before building further.
        </p>
      )}

      <h2>2. Cross-domain transfer</h2>
      <p className="muted">Fit on the other domains, predict the held-out one. Above 0.50 means
        your taste in one medium says something about another.</p>
      <table>
        <thead>
          <tr><th>held-out domain</th><th>accuracy</th><th>95% interval</th><th></th></tr>
        </thead>
        <tbody>
          {Object.entries(result.cross_domain).map(([d, m]) => (
            <tr key={d}>
              <td>{d}</td>
              <td className={m.low > CHANCE ? "good" : m.high < CHANCE ? "bad" : ""}>{fmtAcc(m)}</td>
              <td className="muted">{fmtCi(m)}</td>
              <td className="muted">{clearsChance(m)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>3. Ablation</h2>
      <p className="muted">Does adding your choices from other mediums improve prediction inside
        one medium? Averaged over repeated splits; differences whose intervals overlap are
        reported as noise.</p>
      <table>
        <thead>
          <tr><th>domain</th><th>that domain only</th><th>all domains</th><th></th></tr>
        </thead>
        <tbody>
          {Object.entries(result.ablation).map(([d, v]) => (
            <tr key={d}>
              <td>{d}</td>
              <td>{fmtAcc(v.target_only)} <span className="muted">{fmtCi(v.target_only)}</span></td>
              <td>{fmtAcc(v.all_domains)} <span className="muted">{fmtCi(v.all_domains)}</span></td>
              <td className="muted">{ablationVerdict(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="muted small">
        These numbers are measured on your own held-out choices, not asserted. They describe a
        preference model, not you.
      </p>
    </div>
  );
}

function fmtAcc(m: Metric | null): string {
  return m == null ? "—" : m.accuracy.toFixed(2);
}

function fmtCi(m: Metric | null): string {
  if (m == null) return "";
  return `${m.low.toFixed(2)}–${m.high.toFixed(2)}, n=${m.n_test}`;
}

function clearsChance(m: Metric): string {
  if (m.low > CHANCE) return "above chance";
  if (m.high < CHANCE) return "below chance";
  return "within noise of chance";
}

function perDomain(counts: Record<string, number>): string {
  return Object.entries(counts).map(([d, n]) => `${d}: ${n}`).join(", ");
}

function ablationVerdict(v: { target_only: Metric; all_domains: Metric }): string {
  // Overlapping intervals mean the difference is not distinguishable from noise.
  const overlap = v.target_only.low <= v.all_domains.high && v.all_domains.low <= v.target_only.high;
  if (overlap) return "no clear difference";
  return v.all_domains.accuracy > v.target_only.accuracy
    ? "other mediums helped"
    : "other mediums hurt";
}
