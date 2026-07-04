import { stimulusImageUrl, type Stimulus } from "../api";

// The core interaction: two options, one click records a preference.
// No text input, no rating — recognition and choice only.
export function PairChoice({
  a,
  b,
  onChoose,
}: {
  a: Stimulus;
  b: Stimulus;
  onChoose: (chosen: Stimulus, rejected: Stimulus) => void;
}) {
  return (
    <div className="pair">
      <Card stim={a} onClick={() => onChoose(a, b)} />
      <div className="vs">or</div>
      <Card stim={b} onClick={() => onChoose(b, a)} />
    </div>
  );
}

// Image only — no caption. Words during elicitation invite choosing by description
// (verbal overshadowing), and the pixels are what gets embedded. The caption stays in the
// aria-label for screen readers.
function Card({ stim, onClick }: { stim: Stimulus; onClick: () => void }) {
  return (
    <button className="card" onClick={onClick} aria-label={stim.caption}>
      <div className="thumb">
        {/* Falls back to a domain label tile if the image is missing from the bank. */}
        <img
          key={stim.id}
          src={stimulusImageUrl(stim)}
          alt=""
          onError={(e) => (e.currentTarget.style.display = "none")}
        />
        <span className="thumb-fallback">{stim.domain}</span>
      </div>
    </button>
  );
}
