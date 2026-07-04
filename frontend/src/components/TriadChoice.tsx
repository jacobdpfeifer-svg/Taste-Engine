import { stimulusImageUrl, type Stimulus, type Triad } from "../api";

// Odd-one-out: three options, one click marks the one that feels LEAST like the person.
// Denser signal than a pair (decomposes to two pairwise choices server-side) and reveals
// which dimensions the person sorts on. Still pure choice — no text input, no rating.
export function TriadChoice({
  triad,
  onOdd,
}: {
  triad: Triad;
  onOdd: (odd: Stimulus, others: Stimulus[]) => void;
}) {
  const items = [triad.a, triad.b, triad.c];
  return (
    <div className="triad">
      {items.map((stim) => (
        <Card
          key={stim.id}
          stim={stim}
          onClick={() => onOdd(stim, items.filter((s) => s.id !== stim.id))}
        />
      ))}
    </div>
  );
}

// Image only — no caption (see PairChoice for rationale).
function Card({ stim, onClick }: { stim: Stimulus; onClick: () => void }) {
  return (
    <button className="card" onClick={onClick} aria-label={stim.caption}>
      <div className="thumb">
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
