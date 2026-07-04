import { stimulusImageUrl, type Stimulus } from "../api";
import type { ChoiceRecord } from "./RankStrip";

/** Four-way odd-one-out: tap the one that feels least like you. */
export function OddOneOut({
  items,
  onDone,
}: {
  items: Stimulus[];
  onDone: (records: ChoiceRecord[]) => void;
}) {
  function chooseOdd(odd: Stimulus) {
    const others = items.filter((s) => s.id !== odd.id);
    onDone(
      others.map((o) => ({
        chosen_id: o.id,
        rejected_id: odd.id,
        domain: odd.domain,
        kind: "odd",
        weight: 1,
      })),
    );
  }

  return (
    <div>
      <p className="muted small">Which feels least like you?</p>
      <div className="odd-grid">
        {items.map((stim) => (
          <button
            key={stim.id}
            type="button"
            className="card"
            onClick={() => chooseOdd(stim)}
            aria-label={stim.caption}
          >
            <div className="thumb">
              <img
                src={stimulusImageUrl(stim)}
                alt=""
                draggable={false}
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
              <span className="thumb-fallback">{stim.domain}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
