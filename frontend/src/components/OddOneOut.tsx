import { stimulusImageUrl, type RecordIn, type Stimulus } from "../api";

/** Four-way odd-one-out: tap the one that feels least like you. */
export function OddOneOut({
  items,
  onDone,
}: {
  items: Stimulus[];
  onDone: (records: RecordIn[]) => void;
}) {
  function chooseOdd(odd: Stimulus) {
    const shown = items.slice(0, 4);
    if (shown.length !== 4) return;
    const others = shown.filter((s) => s.id !== odd.id);
    onDone(
      others.map((o) => ({
        chosen_id: o.id,
        rejected_id: odd.id,
        domain: o.domain,
        kind: "odd",
        weight: 1,
      })),
    );
  }

  const shown = items.slice(0, 4);
  if (shown.length !== 4) return null;

  return (
    <div>
      <div className="odd-grid">
        {shown.map((stim) => (
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
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
