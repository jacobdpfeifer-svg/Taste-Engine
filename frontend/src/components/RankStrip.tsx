import { useCallback, useEffect, useState } from "react";
import { stimulusImageUrl, type Stimulus } from "../api";

export type ChoiceRecord = {
  chosen_id: string;
  rejected_id: string;
  domain: string;
  kind: string;
  weight: number;
};

/** Sort the Spectrum: drag (or nudge) five items into preference order left → right.
 * Done emits every ranked pair: earlier beats later (kind "rank", weight 1). */
export function RankStrip({
  items,
  onDone,
}: {
  items: Stimulus[];
  onDone: (records: ChoiceRecord[]) => void;
}) {
  const [order, setOrder] = useState(items);
  const [dragIdx, setDragIdx] = useState<number | null>(null);

  useEffect(() => {
    setOrder(items);
  }, [items]);

  const move = useCallback((from: number, to: number) => {
    if (from === to || to < 0 || to >= order.length) return;
    setOrder((prev) => {
      const next = [...prev];
      const [item] = next.splice(from, 1);
      next.splice(to, 0, item);
      return next;
    });
  }, [order.length]);

  function handleDone() {
    const domain = order[0]?.domain ?? "cross";
    const records: ChoiceRecord[] = [];
    for (let a = 0; a < order.length; a++) {
      for (let b = a + 1; b < order.length; b++) {
        records.push({
          chosen_id: order[a].id,
          rejected_id: order[b].id,
          domain,
          kind: "rank",
          weight: 1,
        });
      }
    }
    onDone(records);
  }

  return (
    <div className="rank-strip">
      <p className="muted small rank-hint">Most like you ← drag to reorder → least like you</p>
      <div className="rank-row">
        {order.map((stim, idx) => (
          <div
            key={stim.id}
            className={`rank-slot ${dragIdx === idx ? "dragging" : ""}`}
            draggable
            onDragStart={() => setDragIdx(idx)}
            onDragEnd={() => setDragIdx(null)}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              if (dragIdx !== null) move(dragIdx, idx);
              setDragIdx(null);
            }}
          >
            <span className="rank-num">{idx + 1}</span>
            <div className="rank-thumb" aria-label={stim.caption}>
              <img
                src={stimulusImageUrl(stim)}
                alt=""
                draggable={false}
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
              <span className="thumb-fallback">{stim.domain}</span>
            </div>
            <div className="rank-nudge">
              <button type="button" aria-label="Move left" disabled={idx === 0} onClick={() => move(idx, idx - 1)}>
                ←
              </button>
              <button
                type="button"
                aria-label="Move right"
                disabled={idx === order.length - 1}
                onClick={() => move(idx, idx + 1)}
              >
                →
              </button>
            </div>
          </div>
        ))}
      </div>
      <button type="button" className="primary" onClick={handleDone}>
        Done
      </button>
    </div>
  );
}
