import { useEffect, useState } from "react";
import { stimulusImageUrl, type Stimulus } from "../api";
import type { ChoiceRecord } from "./RankStrip";

/** Mood-board wall: pick exactly `pickCount` from a larger grid. Each picked beats each unpicked. */
export function Wall({
  items,
  pickCount,
  onDone,
}: {
  items: Stimulus[];
  pickCount: number;
  onDone: (records: ChoiceRecord[]) => void;
}) {
  const [picked, setPicked] = useState<Set<string>>(new Set());

  useEffect(() => {
    setPicked(new Set());
  }, [items]);

  function toggle(id: string) {
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < pickCount) next.add(id);
      return next;
    });
  }

  function handleDone() {
    if (picked.size !== pickCount) return;
    const chosen = items.filter((s) => picked.has(s.id));
    const rejected = items.filter((s) => !picked.has(s.id));
    const domain = items[0].domain;
    const records: ChoiceRecord[] = [];
    for (const a of chosen) {
      for (const b of rejected) {
        records.push({
          chosen_id: a.id,
          rejected_id: b.id,
          domain,
          kind: "wall",
          weight: 1,
        });
      }
    }
    onDone(records);
  }

  return (
    <div className="wall">
      <p className="muted small">
        Pick {pickCount} that belong on your wall ({picked.size}/{pickCount}).
      </p>
      <div className="choice-grid wall-grid">
        {items.map((stim) => (
          <button
            key={stim.id}
            type="button"
            className={`grid-card ${picked.has(stim.id) ? "selected" : ""}`}
            onClick={() => toggle(stim.id)}
            aria-label={stim.caption}
            aria-pressed={picked.has(stim.id)}
          >
            <img
              src={stimulusImageUrl(stim)}
              alt=""
              draggable={false}
              onError={(e) => (e.currentTarget.style.display = "none")}
            />
          </button>
        ))}
      </div>
      <button
        type="button"
        className="primary"
        disabled={picked.size !== pickCount}
        onClick={handleDone}
      >
        Done
      </button>
    </div>
  );
}
