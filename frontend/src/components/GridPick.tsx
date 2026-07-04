import { useEffect, useState } from "react";
import { stimulusImageUrl, type RecordIn, type Stimulus } from "../api";

export function GridPick({
  items,
  pick,
  onDone,
}: {
  items: Stimulus[];
  pick: number;
  onDone: (records: RecordIn[]) => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    setSelected(new Set());
  }, [items, pick]);

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < pick) next.add(id);
      return next;
    });
  }

  function handleDone() {
    if (selected.size !== pick) return;
    const chosen = items.filter((s) => selected.has(s.id));
    const rejected = items.filter((s) => !selected.has(s.id));
    const records: RecordIn[] = [];

    for (const s of chosen) {
      for (const u of rejected) {
        records.push({
          chosen_id: s.id,
          rejected_id: u.id,
          domain: s.domain,
          kind: "grid",
          weight: 1,
        });
      }
    }

    onDone(records);
  }

  return (
    <div className="grid-pick">
      <div className="choice-grid wall-grid">
        {items.map((stim) => {
          const isSelected = selected.has(stim.id);
          return (
            <button
              key={stim.id}
              type="button"
              className={`card grid-pick-card ${isSelected ? "selected" : ""}`}
              onClick={() => toggle(stim.id)}
              aria-label={stim.caption}
              aria-pressed={isSelected}
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
          );
        })}
      </div>
      {selected.size === pick && (
        <button type="button" className="primary" onClick={handleDone}>
          Done
        </button>
      )}
    </div>
  );
}
