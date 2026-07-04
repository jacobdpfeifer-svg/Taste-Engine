import { useEffect, useState } from "react";
import { stimulusImageUrl, type RecordIn, type Stimulus } from "../api";

type Kept = {
  stimulus: Stimulus;
  weight: number;
};

export function GutCheck({
  items,
  onDone,
}: {
  items: Stimulus[];
  onDone: (records: RecordIn[]) => void;
}) {
  const [index, setIndex] = useState(0);
  const [kept, setKept] = useState<Kept[]>([]);
  const [tossed, setTossed] = useState<Stimulus[]>([]);

  useEffect(() => {
    setIndex(0);
    setKept([]);
    setTossed([]);
  }, [items]);

  function finish(nextKept: Kept[], nextTossed: Stimulus[]) {
    const records: RecordIn[] = [];

    if (nextKept.length > 0 && nextTossed.length > 0) {
      for (const k of nextKept) {
        for (const t of nextTossed) {
          records.push({
            chosen_id: k.stimulus.id,
            rejected_id: t.id,
            domain: k.stimulus.domain,
            kind: "swipe",
            weight: k.weight,
          });
        }
      }
    }

    onDone(records);
  }

  function choose(action: "toss" | "keep" | "love") {
    const current = items[index];
    if (!current) {
      finish(kept, tossed);
      return;
    }

    const nextKept =
      action === "toss"
        ? kept
        : [...kept, { stimulus: current, weight: action === "love" ? 2 : 1 }];
    const nextTossed = action === "toss" ? [...tossed, current] : tossed;
    const nextIndex = index + 1;

    if (nextIndex >= items.length) {
      finish(nextKept, nextTossed);
      return;
    }

    setKept(nextKept);
    setTossed(nextTossed);
    setIndex(nextIndex);
  }

  const current = items[index];
  if (!current) return null;

  return (
    <div className="gut-check">
      <div className="gut-stage">
        <div className="card gut-card" aria-label={current.caption}>
          <div className="thumb">
            <img
              src={stimulusImageUrl(current)}
              alt=""
              draggable={false}
              onError={(e) => (e.currentTarget.style.display = "none")}
            />
          </div>
        </div>
      </div>
      <div className="gut-actions" aria-label={`Item ${index + 1} of ${items.length}`}>
        <button type="button" className="secondary" onClick={() => choose("toss")}>
          Toss
        </button>
        <button type="button" className="primary" onClick={() => choose("keep")}>
          Keep
        </button>
        <button type="button" className="primary love" onClick={() => choose("love")}>
          Love
        </button>
      </div>
    </div>
  );
}
