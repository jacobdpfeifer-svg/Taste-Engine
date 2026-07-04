import { useEffect, useRef, useState } from "react";
import { stimulusImageUrl, type Stimulus } from "../api";
import type { ChoiceRecord } from "./RankStrip";

const PAD = 360;
const THUMB = 56;

type Pos = { id: string; x: number; y: number };

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function initialPositions(items: Stimulus[]): Pos[] {
  const cols = Math.ceil(Math.sqrt(items.length));
  const gap = (PAD - THUMB) / Math.max(cols, 1);
  return items.map((item, i) => ({
    id: item.id,
    x: clamp(12 + (i % cols) * gap, 0, PAD - THUMB),
    y: clamp(12 + Math.floor(i / cols) * gap, 0, PAD - THUMB),
  }));
}

function farthestId(from: Pos, all: Pos[]): string {
  let best = "";
  let maxD = -1;
  for (const p of all) {
    if (p.id === from.id) continue;
    const d = Math.hypot(p.x - from.x, p.y - from.y);
    if (d > maxD) {
      maxD = d;
      best = p.id;
    }
  }
  return best;
}

/** Projective map: drag thumbnails on a pad. Similar items cluster; distant pairs are weak
 * dissimilarity signal (item vs its farthest neighbor, weight 0.5). */
export function Constellation({
  items,
  onDone,
}: {
  items: Stimulus[];
  onDone: (records: ChoiceRecord[]) => void;
}) {
  const [positions, setPositions] = useState<Pos[]>(() => initialPositions(items));
  const padRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: string; ox: number; oy: number } | null>(null);

  useEffect(() => {
    setPositions(initialPositions(items));
  }, [items]);

  function onPointerDown(e: React.PointerEvent, id: string) {
    const pad = padRef.current;
    if (!pad) return;
    const pos = positions.find((p) => p.id === id);
    if (!pos) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const rect = pad.getBoundingClientRect();
    dragRef.current = {
      id,
      ox: e.clientX - rect.left - pos.x,
      oy: e.clientY - rect.top - pos.y,
    };
    e.preventDefault();
  }

  function onPointerMove(e: React.PointerEvent) {
    const drag = dragRef.current;
    const pad = padRef.current;
    if (!drag || !pad) return;
    const rect = pad.getBoundingClientRect();
    const x = clamp(e.clientX - rect.left - drag.ox, 0, PAD - THUMB);
    const y = clamp(e.clientY - rect.top - drag.oy, 0, PAD - THUMB);
    setPositions((prev) => prev.map((p) => (p.id === drag.id ? { ...p, x, y } : p)));
  }

  function onPointerUp() {
    dragRef.current = null;
  }

  function handleDone() {
    const domain = items[0]?.domain ?? "cross";
    const byId = new Map(items.map((s) => [s.id, s]));
    const records: ChoiceRecord[] = positions
      .map((pos) => {
        const rejected = farthestId(pos, positions);
        if (!rejected) return null;
        return {
          chosen_id: pos.id,
          rejected_id: rejected,
          domain: byId.get(pos.id)?.domain ?? domain,
          kind: "constellation",
          weight: 0.5,
        };
      })
      .filter((r): r is ChoiceRecord => r !== null);
    onDone(records);
  }

  return (
    <div className="constellation">
      <p className="muted small rank-hint">
        Drag images — put similar ones close, different ones far apart.
      </p>
      <div
        ref={padRef}
        className="constellation-pad"
        style={{ width: PAD, height: PAD }}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        {items.map((stim) => {
          const pos = positions.find((p) => p.id === stim.id)!;
          return (
            <div
              key={stim.id}
              className="constellation-thumb"
              style={{ left: pos.x, top: pos.y, width: THUMB, height: THUMB }}
              onPointerDown={(e) => onPointerDown(e, stim.id)}
              aria-label={stim.caption}
            >
              <img
                src={stimulusImageUrl(stim)}
                alt=""
                draggable={false}
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
              <span className="thumb-fallback">{stim.domain.slice(0, 3)}</span>
            </div>
          );
        })}
      </div>
      <button type="button" className="primary" onClick={handleDone}>
        Done
      </button>
    </div>
  );
}
