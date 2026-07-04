import { useEffect, useRef, useState } from "react";
import {
  api,
  stimulusImageUrl,
  type MetaResult,
  type RecordIn,
  type ReflectResult,
  type Stimulus,
  type ValidationResult,
} from "./api";
import { Constellation } from "./components/Constellation";
import { GutCheck } from "./components/GutCheck";
import { OddOneOut } from "./components/OddOneOut";
import { Portrait } from "./components/Portrait";
import { ProvenanceBanner } from "./components/ProvenanceBanner";
import { RankStrip } from "./components/RankStrip";
import { ReflectBack } from "./components/ReflectBack";
import { Wall } from "./components/Wall";

const STEPS = [
  "gut",
  "wall",
  "odd",
  "reflect",
  "rank",
  "constellation",
  "wall2",
  "reflect2",
  "portrait",
] as const;

type StepId = (typeof STEPS)[number];

const GRID_K: Partial<Record<StepId, number>> = {
  gut: 12,
  wall: 20,
  odd: 4,
  rank: 5,
  constellation: 8,
  wall2: 20,
};

const STEP_HEADLINE: Record<StepId, string> = {
  gut: "What feels like you?",
  wall: "Build your wall",
  odd: "Which is least like you?",
  reflect: "Does this sound right?",
  rank: "Sort the spectrum",
  constellation: "Place what feels close",
  wall2: "Build your wall again",
  reflect2: "Still sound like you?",
  portrait: "Your taste portrait",
};

const DOMAIN = "interior";

function heroUrls(ids: string[], bank: Map<string, Stimulus>): string[] {
  return ids
    .map((id) => {
      const s = bank.get(id);
      return s ? stimulusImageUrl(s) : null;
    })
    .filter((u): u is string => u != null);
}

function preload(items: Stimulus[]) {
  for (const s of items) {
    const img = new Image();
    img.src = stimulusImageUrl(s);
  }
}

export default function App() {
  const [phase, setPhase] = useState<"loading" | "journey" | "error">("loading");
  const [stepIdx, setStepIdx] = useState(0);
  const [meta, setMeta] = useState<MetaResult | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [items, setItems] = useState<Stimulus[]>([]);
  const [reflect, setReflect] = useState<ReflectResult | null>(null);
  const [heroSrcs, setHeroSrcs] = useState<string[]>([]);
  const [validate, setValidate] = useState<ValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const bankRef = useRef<Map<string, Stimulus>>(new Map());
  const started = useRef(false);

  const step = STEPS[stepIdx];

  async function loadStep(idx: number, sid: string) {
    const id = STEPS[idx];
    if (id === "reflect" || id === "reflect2") {
      const r = await api.reflect(sid);
      setReflect(r);
      setHeroSrcs(heroUrls(r.hero_ids, bankRef.current));
      setItems([]);
      return;
    }
    if (id === "portrait") {
      await api.fit(sid);
      setValidate(await api.validate(sid));
      setItems([]);
      return;
    }
    const k = GRID_K[id] ?? 9;
    const { items: gridItems } = await api.grid(sid, k, DOMAIN);
    preload(gridItems);
    setItems(gridItems);
  }

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    (async () => {
      try {
        const [m, stimuli, { session_id }] = await Promise.all([
          api.meta(),
          api.listStimuli(),
          api.createSession(),
        ]);
        bankRef.current = new Map(stimuli.map((s) => [s.id, s]));
        setMeta(m);
        setSessionId(session_id);
        await loadStep(0, session_id);
        setPhase("journey");
      } catch (e) {
        setError(String(e));
        setPhase("error");
      }
    })();
  }, []);

  async function submitAndAdvance(records: RecordIn[]) {
    if (busy) return;
    setBusy(true);
    try {
      if (records.length > 0) {
        await api.postBatch(sessionId, records);
      }
      const next = stepIdx + 1;
      if (next >= STEPS.length) return;
      setStepIdx(next);
      await loadStep(next, sessionId);
    } catch (e) {
      setError(String(e));
      setPhase("error");
    } finally {
      setBusy(false);
    }
  }

  async function confirmReflect() {
    if (busy) return;
    setBusy(true);
    try {
      const next = stepIdx + 1;
      setStepIdx(next);
      await loadStep(next, sessionId);
    } catch (e) {
      setError(String(e));
      setPhase("error");
    } finally {
      setBusy(false);
    }
  }

  if (phase === "error" || error) {
    return (
      <div className="wrap">
        <h1>Something broke</h1>
        <p className="muted">{error}</p>
        <p className="muted">
          Is the backend running? <code>uvicorn app.main:app --reload</code> in <code>backend/</code>.
        </p>
      </div>
    );
  }

  if (phase === "loading" || !meta) {
    return <div className="wrap">Loading…</div>;
  }

  if (step === "portrait" && validate) {
    return (
      <>
        <ProvenanceBanner meta={meta} />
        <Portrait
          result={validate}
          heroSrcs={heroSrcs}
          topTags={reflect?.top_tags ?? []}
          meta={{
            ...meta,
            n_choices: validate.n_choices,
            provenance_banner:
              "Measured on your own held-out choices, not asserted. A preference model, not a diagnosis of you.",
          }}
        />
      </>
    );
  }

  return (
    <>
      <ProvenanceBanner meta={meta} />
      <div className="wrap">
        <p className="journey-progress muted small">
          Step {stepIdx + 1} of {STEPS.length}
        </p>
        <h1>{STEP_HEADLINE[step]}</h1>

        {step === "gut" && items.length > 0 && (
          <GutCheck items={items} onDone={submitAndAdvance} />
        )}

        {(step === "wall" || step === "wall2") && items.length > 0 && (
          <Wall items={items} pickCount={5} onDone={submitAndAdvance} />
        )}

        {step === "odd" && items.length >= 4 && (
          <OddOneOut items={items.slice(0, 4)} onDone={submitAndAdvance} />
        )}

        {(step === "reflect" || step === "reflect2") && reflect && (
          <ReflectBack
            summary={reflect.summary}
            heroSrcs={heroSrcs}
            onConfirm={() => void confirmReflect()}
          />
        )}

        {step === "rank" && items.length >= 5 && (
          <RankStrip items={items.slice(0, 5)} onDone={submitAndAdvance} />
        )}

        {step === "constellation" && items.length >= 3 && (
          <Constellation items={items.slice(0, 8)} onDone={submitAndAdvance} />
        )}

        {busy && step !== "reflect" && step !== "reflect2" && (
          <p className="muted small">Saving…</p>
        )}
      </div>
    </>
  );
}
