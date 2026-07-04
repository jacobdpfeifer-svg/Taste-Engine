import type { MetaResult } from "../api";

/** Persistent honesty banner — always visible during the journey. */
export function ProvenanceBanner({ meta }: { meta: MetaResult }) {
  return (
    <div className="provenance-banner" role="note">
      {meta.provenance_banner}
    </div>
  );
}
