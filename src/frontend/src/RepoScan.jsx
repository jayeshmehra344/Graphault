import { useState } from "react";
import { API, C, scoreColor, Label, PanelHead, Th, Td, Empty, ErrorBox } from "./shared.jsx";

/**
 * Repo Scan view — walks a local repo path via POST /scan-repo, scores every
 * function, and shows a summary + a table of the riskiest functions.
 */
export default function RepoScan() {
  const [repoPath, setRepoPath] = useState("src");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function scan() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/scan-repo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_path: repoPath }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message.includes("fetch") ? "Cannot reach API at " + API + " — is the server running?" : e.message);
    } finally {
      setLoading(false);
    }
  }

  const rows = result?.functions
    ? [...result.functions].sort((a, b) => b.risk_score - a.risk_score)
    : [];

  return (
    <main
      style={{
        maxWidth: 1180,
        margin: "0 auto",
        padding: 24,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      {/* input row */}
      <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <Label>Repo path</Label>
        <div style={{ display: "flex", gap: 12 }}>
          <input
            value={repoPath}
            onChange={(e) => setRepoPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !loading && repoPath.trim()) scan();
            }}
            spellCheck={false}
            placeholder="path to repo, e.g. src"
            style={{
              flex: 1,
              background: C.panel,
              border: `1px solid ${C.border}`,
              color: C.text,
              fontFamily: C.mono,
              fontSize: 13,
              padding: "10px 14px",
              outline: "none",
            }}
            onFocus={(e) => (e.target.style.borderColor = C.borderBright)}
            onBlur={(e) => (e.target.style.borderColor = C.border)}
          />
          <button
            onClick={scan}
            disabled={loading || !repoPath.trim()}
            style={{
              background: loading ? C.panelAlt : C.accent,
              color: loading ? C.textDim : "#fff",
              border: "none",
              padding: "10px 20px",
              fontFamily: C.mono,
              fontSize: 13,
              fontWeight: 500,
              cursor: loading ? "default" : "pointer",
              letterSpacing: "0.02em",
            }}
          >
            {loading ? "scanning…" : "scan ↵"}
          </button>
        </div>
        {error && <ErrorBox>{error}</ErrorBox>}
      </section>

      {!result && !loading && (
        <Empty>
          Enter a path to a local Python repo or folder (relative to where the API server runs)
          and run a scan to score every function it can parse.
        </Empty>
      )}

      {loading && (
        <Empty>
          Scanning repo — walking every .py file and scoring each function. This takes longer
          than a single-function analysis…
        </Empty>
      )}

      {result && (
        <>
          {/* summary */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: 12,
            }}
          >
            <StatBox label="total functions" value={result.summary.total_functions} />
            <StatBox label="files scanned" value={result.summary.files_scanned} />
            <StatBox label="flagged" value={result.summary.flagged_count} color={C.red} />
            <StatBox label="file parse failures" value={result.summary.file_parse_failures} />
            <StatBox label="graph build failures" value={result.summary.function_graph_failures} />
          </div>

          {/* table */}
          <div style={{ border: `1px solid ${C.border}`, background: C.panel }}>
            <PanelHead>functions · sorted by risk score (desc)</PanelHead>
            {rows.length === 0 ? (
              <Empty>No functions found to score.</Empty>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: C.mono, fontSize: 12 }}>
                <thead>
                  <tr style={{ color: C.textDim, textAlign: "left" }}>
                    <Th>risk score</Th>
                    <Th>function</Th>
                    <Th>file</Th>
                    <Th>line</Th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((f, i) => (
                    <tr
                      key={`${f.file_path}:${f.lineno}:${f.function_name}:${i}`}
                      style={{ borderTop: `1px solid ${C.panelAlt}`, color: C.text }}
                    >
                      <Td>
                        <span style={{ color: scoreColor(f.risk_score), fontWeight: 600 }}>
                          {f.risk_score.toFixed(3)}
                        </span>
                      </Td>
                      <Td>{f.function_name}</Td>
                      <Td style={{ color: C.textDim }}>{f.file_path}</Td>
                      <Td style={{ color: C.textDim }}>{f.lineno}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div style={{ fontFamily: C.mono, fontSize: 11, color: C.textFaint, lineHeight: 1.6 }}>
            Risk scores come from the same baseline GNN as the single-function view (PR-AUC ~0.24).
            Useful for triage and prioritisation — not validated bug-detection. Treat as a starting
            point for review, not ground truth.
          </div>
        </>
      )}
    </main>
  );
}

const StatBox = ({ label, value, color }) => (
  <div style={{ border: `1px solid ${C.border}`, background: C.panel, padding: "12px 14px" }}>
    <Label>{label}</Label>
    <div style={{ fontFamily: C.mono, fontSize: 24, fontWeight: 600, color: color || C.text, marginTop: 4 }}>
      {value}
    </div>
  </div>
);
