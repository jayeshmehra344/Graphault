import { useState, useRef } from "react";
import { API, THRESHOLD, C, scoreColor, scoreLabel, Label, PanelHead, Th, Td, Empty } from "./shared.jsx";
import RepoScan from "./RepoScan.jsx";

/**
 * Graphault Dashboard — single-file React component (Vite).
 * Connects to FastAPI at http://localhost:8000 (/predict, /explain, /model-info).
 *
 * Aesthetic: utilitarian dev-tool. Dark slate, monospace code, one muted accent.
 * No gradients, no rounded-everything, no emoji. Dense and precise.
 *
 * Setup:
 *   npm create vite@latest graphault-ui -- --template react
 *   put this in src/App.jsx, then: npm install && npm run dev
 *   (font: add the two <link> lines from the comment at bottom to index.html)
 */

const SAMPLE = `def get_user(users, idx):
    if idx <= len(users):
        return users[idx - 1]
    return None`;

export default function App() {
  const [tab, setTab] = useState("single");
  const [code, setCode] = useState(SAMPLE);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredLine, setHoveredLine] = useState(null);
  const taRef = useRef(null);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
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

  // lines that have a flagged node, mapped to their max contribution
  const lineHeat = {};
  if (result?.top_nodes) {
    for (const n of result.top_nodes) {
      if (n.lineno != null) {
        lineHeat[n.lineno] = Math.max(lineHeat[n.lineno] || 0, n.contribution);
      }
    }
  }

  const codeLines = code.split("\n");

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: C.sans }}>
      {/* top bar */}
      <header
        style={{
          borderBottom: `1px solid ${C.border}`,
          padding: "0 24px",
          height: 52,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: C.panel,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 8, height: 8, background: C.accent, transform: "rotate(45deg)" }} />
          <span style={{ fontWeight: 600, letterSpacing: "-0.01em" }}>Graphault</span>
          <span style={{ color: C.textFaint, fontSize: 12, fontFamily: C.mono }}>
            GNN vulnerability detection
          </span>
        </div>
        <div style={{ fontFamily: C.mono, fontSize: 11, color: C.textDim, display: "flex", gap: 16 }}>
          <span>PR-AUC 0.23</span>
          <span>thr {THRESHOLD}</span>
          <span style={{ color: C.green }}>● live</span>
        </div>
      </header>

      {/* tab bar */}
      <nav
        style={{
          borderBottom: `1px solid ${C.border}`,
          padding: "0 24px",
          display: "flex",
          gap: 4,
          background: C.panel,
        }}
      >
        <TabButton active={tab === "single"} onClick={() => setTab("single")}>
          Single Function
        </TabButton>
        <TabButton active={tab === "repo"} onClick={() => setTab("repo")}>
          Repo Scan
        </TabButton>
      </nav>

      {tab === "repo" && <RepoScan />}

      {tab === "single" && (
      <main
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          padding: 24,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        {/* LEFT: input */}
        <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Label>Function source</Label>
          <textarea
            ref={taRef}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck={false}
            style={{
              background: C.panel,
              border: `1px solid ${C.border}`,
              color: C.text,
              fontFamily: C.mono,
              fontSize: 13,
              lineHeight: 1.6,
              padding: 16,
              minHeight: 280,
              resize: "vertical",
              outline: "none",
              tabSize: 4,
            }}
            onFocus={(e) => (e.target.style.borderColor = C.borderBright)}
            onBlur={(e) => (e.target.style.borderColor = C.border)}
          />
          <button
            onClick={analyze}
            disabled={loading || !code.trim()}
            style={{
              background: loading ? C.panelAlt : C.accent,
              color: loading ? C.textDim : "#fff",
              border: "none",
              padding: "10px 16px",
              fontFamily: C.mono,
              fontSize: 13,
              fontWeight: 500,
              cursor: loading ? "default" : "pointer",
              alignSelf: "flex-start",
              letterSpacing: "0.02em",
            }}
          >
            {loading ? "analyzing…" : "analyze ↵"}
          </button>
          {error && (
            <div
              style={{
                border: `1px solid ${C.red}`,
                background: "rgba(248,81,73,0.08)",
                color: C.red,
                padding: "10px 14px",
                fontFamily: C.mono,
                fontSize: 12,
              }}
            >
              {error}
            </div>
          )}
        </section>

        {/* RIGHT: results */}
        <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Label>Analysis</Label>
          {!result && !loading && (
            <Empty>Run an analysis to see the risk score and the AST nodes driving it.</Empty>
          )}
          {result && (
            <>
              {/* score block */}
              <div style={{ border: `1px solid ${C.border}`, background: C.panel, padding: 18 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                  <span
                    style={{
                      fontFamily: C.mono,
                      fontSize: 42,
                      fontWeight: 600,
                      color: scoreColor(result.risk_score),
                      lineHeight: 1,
                    }}
                  >
                    {result.risk_score.toFixed(3)}
                  </span>
                  <span
                    style={{
                      fontFamily: C.mono,
                      fontSize: 12,
                      fontWeight: 600,
                      color: scoreColor(result.risk_score),
                      border: `1px solid ${scoreColor(result.risk_score)}`,
                      padding: "3px 8px",
                      letterSpacing: "0.08em",
                    }}
                  >
                    {scoreLabel(result.risk_score)}
                  </span>
                </div>
                {/* bar */}
                <div style={{ marginTop: 14, height: 4, background: C.panelAlt, position: "relative" }}>
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${result.risk_score * 100}%`,
                      background: scoreColor(result.risk_score),
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      left: `${THRESHOLD * 100}%`,
                      top: -3,
                      bottom: -3,
                      width: 1,
                      background: C.textDim,
                    }}
                    title={`threshold ${THRESHOLD}`}
                  />
                </div>
                <div style={{ fontFamily: C.mono, fontSize: 11, color: C.textFaint, marginTop: 6 }}>
                  risk score · vertical mark = decision threshold ({THRESHOLD})
                </div>
              </div>

              {/* code with highlighting */}
              <div style={{ border: `1px solid ${C.border}`, background: C.panel }}>
                <PanelHead>source · flagged lines highlighted</PanelHead>
                <div style={{ fontFamily: C.mono, fontSize: 13, lineHeight: 1.7 }}>
                  {codeLines.map((line, i) => {
                    const ln = i + 1;
                    const heat = lineHeat[ln];
                    return (
                      <div
                        key={i}
                        style={{
                          display: "flex",
                          background: heat ? `rgba(248,81,73,${0.06 + heat * 0.14})` : "transparent",
                          borderLeft: heat
                            ? `2px solid rgba(248,81,73,${0.4 + heat * 0.6})`
                            : "2px solid transparent",
                        }}
                      >
                        <span
                          style={{
                            width: 38,
                            textAlign: "right",
                            paddingRight: 12,
                            color: C.textFaint,
                            userSelect: "none",
                            flexShrink: 0,
                          }}
                        >
                          {ln}
                        </span>
                        <span style={{ whiteSpace: "pre", paddingRight: 16 }}>{line || " "}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* top nodes table */}
              <div style={{ border: `1px solid ${C.border}`, background: C.panel }}>
                <PanelHead>influential AST nodes · gradient saliency</PanelHead>
                <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: C.mono, fontSize: 12 }}>
                  <thead>
                    <tr style={{ color: C.textDim, textAlign: "left" }}>
                      <Th>node</Th>
                      <Th>line</Th>
                      <Th>contribution</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.top_nodes.slice(0, 8).map((n) => (
                      <tr
                        key={n.node_index}
                        onMouseEnter={() => setHoveredLine(n.lineno)}
                        onMouseLeave={() => setHoveredLine(null)}
                        style={{ borderTop: `1px solid ${C.panelAlt}`, color: C.text }}
                      >
                        <Td>{n.node_type}</Td>
                        <Td style={{ color: C.textDim }}>{n.lineno ?? "—"}</Td>
                        <Td>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 80, height: 3, background: C.panelAlt }}>
                              <div
                                style={{
                                  width: `${n.contribution * 100}%`,
                                  height: "100%",
                                  background: C.accent,
                                }}
                              />
                            </div>
                            <span style={{ color: C.textDim }}>{n.contribution.toFixed(3)}</span>
                          </div>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ fontFamily: C.mono, fontSize: 11, color: C.textFaint, lineHeight: 1.6 }}>
                Saliency shows which nodes most affected the score. Diagnostic, not validated
                bug-localization — interpret alongside the score, not as ground truth.
              </div>
            </>
          )}
        </section>
      </main>
      )}
    </div>
  );
}

const TabButton = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    style={{
      background: "transparent",
      border: "none",
      borderBottom: active ? `2px solid ${C.accent}` : "2px solid transparent",
      color: active ? C.text : C.textDim,
      fontFamily: C.mono,
      fontSize: 12,
      fontWeight: 500,
      padding: "12px 4px",
      margin: "0 20px 0 0",
      cursor: "pointer",
      letterSpacing: "0.04em",
    }}
  >
    {children}
  </button>
);

/*
Add to index.html <head> for the fonts:
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
*/
