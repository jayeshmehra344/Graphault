/**
 * Shared theme, constants, and small UI primitives used by both the
 * single-function analyzer and the repo-scan view.
 *
 * Aesthetic: utilitarian dev-tool. Dark slate, monospace code, one muted accent.
 * No gradients, no rounded-everything, no emoji. Dense and precise.
 */

export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const THRESHOLD = 0.683; // CodeBERT F1-optimal threshold (keep in sync with app.py RISK_THRESHOLD_CODEBERT)

// muted, functional palette — no AI purple
export const C = {
  bg: "#0d1117",
  panel: "#161b22",
  panelAlt: "#1c2128",
  border: "#30363d",
  borderBright: "#3d444d",
  text: "#e6edf3",
  textDim: "#7d8590",
  textFaint: "#484f58",
  accent: "#388bfd",
  green: "#3fb950",
  yellow: "#d29922",
  red: "#f85149",
  mono: "'JetBrains Mono', 'SF Mono', ui-monospace, monospace",
  sans: "'Inter Tight', -apple-system, system-ui, sans-serif",
};

export function scoreColor(score) {
  if (score >= THRESHOLD) return C.red;
  if (score >= THRESHOLD * 0.6) return C.yellow;
  return C.green;
}

export function scoreLabel(score) {
  if (score >= THRESHOLD) return "FLAGGED";
  if (score >= THRESHOLD * 0.6) return "REVIEW";
  return "LOW RISK";
}

export const Label = ({ children }) => (
  <span style={{ fontFamily: C.mono, fontSize: 11, color: C.textDim, letterSpacing: "0.08em", textTransform: "uppercase" }}>
    {children}
  </span>
);

export const PanelHead = ({ children }) => (
  <div
    style={{
      fontFamily: C.mono,
      fontSize: 11,
      color: C.textDim,
      padding: "8px 14px",
      borderBottom: `1px solid ${C.border}`,
      letterSpacing: "0.04em",
    }}
  >
    {children}
  </div>
);

export const Th = ({ children }) => (
  <th style={{ padding: "8px 14px", fontWeight: 500, borderBottom: `1px solid ${C.border}` }}>{children}</th>
);

export const Td = ({ children, style }) => <td style={{ padding: "8px 14px", ...style }}>{children}</td>;

export const Empty = ({ children }) => (
  <div
    style={{
      border: `1px dashed ${C.border}`,
      color: C.textFaint,
      padding: 32,
      fontFamily: C.mono,
      fontSize: 12,
      textAlign: "center",
      lineHeight: 1.6,
    }}
  >
    {children}
  </div>
);

export const ErrorBox = ({ children }) => (
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
    {children}
  </div>
);
