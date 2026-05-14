"""
Premium CSS stylesheet for the Churn Intelligence Platform.
Dark futuristic SaaS design with glassmorphism and neon accents.
"""

CUSTOM_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS Variables ───────────────────────────────────────── */
:root {
  --bg-base:      #080C1A;
  --bg-surface:   #0D1224;
  --bg-card:      rgba(17, 24, 50, 0.85);
  --bg-glass:     rgba(255, 255, 255, 0.04);
  --border:       rgba(255, 255, 255, 0.08);
  --border-glow:  rgba(79, 142, 247, 0.35);
  --text-primary: #E2E8F0;
  --text-muted:   #8892A4;
  --accent-blue:  #4F8EF7;
  --accent-violet:#A855F7;
  --accent-cyan:  #06B6D4;
  --accent-amber: #F59E0B;
  --danger:       #EF4444;
  --success:      #22C55E;
  --font-head:    'Sora', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
  --radius-lg:    16px;
  --radius-xl:    24px;
  --shadow-glow:  0 0 40px rgba(79, 142, 247, 0.15);
  --shadow-card:  0 8px 32px rgba(0, 0, 0, 0.4);
}

/* ── Global Reset ────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
}

.stApp {
  background: linear-gradient(135deg, #080C1A 0%, #0D1224 50%, #0A0F20 100%) !important;
  background-attachment: fixed !important;
}

/* ── Hide Streamlit Default Branding ──────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.viewerBadge_container__1QSob { display: none !important; }

/* ── Scrollbar Styling ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }

/* ── Main Content ─────────────────────────────────────────── */
.main .block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1280px !important;
}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A0E1F 0%, #0D1228 100%) !important;
  border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1rem !important;
}

/* Sidebar logo area */
.sidebar-logo {
  text-align: center;
  padding: 1rem 0 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}

.sidebar-logo h2 {
  font-family: var(--font-head) !important;
  font-size: 1.1rem !important;
  font-weight: 700 !important;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0.5rem 0 0;
}

/* Sidebar radio buttons */
.stRadio > div { gap: 0.3rem !important; }
.stRadio label {
  font-family: var(--font-body) !important;
  font-size: 0.88rem !important;
  color: var(--text-muted) !important;
  padding: 0.55rem 0.75rem !important;
  border-radius: 10px !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
}
.stRadio label:hover {
  color: var(--text-primary) !important;
  background: var(--bg-glass) !important;
}

/* ── Typography ───────────────────────────────────────────── */
h1, h2, h3, h4, h5 {
  font-family: var(--font-head) !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.02em !important;
}

.page-title {
  font-family: var(--font-head);
  font-size: 2rem;
  font-weight: 800;
  background: linear-gradient(135deg, #E2E8F0 0%, var(--accent-blue) 60%, var(--accent-violet) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.15;
  margin-bottom: 0.25rem;
}

.page-subtitle {
  font-family: var(--font-body);
  font-size: 0.95rem;
  color: var(--text-muted);
  margin-bottom: 2rem;
}

/* ── KPI Metric Cards ─────────────────────────────────────── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-violet));
  opacity: 0.8;
}

.kpi-card:hover {
  border-color: var(--border-glow);
  box-shadow: var(--shadow-glow), var(--shadow-card);
  transform: translateY(-2px);
}

.kpi-label {
  font-family: var(--font-body);
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.5rem;
}

.kpi-value {
  font-family: var(--font-head);
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
  margin-bottom: 0.25rem;
}

.kpi-delta {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--success);
}

.kpi-delta.negative { color: var(--danger); }

/* ── Feature Cards ────────────────────────────────────────── */
.feature-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  padding: 1.75rem;
  backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card);
  transition: all 0.3s ease;
  height: 100%;
}

.feature-card:hover {
  border-color: var(--border-glow);
  box-shadow: var(--shadow-glow), var(--shadow-card);
  transform: translateY(-3px);
}

.feature-icon {
  font-size: 2rem;
  margin-bottom: 1rem;
  display: block;
}

.feature-title {
  font-family: var(--font-head);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}

.feature-desc {
  font-family: var(--font-body);
  font-size: 0.85rem;
  color: var(--text-muted);
  line-height: 1.6;
}

/* ── Section Divider ──────────────────────────────────────── */
.section-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2rem 0;
}

/* ── Risk Badge ───────────────────────────────────────────── */
.risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1.25rem;
  border-radius: 100px;
  font-family: var(--font-head);
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.risk-low    { background: rgba(34,197,94,0.12);  color: #22C55E; border: 1px solid rgba(34,197,94,0.3);  }
.risk-medium { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
.risk-high   { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3);  }

/* ── Prediction Result Panel ──────────────────────────────── */
.prediction-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  padding: 2rem;
  backdrop-filter: blur(24px);
  box-shadow: var(--shadow-glow), var(--shadow-card);
  margin-top: 1.5rem;
}

.prediction-prob {
  font-family: var(--font-head);
  font-size: 3.5rem;
  font-weight: 800;
  line-height: 1;
}

/* ── Insight Cards ────────────────────────────────────────── */
.insight-card {
  background: linear-gradient(135deg, rgba(79,142,247,0.08), rgba(168,85,247,0.06));
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-blue);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  margin: 0.75rem 0;
  font-family: var(--font-body);
  font-size: 0.9rem;
  color: var(--text-primary);
  line-height: 1.6;
}

.insight-card.danger { border-left-color: var(--danger); }
.insight-card.success { border-left-color: var(--success); }
.insight-card.warning { border-left-color: var(--accent-amber); }

/* ── Streamlit Widget Overrides ───────────────────────────── */
div[data-testid="stMetricValue"] {
  font-family: var(--font-head) !important;
  font-size: 1.8rem !important;
  font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
  font-family: var(--font-body) !important;
  font-size: 0.8rem !important;
  color: var(--text-muted) !important;
}

.stButton > button {
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  border-radius: 12px !important;
  transition: all 0.2s ease !important;
  border: none !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet)) !important;
  color: white !important;
  box-shadow: 0 4px 20px rgba(79,142,247,0.3) !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(79,142,247,0.4) !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label,
.stTextInput label, .stMultiSelect label {
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  color: var(--text-muted) !important;
}

.stDataFrame, [data-testid="stTable"] {
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
}

/* ── Hero Banner ──────────────────────────────────────────── */
.hero-banner {
  background: linear-gradient(135deg,
    rgba(79,142,247,0.12) 0%,
    rgba(168,85,247,0.08) 50%,
    rgba(6,182,212,0.06) 100%
  );
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  padding: 3rem 2.5rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}

.hero-banner::after {
  content: '';
  position: absolute;
  top: -50%; right: -10%;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(79,142,247,0.08) 0%, transparent 70%);
  pointer-events: none;
}

.hero-title {
  font-family: var(--font-head);
  font-size: 2.6rem;
  font-weight: 800;
  background: linear-gradient(135deg, #E2E8F0, var(--accent-blue) 50%, var(--accent-violet));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.15;
  margin-bottom: 0.75rem;
}

.hero-sub {
  font-family: var(--font-body);
  font-size: 1.05rem;
  color: var(--text-muted);
  max-width: 600px;
  line-height: 1.7;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: rgba(79,142,247,0.15);
  border: 1px solid rgba(79,142,247,0.3);
  border-radius: 100px;
  padding: 0.3rem 0.9rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--accent-blue);
  margin-bottom: 1rem;
}

/* ── Table / DataFrame ────────────────────────────────────── */
.model-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  background: var(--bg-card);
  border-radius: var(--radius-lg);
  overflow: hidden;
  border: 1px solid var(--border);
}

.model-table th {
  background: rgba(79,142,247,0.1);
  color: var(--accent-blue);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 0.72rem;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
}

.model-table td {
  padding: 0.7rem 1rem;
  color: var(--text-primary);
  border-bottom: 1px solid var(--border);
}

.model-table tr:last-child td { border-bottom: none; }
.model-table tr:hover td { background: var(--bg-glass); }

/* ── Status Pill ──────────────────────────────────────────── */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.7rem;
  border-radius: 100px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 600;
}

.status-best {
  background: rgba(79,142,247,0.15);
  color: var(--accent-blue);
  border: 1px solid rgba(79,142,247,0.3);
}

/* ── Loading Animation ────────────────────────────────────── */
@keyframes pulse-glow {
  0%, 100% { opacity: 1; box-shadow: 0 0 20px rgba(79,142,247,0.3); }
  50% { opacity: 0.7; box-shadow: 0 0 40px rgba(79,142,247,0.6); }
}

.loading-indicator {
  animation: pulse-glow 2s ease-in-out infinite;
}

/* ── About Section ────────────────────────────────────────── */
.tech-badge {
  display: inline-block;
  background: rgba(168,85,247,0.1);
  border: 1px solid rgba(168,85,247,0.25);
  color: #C084FC;
  border-radius: 8px;
  padding: 0.25rem 0.6rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  margin: 0.2rem;
}

/* ── Plotly Chart Container ───────────────────────────────── */
.js-plotly-plot {
  border-radius: var(--radius-lg) !important;
}

/* ── Stagger animation for cards ──────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

.fade-up-1 { animation: fadeUp 0.5s ease forwards; }
.fade-up-2 { animation: fadeUp 0.5s 0.1s ease forwards; opacity: 0; }
.fade-up-3 { animation: fadeUp 0.5s 0.2s ease forwards; opacity: 0; }
.fade-up-4 { animation: fadeUp 0.5s 0.3s ease forwards; opacity: 0; }

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: var(--bg-card);
  border-radius: 12px;
  padding: 0.25rem;
  border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
  font-family: var(--font-body) !important;
  font-size: 0.85rem !important;
  border-radius: 10px !important;
  color: var(--text-muted) !important;
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(79,142,247,0.2), rgba(168,85,247,0.15)) !important;
  color: var(--text-primary) !important;
}
</style>
"""

# ── Sidebar HTML ─────────────────────────────────────────────────────────────
SIDEBAR_HTML = """
<div class="sidebar-logo">
  <div style="font-size:2rem;">🔮</div>
  <h2>ChurnIQ Platform</h2>
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#4F8EF7; margin-top:0.25rem;">
    v2.4.1 · Enterprise Edition
  </div>
</div>
"""

# ── Divider HTML ─────────────────────────────────────────────────────────────
def section_header(icon: str, title: str, subtitle: str = "") -> str:
    return f"""
<div style="margin-bottom:1.5rem;">
  <div class="page-title">{icon} {title}</div>
  {f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ''}
</div>
"""
