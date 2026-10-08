"""Visual layer for the dashboard: a Notion-like page (cover, page icon, title, callouts) in a collage-moodboard
palette, with Revi the mascot as the page icon and smooth, eased animations.

Layers, back to front: the page (warm paper / deep forest, with paper grain and slowly drifting colour washes), then
raised "sheets" that hold each tab's content (cream / warm charcoal), then tinted tiles and callouts.

Everything is plain HTML/CSS/SVG rendered with st.markdown (no extra packages). Colours follow Streamlit's light/dark
theme through CSS light-dark(), because Streamlit sets color-scheme on the app container. A small script powers the
light/dark toggle by driving Streamlit's own theme picker. Animations are off for visitors who ask for reduced motion.
"""
import base64
import math
import random
from functools import lru_cache
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

# moodboard palette (decorative: cover, swatches, tints). Text always uses the --text / --mute tokens.
P = dict(blush="#f8ddd3", fog="#ecebe7", forest="#0f3d35", sage="#6fae9f", mint="#bfe3d8", terracotta="#c4552f",
         orange="#e5683a", salmon="#f2937a", pink="#f6b9b0", mustard="#e3b23c", gold="#c9a227", cream="#fdf3ec",
         gray="#6e6e6e")
ACCENT = "#d9623a"  # chart series colour; passes contrast on the light and the dark sheet colour
SURFACE = {"light": "#fdf8f3", "dark": "#191f1d"}  # sheet colour = Streamlit backgroundColor in .streamlit/config.toml
MUTE = {"light": "#56625d", "dark": "#a3b0ab"}

_GRAIN = ("<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence "
          "type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/><feColorMatrix type='saturate' "
          "values='0'/></filter><rect width='100%' height='100%' filter='url(#n)'/></svg>")
GRAIN_URI = "data:image/svg+xml;base64," + base64.b64encode(_GRAIN.encode()).decode()

CSS = """
<style>
.stApp {
  --font: 'Inter', ui-sans-serif, -apple-system, 'Segoe UI', sans-serif;
  --serif: 'Fraunces', Georgia, serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
  --page: light-dark(#f2e8df, #0a1714);
  --card: light-dark(#fdf8f3, #191f1d);
  --text: light-dark(#1d2e2a, #f1ebe3);
  --mute: light-dark(#56625d, #a3b0ab);
  --line: light-dark(#e3d4c8, #2b3532);
  --fill: light-dark(#eadfd5, #1d2925);
  --accent: light-dark(#b84a26, #f08f6f);
  --shadow-c: light-dark(rgba(77, 52, 36, .18), rgba(0, 0, 0, .55));
  --tile-line: light-dark(rgba(15, 61, 53, .07), rgba(255, 255, 255, .06));
  --t-mint: light-dark(#dcefe6, rgba(111, 174, 159, .14));
  --t-blush: light-dark(#f9e0d6, rgba(242, 147, 122, .13));
  --t-mustard: light-dark(#f6e5bd, rgba(227, 178, 60, .13));
  --t-sage: light-dark(#e3ebe6, rgba(191, 227, 216, .07));
  --w1: light-dark(rgba(242, 147, 122, .34), rgba(229, 104, 58, .13));
  --w2: light-dark(rgba(111, 174, 159, .28), rgba(111, 174, 159, .14));
  --w3: light-dark(rgba(227, 178, 60, .24), rgba(227, 178, 60, .07));
  --cream: #fdf3ec; --orange: #e5683a;
  --ease: cubic-bezier(.16, 1, .3, 1);
  --spring: cubic-bezier(.34, 1.56, .64, 1);
  font-feature-settings: "cv11", "ss01"; -webkit-font-smoothing: antialiased; letter-spacing: -0.011em;
}

/* the page: base colour, slowly drifting colour washes, paper grain (all behind the content) */
.stApp.stApp { background-color: var(--page); }
.stApp::before { content: ""; position: fixed; inset: -12%; z-index: 0; pointer-events: none;
  background: radial-gradient(38% 34% at 10% 6%, var(--w1), transparent 70%),
              radial-gradient(34% 42% at 94% 34%, var(--w2), transparent 70%),
              radial-gradient(46% 40% at 55% 104%, var(--w3), transparent 70%);
  animation: drift 38s ease-in-out infinite alternate; }
.stApp::after { content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: .14;
                mix-blend-mode: overlay; background-image: url("__GRAIN__"); }
@keyframes drift { to { transform: translate3d(3%, -2%, 0) scale(1.07); } }
[data-testid="stAppViewContainer"] { z-index: 1; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1100px; padding: 1.25rem 24px 4rem; }
@media (max-width: 640px) { [data-testid="stMainBlockContainer"] { padding: .75rem 16px 3rem; } }

/* sheets: each tab's content sits on a raised card */
.stApp [class*="st-key-sheet"] { background: var(--card); border: 1px solid var(--line); border-radius: 18px;
  padding: 22px 26px 28px; box-shadow: inset 0 1px 0 light-dark(rgba(255, 255, 255, .7), rgba(255, 255, 255, .04)),
  0 24px 48px -32px var(--shadow-c); }
@media (max-width: 640px) { .stApp [class*="st-key-sheet"] { padding: 16px 14px 20px; border-radius: 14px; } }

/* entrance: fade up, eased; 'backwards' so hover transitions still work afterwards */
.reveal { animation: fadeUp .8s var(--ease) var(--d, 0s) backwards; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } }
@keyframes pop { from { opacity: 0; transform: scale(.6); } }

/* cover + page icon (Notion anatomy) */
.cover { height: 190px; border-radius: 16px; overflow: hidden; border: 1px solid var(--line);
         box-shadow: 0 24px 48px -36px var(--shadow-c); animation: fadeUp .9s var(--ease) backwards; }
.cover svg { width: 100%; height: 100%; display: block; }
.cover .cv-a { fill: light-dark(#f8ddd3, #3a2924); }
.cover .cv-b { fill: light-dark(#ecebe7, #1f2d29); }
.cover .cv-c { fill: light-dark(#0f3d35, #134a40); }
.cover .pc { animation: drop 1s var(--spring) calc(var(--i) * 70ms + .15s) backwards;
             transform-box: fill-box; transform-origin: center; }
.cover .fl { animation: float var(--f) ease-in-out infinite alternate; transform-box: fill-box; transform-origin: center; }
@keyframes drop { from { opacity: 0; transform: translateY(-16px) rotate(-5deg) scale(.94); } }
@keyframes float { to { transform: translateY(-3px) rotate(1.4deg); } }
.badge { position: relative; width: 128px; height: 128px; margin: -64px 0 0 32px; transform: rotate(-3deg);
         animation: badgeIn 1.1s var(--spring) .5s backwards; transition: transform .45s var(--ease); }
.badge:hover { transform: rotate(1deg) scale(1.04); }
.badge img { width: 100%; height: 100%; display: block; box-sizing: border-box; border-radius: 14px;
             border: 5px solid var(--cream); background: var(--cream);
             box-shadow: 0 14px 28px -12px var(--shadow-c), 0 0 0 1px rgba(15, 61, 53, .08); }
.badge .tape { position: absolute; z-index: 2; top: -9px; left: 50%; width: 60px; height: 20px; margin-left: -30px;
               background: rgba(227, 178, 60, .75); transform: rotate(5deg); }
@keyframes badgeIn { from { opacity: 0; transform: translateY(-22px) rotate(-14deg) scale(.88); } }
@media (max-width: 640px) {
  .cover { height: 130px; }
  .badge { width: 96px; height: 96px; margin: -48px 0 0 18px; }
}

/* chips (Notion page properties) + the light/dark toggle */
.chips { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 16px 0 0; }
.chip { font-size: 12.5px; color: var(--mute); background: var(--fill); padding: 3px 10px; border-radius: 7px; }
.chip.live { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase;
             color: var(--accent); background: transparent; box-shadow: inset 0 0 0 1px var(--line);
             display: inline-flex; align-items: center; gap: 7px; }
.chip.live i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; animation: breathe 2.4s ease-in-out infinite; }
@keyframes breathe { 50% { opacity: .3; transform: scale(.75); } }
.tt { display: none; position: relative; flex: 0 0 auto; margin-left: auto; width: 58px; height: 30px; padding: 0;
      border-radius: 999px; border: 1px solid var(--line); background: var(--fill); cursor: pointer; }
html.rt-ready .tt { display: inline-block; }
.tt .knob { position: absolute; top: 3px; left: 3px; width: 22px; height: 22px; border-radius: 50%; display: grid;
            place-items: center; color: var(--text); background: var(--card); box-shadow: 0 2px 6px -1px var(--shadow-c);
            transition: transform .6s var(--spring); }
.tt .knob svg { grid-area: 1 / 1; width: 14px; height: 14px; transition: opacity .35s var(--ease), transform .6s var(--spring); }
.tt .moon { opacity: 0; transform: rotate(-70deg) scale(.5); }
html[data-scheme="dark"] .tt .knob { transform: translateX(28px); }
html[data-scheme="dark"] .tt .sun { opacity: 0; transform: rotate(70deg) scale(.5); }
html[data-scheme="dark"] .tt .moon { opacity: 1; transform: none; }
.tt:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
html.rt-theming [data-testid="stMainMenuPopover"] { opacity: 0 !important; pointer-events: none !important; }
::view-transition-old(root), ::view-transition-new(root) { animation: none; mix-blend-mode: normal; }

/* title: words rise and sharpen one by one, then a hand-drawn underline draws itself */
.title { font-family: var(--serif); font-weight: 600; font-size: clamp(32px, 4.6vw, 54px); line-height: 1.08;
         letter-spacing: -0.02em; color: var(--text); margin: 14px 0 10px; }
.title .w { display: inline-block; animation: wordIn .9s var(--ease) calc(.35s + var(--k) * 70ms) backwards; }
@keyframes wordIn { from { opacity: 0; transform: translateY(.35em); filter: blur(6px); } }
.title .mark { position: relative; display: inline-block; white-space: nowrap; }
.title .mark svg { position: absolute; left: -2%; bottom: -.14em; width: 104%; height: .32em; overflow: visible; }
.title .mark path { fill: none; stroke: var(--orange); stroke-width: 5; stroke-linecap: round; vector-effect: non-scaling-stroke;
                    stroke-dasharray: 1; stroke-dashoffset: 0; animation: draw 1.1s var(--ease) 1.05s backwards; }
@keyframes draw { from { stroke-dashoffset: 1; } }
.lead { font-size: 16px; line-height: 1.75; color: var(--text); opacity: .88; max-width: 700px; margin: 0; }
.lead b { font-weight: 600; }

/* tinted blocks: a tint layered over the sheet colour, so they read the same on the page and inside a sheet */
.kpi, .callout, .good .g, [data-testid="stMetric"] {
  background: linear-gradient(var(--tint), var(--tint)), var(--card); border: 1px solid var(--tile-line); }

/* KPI tiles */
@property --i { syntax: '<integer>'; initial-value: 0; inherits: false; }
@property --d { syntax: '<integer>'; initial-value: 0; inherits: false; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 28px 0 8px; }
.kpi { border-radius: 14px; padding: 16px 18px; box-shadow: 0 16px 32px -28px var(--shadow-c);
       transition: transform .4s var(--ease), box-shadow .4s var(--ease); }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 18px 32px -20px var(--shadow-c); }
.kpi .lab { display: flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 500; color: var(--mute); }
.kpi .lab i { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.kpi .num { font-weight: 600; font-size: 32px; line-height: 1.25; letter-spacing: -0.025em; color: var(--text);
            font-variant-numeric: tabular-nums; margin: 6px 0 2px; }
.kpi .num.int::before { counter-reset: i var(--i); content: counter(i) attr(data-suffix);
                        animation: count 1.9s var(--ease) .7s forwards; }
.kpi .num.dec::before { counter-reset: i var(--i) d var(--d); content: counter(i) "." counter(d) attr(data-suffix);
                        animation: count 1.9s var(--ease) .7s forwards; }
@keyframes count { to { --i: var(--ti); --d: var(--td); } }
.kpi .meta { font-size: 12.5px; color: var(--mute); }
@media (max-width: 640px) {
  .kpis { grid-template-columns: 1fr 1fr; gap: 10px; }
  .kpi { padding: 13px 14px; }
  .kpi .num { font-size: 26px; }
}

/* section headings */
.sec { margin: 34px 0 14px; }
.sec .eyebrow { display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 600; letter-spacing: .14em;
                text-transform: uppercase; color: var(--mute); }
.sec .eyebrow i { width: 9px; height: 9px; border-radius: 50%; }
.sec .h2 { font-family: var(--serif); font-weight: 600; font-size: clamp(22px, 2.4vw, 28px); line-height: 1.2;
           letter-spacing: -0.015em; color: var(--text); margin: 6px 0 4px; }
.sec .cap { font-size: 13.5px; color: var(--mute); margin: 0; }

/* callouts: Revi talks */
.callout { display: flex; gap: 12px; align-items: flex-start; padding: 14px 16px; border-radius: 12px; margin: 4px 0; }
.callout .msg { font-size: 15px; line-height: 1.6; padding-top: 1px; color: var(--text); }
.callout .msg b { font-weight: 600; }
.callout .ico { font-size: 20px; line-height: 1.3; }
.revi { position: relative; flex: 0 0 auto; width: 42px; height: 42px; border-radius: 50%; overflow: hidden;
        background: var(--cream); box-shadow: 0 0 0 1px rgba(15, 61, 53, .15); animation: pop .7s var(--spring) .15s backwards; }
.revi img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transform: scale(1.5) translateY(4%); }
.revi .talk { opacity: 0; animation: talk .32s steps(1) .55s 6; }
@keyframes talk { 50% { opacity: 1; } }

/* quotes, praise cards, footer */
.q { border-left: 3px solid var(--orange); padding: 3px 0 3px 16px; margin: 12px 0; font-size: 15px; line-height: 1.65; color: var(--text); }
.good { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
.good .g { --tint: var(--t-sage); display: flex; align-items: baseline; gap: 12px; border-radius: 12px; padding: 12px 14px;
           transition: transform .4s var(--ease); }
.good .g:hover { transform: translateY(-2px); }
.good .pct { font-weight: 600; font-size: 18px; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--text); }
.good .nm { font-size: 14px; color: var(--text); }
.foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 18px; margin-top: 56px; padding-top: 16px;
        border-top: 1px solid var(--line); font-size: 12.5px; color: var(--mute); }
.foot a { color: var(--accent); text-decoration: none; }
.foot a:hover { text-decoration: underline; }
.foot code { font-family: var(--mono); font-size: 11.5px; background: var(--fill); padding: 1px 5px; border-radius: 5px; }

/* Streamlit widgets, restyled to match */
[data-testid="stButtonGroup"] [role="radiogroup"] { display: inline-flex; gap: 4px; padding: 4px; border-radius: 12px;
                                                    background: var(--fill); width: fit-content; max-width: 100%; }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] {
  border: 0; border-radius: 9px; background: transparent; color: var(--mute); padding: 7px 16px; min-height: 0;
  box-shadow: none; transition: background-color .4s var(--ease), color .4s var(--ease), transform .4s var(--ease); }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p { font-size: 14px; font-weight: 500; }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover { color: var(--text); background: transparent; }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:active { transform: scale(.97); }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] {
  background: var(--text); color: var(--page); box-shadow: 0 6px 16px -8px var(--shadow-c); }
[data-testid="stTabs"] [role="tablist"] { gap: 2px; width: fit-content; max-width: 100%; padding: 3px; margin-bottom: 6px;
                                          border-radius: 10px; background: var(--fill); overflow-x: auto; scrollbar-width: none; }
[data-testid="stTabs"] [role="tab"] { height: auto; padding: 5px 14px; border-radius: 7px; color: var(--mute);
                                      transition: background-color .3s var(--ease), color .3s var(--ease), box-shadow .3s var(--ease); }
[data-testid="stTabs"] [role="tab"] p { font-size: 13.5px; font-weight: 500; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--card); color: var(--text);
                                                            box-shadow: 0 1px 3px var(--shadow-c); }
[data-testid="stTabs"] [role="tab"] > div:empty { display: none; }  /* Streamlit's underline indicator */
[data-testid="stTabPanel"] { animation: fadeUp .6s var(--ease) backwards; }
[data-testid="stMetric"] { --tint: var(--t-sage); border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricValue"] { font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stVegaLiteChart"], [data-testid="stDataFrame"] { animation: fadeUp .8s var(--ease) .1s backwards; }

/* paper confetti when you switch apps */
.confetti { position: fixed; inset: 0; z-index: 1000; pointer-events: none; overflow: hidden; }
.confetti .cf { position: absolute; top: -6vh; animation: fall var(--dur) cubic-bezier(.4, .1, .7, 1) var(--del) forwards; }
.confetti .cf > * { display: block; animation: sway calc(var(--dur) / 3) ease-in-out var(--del) infinite alternate; }
.confetti b { width: var(--w); height: var(--h); border-radius: 2px; background: var(--c); }
.confetti .e { font-size: 22px; }
@keyframes fall { 85% { opacity: 1; } to { opacity: 0; transform: translateY(112vh); } }
@keyframes sway { from { transform: translateX(calc(var(--x) * -1)) rotate(calc(var(--r) * -1)); }
                  to { transform: translateX(var(--x)) rotate(var(--r)); } }

@media (prefers-reduced-motion: reduce) {
  .stApp *, .stApp *::before, .stApp::before { animation: none !important; transition: none !important; }
  .kpi .num::before { --i: var(--ti); --d: var(--td); }
  .confetti { display: none; }
}
</style>
""".replace("__GRAIN__", GRAIN_URI)

# Light/dark toggle. Streamlit has no Python API for switching themes, so the toggle clicks Streamlit's own picker in
# the main menu (hidden while it happens) and the new theme spreads out from the toggle as a circle (View Transitions).
# If the menu isn't there, the toggle stays hidden; visitors still get their device's setting.
THEME_JS = """
<script>
(() => {
  if (window.__reviTheme) return;
  window.__reviTheme = true;
  const doc = document, root = doc.documentElement, q = s => doc.querySelector(s);
  const scheme = () => { const a = q('.stApp'); return a && getComputedStyle(a).colorScheme.includes('dark') ? 'dark' : 'light'; };
  const sync = () => {
    root.dataset.scheme = scheme();
    root.classList.toggle('rt-ready', !!q('[data-testid="stMainMenuButton"]'));
    doc.querySelectorAll('.tt').forEach(b => b.setAttribute('aria-pressed', String(root.dataset.scheme === 'dark')));
  };
  // timers, not requestAnimationFrame: frames pause during a view transition and in background tabs
  const until = (test, ms = 1500) => new Promise((ok, fail) => {
    const t0 = Date.now();
    (function tick() { const v = test(); if (v) return ok(v); if (Date.now() - t0 > ms) return fail(); setTimeout(tick, 25); })();
  });
  const switchTo = async target => {
    root.classList.add('rt-theming');
    try {
      q('[data-testid="stMainMenuButton"]').click();
      (await until(() => q(`[data-testid="stMainMenuItem-theme-${target}"]`))).click();
      await until(() => scheme() === target.toLowerCase());
      if (q('[data-testid="stMainMenuPopover"]')) q('[data-testid="stMainMenuButton"]').click();
      await until(() => !q('[data-testid="stMainMenuPopover"]'), 600).catch(() => {});
    } finally {
      root.classList.remove('rt-theming');
      if (doc.activeElement) doc.activeElement.blur();
      sync();
    }
  };
  doc.addEventListener('click', ev => {
    const btn = ev.target.closest('.tt');
    if (!btn || root.classList.contains('rt-theming')) return;
    const target = scheme() === 'dark' ? 'Light' : 'Dark';
    root.dataset.scheme = target.toLowerCase();  // the knob starts sliding straight away
    const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!doc.startViewTransition || calm) { switchTo(target).catch(sync); return; }
    const r = btn.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
    const end = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    const vt = doc.startViewTransition(() => switchTo(target).catch(sync));
    vt.ready.then(() => root.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${end}px at ${x}px ${y}px)`] },
                                     { duration: 750, easing: 'cubic-bezier(.16, 1, .3, 1)', pseudoElement: '::view-transition-new(root)' }))
      .catch(() => {});
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => setTimeout(sync, 60));
  const watch = () => { const a = q('.stApp'); if (a) new MutationObserver(sync).observe(a, { attributes: true, attributeFilter: ['class'] }); };
  sync(); watch();
  setTimeout(sync, 400);
})();
</script>
"""


@lru_cache(maxsize=None)
def asset_uri(name: str) -> str:
    data = (ASSETS / name).read_bytes()
    mime = "image/gif" if name.endswith(".gif") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _html(markup: str) -> None:
    """Render into the main page (st.html isolates each block, so shared CSS wouldn't reach it).
    Lines are stripped (and joined with spaces, which keeps SVG attributes apart) so the Markdown parser never
    mistakes indented HTML for a code block."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def inject_css() -> None:
    _html(CSS)


def theme_script() -> None:
    """Powers the light/dark toggle (call once, at the end of the page)."""
    st.html(THEME_JS, unsafe_allow_javascript=True)


def theme_type() -> str:
    """'light' or 'dark' (for chart colours that must match the sheet colour)."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


# ---------- cover collage (inline SVG, drawn from the moodboard palette) ----------

def _torn(x, y, w, h, rnd, jit=2.2, step=9) -> str:
    """A rectangle with slightly torn edges, as an SVG path."""
    pts = []
    for x0, y0, x1, y1 in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        pts += [(x0 + (x1 - x0) * t / n + rnd.uniform(-jit, jit), y0 + (y1 - y0) * t / n + rnd.uniform(-jit, jit))
                for t in range(n)]
    return "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + " Z"


def _blob(cx, cy, r, rnd) -> str:
    """A paint dab: a circle with a wobbly edge."""
    pts = [(cx + r * (1 + rnd.uniform(-.09, .09)) * math.cos(a), cy + r * (1 + rnd.uniform(-.09, .09)) * math.sin(a))
           for a in (i * 2 * math.pi / 18 for i in range(18))]
    return "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + " Z"


def _flower(cx, cy, r, petal, center) -> str:
    petals = "".join(f'<circle cx="{cx + r * math.cos(a):.1f}" cy="{cy + r * math.sin(a):.1f}" r="{r * .8:.1f}" fill="{petal}"/>'
                     for a in (i * 2 * math.pi / 5 - math.pi / 2 for i in range(5)))
    return petals + f'<circle cx="{cx}" cy="{cy}" r="{r * .55:.1f}" fill="{center}"/>'


def _scribble(x, y, w, rnd, color, lines=3, gap=11) -> str:
    out = []
    for j in range(lines):
        ww = w * rnd.uniform(.6, 1)
        pts = " ".join(f"{x + t:.1f} {y + j * gap + 1.6 * math.sin(t / 4.5 + j):.1f}" for t in range(0, int(ww), 3))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>')
    return "".join(out)


def _pin(cx, cy) -> str:
    return (f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{P["gold"]}"/>'
            f'<circle cx="{cx - 2}" cy="{cy - 2}" r="2.2" fill="#fff" opacity=".6"/>')


@lru_cache(maxsize=None)
def cover_svg() -> str:
    rnd = random.Random(7)
    pieces = [  # (svg, float seconds); drawn in order, so later pieces sit on top
        (f'<g transform="rotate(-6 124 65)"><rect x="60" y="22" width="128" height="86" fill="{P["mint"]}" '
         f'stroke="{P["cream"]}" stroke-width="5"/>{_pin(124, 26)}</g>', 7.5),
        (f'<path d="{_torn(196, 92, 118, 90, rnd)}" fill="{P["sage"]}" transform="rotate(4 255 137)"/>', 9),
        (f'<g transform="rotate(7 376 100)"><rect x="330" y="26" width="92" height="148" rx="3" fill="{P["terracotta"]}"/>'
         f'<rect x="330" y="26" width="92" height="148" rx="3" fill="url(#rug)"/></g>', 10),
        (f'<g transform="rotate(-3 504 66)"><rect x="440" y="18" width="128" height="96" fill="{P["cream"]}"/>'
         f'{_flower(472, 50, 11, P["orange"], P["mustard"])}{_flower(530, 82, 9, P["pink"], P["salmon"])}'
         f'{_flower(540, 38, 7, P["salmon"], P["cream"])}'
         f'<ellipse cx="497" cy="92" rx="11" ry="5" fill="{P["forest"]}" transform="rotate(-30 497 92)"/>'
         f'<ellipse cx="505" cy="40" rx="9" ry="4" fill="{P["sage"]}" transform="rotate(35 505 40)"/></g>', 8),
        (f'<g transform="rotate(-5 648 88)"><rect x="600" y="40" width="96" height="96" fill="{P["mustard"]}"/>'
         f'{_scribble(614, 70, 66, rnd, P["forest"])}'
         f'<rect x="626" y="32" width="44" height="15" fill="#fff" opacity=".55" transform="rotate(-4 648 39)"/></g>', 6.5),
        (f'<path d="{_torn(716, 96, 128, 86, rnd)}" fill="{P["cream"]}" transform="rotate(3 780 139)"/>'
         f'<path d="{_torn(724, 104, 112, 70, rnd, jit=3)}" fill="url(#dots)" transform="rotate(3 780 139)"/>', 11),
        ("".join(f'<path d="{_blob(872, 26 + 34 * k, 12.5, rnd)}" fill="{c}"/>'
                 for k, c in enumerate([P["salmon"], P["forest"], P["orange"], P["blush"], P["gray"]])), 9.5),
        ("".join(f'<ellipse cx="972" cy="44" rx="6" ry="15" fill="{P["mustard"]}" transform="rotate({a} 972 62)"/>'
                 for a in range(0, 360, 30)) + f'<circle cx="972" cy="62" r="10" fill="#3b2a1a"/>', 12),
        (f'<g transform="rotate(-6 1036 140)"><path d="{_torn(992, 102, 88, 78, rnd)}" fill="{P["cream"]}"/>'
         f'{_scribble(1004, 126, 62, rnd, P["forest"], lines=2, gap=14)}</g>', 8.5),
    ]
    body = "".join(f'<g class="pc" style="--i:{i}"><g class="fl" style="--f:{f}s">{svg}</g></g>'
                   for i, (svg, f) in enumerate(pieces))
    return (
        '<svg viewBox="0 0 1100 200" preserveAspectRatio="xMidYMid slice" role="img" aria-label="Collage cover">'
        '<defs>'
        f'<pattern id="dots" width="6" height="6" patternUnits="userSpaceOnUse"><circle cx="3" cy="3" r="1.4" fill="{P["forest"]}"/></pattern>'
        f'<pattern id="rug" width="8" height="8" patternUnits="userSpaceOnUse"><circle cx="4" cy="4" r="1.1" fill="{P["cream"]}" opacity=".28"/></pattern>'
        '</defs>'
        f'<rect class="cv-a" width="1100" height="200" fill="{P["blush"]}"/>'
        f'<polygon class="cv-b" points="560,0 900,0 900,200 470,200" fill="{P["fog"]}"/>'
        f'<rect class="cv-c" x="900" width="200" height="200" fill="{P["forest"]}"/>'
        f'{body}</svg>')


# ---------- page blocks ----------

_TOGGLE = ('<button class="tt" type="button" aria-label="Dark mode" aria-pressed="false" title="Light / dark">'
           '<span class="knob">'
           '<svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
           '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2'
           'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
           '<svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
           'stroke-linejoin="round"><path d="M20.5 14.5A8.5 8.5 0 1 1 9.5 3.5a7 7 0 0 0 11 11z"/></svg>'
           '</span></button>')


def hero(title: str, highlight: str, lead: str, chips: list[str]) -> None:
    """Cover, Revi as the page icon, property chips + theme toggle, the title (highlight = the underlined part), lead."""
    start = title.index(highlight)
    words = title[:start].split()
    spans = [f'<span class="w" style="--k:{k}">{w}</span>' for k, w in enumerate(words)]
    marked = " ".join(f'<span class="w" style="--k:{len(words) + k}">{w}</span>' for k, w in enumerate(highlight.split()))
    underline = ('<svg viewBox="0 0 200 20" preserveAspectRatio="none" aria-hidden="true">'
                 '<path pathLength="1" d="M3 13 C 45 5, 110 3, 197 9"/></svg>')
    chip_html = '<span class="chip live"><i></i>On air</span>' + "".join(f'<span class="chip">{c}</span>' for c in chips)
    _html(f"""
    <div class="cover">{cover_svg()}</div>
    <div class="badge"><span class="tape"></span><img src="{asset_uri('mascot.gif')}" alt="Revi, the mascot"></div>
    <div class="chips reveal" style="--d:.3s">{chip_html}{_TOGGLE}</div>
    <div class="title" role="heading" aria-level="1">{" ".join(spans)} <span class="mark">{marked}{underline}</span></div>
    <p class="lead reveal" style="--d:.75s">{lead}</p>""")


TINTS = {"mint": "var(--t-mint)", "blush": "var(--t-blush)", "mustard": "var(--t-mustard)", "sage": "var(--t-sage)"}


def kpis(items: list[tuple[str, float, str, str, str, str]]) -> None:
    """items: (label, value, kind, meta, tint, dot colour); kind 'int' (585) or 'pct' (0.314 -> 31.4%)."""
    tiles = []
    for k, (label, value, kind, meta, tint, dot) in enumerate(items):
        if kind == "pct":
            pct = round(value * 100, 1)
            whole, dec = int(pct), int(round((pct - int(pct)) * 10))
            num = f'<div class="num dec" data-suffix="%" style="--ti:{whole};--td:{dec}"></div>'
        else:
            num = f'<div class="num int" data-suffix="" style="--ti:{int(value)};--td:0"></div>'
        tiles.append(f'<div class="kpi reveal" style="--d:{.85 + k * .08:.2f}s;--tint:{TINTS[tint]}">'
                     f'<div class="lab"><i style="background:{dot}"></i>{label}</div>{num}<div class="meta">{meta}</div></div>')
    _html(f'<div class="kpis">{"".join(tiles)}</div>')


def section(eyebrow: str, title: str = "", caption: str = "", dot: str = P["sage"]) -> None:
    """Small uppercase eyebrow, then (optionally) a serif heading and a caption."""
    head = f'<div class="h2" role="heading" aria-level="2">{title}</div>' if title else ""
    cap = f'<p class="cap">{caption}</p>' if caption else ""
    _html(f'<div class="sec reveal"><div class="eyebrow"><i style="background:{dot}"></i>{eyebrow}</div>{head}{cap}</div>')


def say(message: str, tint: str = "mint") -> None:
    """Revi says something: a Notion-style callout whose avatar 'talks' for a moment when it appears."""
    _html(f'<div class="callout reveal" style="--tint:{TINTS[tint]}"><span class="revi">'
          f'<img src="{asset_uri("mascot_idle.png")}" alt=""><img class="talk" src="{asset_uri("mascot_talk.png")}" alt="">'
          f'</span><div class="msg">{message}</div></div>')


def note(message: str, icon: str = "💡", tint: str = "sage") -> None:
    _html(f'<div class="callout reveal" style="--tint:{TINTS[tint]}"><span class="ico">{icon}</span>'
          f'<div class="msg">{message}</div></div>')


def quote(text: str) -> None:
    _html(f'<div class="q reveal">{text}</div>')


def good(items: list[tuple[str, str]]) -> None:
    """What's working: (share text, theme) cards."""
    cards = "".join(f'<div class="g reveal" style="--d:{k * .06:.2f}s"><span class="pct">{pct}</span><span class="nm">{name}</span></div>'
                    for k, (pct, name) in enumerate(items))
    _html(f'<div class="good">{cards}</div>')


def footer(left: str, right: str) -> None:
    _html(f'<div class="foot"><span>{left}</span><span>{right}</span></div>')


def confetti(emoji: str, active: bool, n: int = 34) -> None:
    """Paper confetti in the moodboard colours (plus a few app emoji). Always renders a container, so the page keeps
    the same number of elements whether or not it's raining."""
    if not active:
        _html('<div class="confetti"></div>')
        return
    rnd = random.Random()
    colours = [P["orange"], P["salmon"], P["mustard"], P["sage"], P["mint"], P["forest"], P["pink"]]
    drops = []
    for k in range(n):
        inner = (f'<span class="e">{emoji}</span>' if k % 6 == 0 else
                 f'<b style="--c:{rnd.choice(colours)};--w:{rnd.randint(7, 12)}px;--h:{rnd.randint(10, 18)}px"></b>')
        drops.append(f'<span class="cf" style="left:{rnd.uniform(1, 97):.1f}%;--dur:{rnd.uniform(2.6, 4.2):.2f}s;'
                     f'--del:{rnd.uniform(0, .9):.2f}s;--x:{rnd.randint(8, 22)}px;--r:{rnd.randint(25, 70)}deg">{inner}</span>')
    _html(f'<div class="confetti">{"".join(drops)}</div>')
