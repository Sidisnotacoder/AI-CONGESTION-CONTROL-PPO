"""Builds the two frontend outputs from dashboard_template.html:

- dashboard.html: fetches replay-data.json and opens an EventSource
  for live mode against the Flask backend (dashboard/backend.py) --
  meant to be opened as a normal browser tab pointed at that
  same-origin server, not shared standalone.
- replay_artifact.html: replay_data.json inlined directly into the
  page, all network calls removed, live-mode UI hidden. Zero network
  calls of any kind, satisfying the Claude Artifact tool's strict CSP
  by construction -- this is the file to hand to the Artifact tool or
  screenshot for sharing.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, "static", "dashboard_template.html")
REPLAY_JSON = os.path.join(_HERE, "static", "replay_data.json")
DASHBOARD_OUT = os.path.join(_HERE, "static", "dashboard.html")
ARTIFACT_OUT = os.path.join(_HERE, "static", "replay_artifact.html")

DASHBOARD_LOADER = """
function loadInitialData() {
  fetch('/replay-data.json').then(r => r.json()).then(data => {
    DATA = data;
    init();
    document.getElementById('liveBtn').addEventListener('click', startLive);
  }).catch(err => {
    document.getElementById('liveStatus').textContent = 'Failed to load replay data: ' + err;
  });
}

let liveSource = null;

function startLive() {
  if (liveSource) { liveSource.close(); liveSource = null; }
  const run = state.run || Object.keys(DATA.runs)[0];
  document.getElementById('liveStatus').innerHTML = '<span class="live-dot"></span> connecting...';
  document.getElementById('liveBtn').disabled = true;
  DATA.runs[run] = [];
  state.stepIndex = 0;
  liveSource = new EventSource('/stream?run=' + encodeURIComponent(run));
  liveSource.onmessage = (e) => {
    const row = JSON.parse(e.data);
    DATA.runs[run].push(row);
    state.stepIndex = DATA.runs[run].length - 1;
    render();
    document.getElementById('liveStatus').innerHTML = '<span class="live-dot"></span> live: step ' + state.stepIndex;
  };
  liveSource.addEventListener('done', () => {
    liveSource.close();
    liveSource = null;
    document.getElementById('liveBtn').disabled = false;
    document.getElementById('liveStatus').textContent = 'live run complete';
  });
  liveSource.addEventListener('error', () => {
    if (liveSource) { liveSource.close(); }
    liveSource = null;
    document.getElementById('liveBtn').disabled = false;
    document.getElementById('liveStatus').textContent = 'live run failed -- showing replay data instead';
    // Fall back to replay data in-browser rather than hanging or
    // ending the demo mid-presentation.
    fetch('/replay-data.json').then(r => r.json()).then(data => {
      DATA.runs[run] = data.runs[run] || [];
      render();
    });
  });
}
"""

ARTIFACT_LOADER_TEMPLATE = """
const REPLAY_DATA = %s;

function loadInitialData() {
  DATA = REPLAY_DATA;
  init();
  document.getElementById('liveBtn').style.display = 'none';
}
"""


def build():
    with open(TEMPLATE) as f:
        template = f.read()

    if "/*__DATA_LOADER__*/" not in template:
        raise SystemExit("dashboard_template.html is missing the /*__DATA_LOADER__*/ marker")

    dashboard_html = template.replace("/*__DATA_LOADER__*/", DASHBOARD_LOADER)
    with open(DASHBOARD_OUT, "w") as f:
        f.write(dashboard_html)
    print(f"Wrote {DASHBOARD_OUT}")

    if os.path.exists(REPLAY_JSON):
        with open(REPLAY_JSON) as f:
            replay_data = json.load(f)
        artifact_loader = ARTIFACT_LOADER_TEMPLATE % json.dumps(replay_data)
        artifact_html = template.replace("/*__DATA_LOADER__*/", artifact_loader)
        with open(ARTIFACT_OUT, "w") as f:
            f.write(artifact_html)
        print(f"Wrote {ARTIFACT_OUT} (zero network calls, Artifact-safe)")
    else:
        print(f"Skipped {ARTIFACT_OUT}: {REPLAY_JSON} not found -- run dashboard/data_prep.py first")


if __name__ == "__main__":
    build()
