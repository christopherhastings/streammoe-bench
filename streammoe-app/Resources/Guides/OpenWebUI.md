# Open WebUI + StreamMoE

StreamMoE replaces Ollama's runtime on port **11434**, keeping the API surface
byte-identical. That means Open WebUI, which already speaks the Ollama admin
API, can point at StreamMoE with zero code changes — you only need to add a
connection and refresh the model dropdown.

## Path 1 — Open WebUI is running in Docker on this Mac

This is the most common install: the `open-webui/open-webui` container runs
inside Docker Desktop and reaches host services through the special DNS name
`host.docker.internal`.

1. Open the Web UI in your browser.
2. Click your avatar → **Admin Panel → Settings → Connections**.
3. Click **Add connection** and pick **Ollama API**.
4. Paste `http://host.docker.internal:11434` into the URL field.
5. Leave the API key field blank.
6. Click **Save**. Open WebUI will list the model advertised by StreamMoE in
   its Models list within 10 seconds; refresh if it doesn't.

Back in the chat, open the model dropdown and pick the model shown in
StreamMoE's "Connect an app → Open WebUI" pane.

## Path 2 — Open WebUI is running natively (Node.js install)

If you installed Open WebUI via `pip install open-webui` or `npm`, it runs on
the Mac host directly and can use plain `localhost`:

- URL: `http://localhost:11434`
- everything else identical to Path 1.

## Path 3 — Open WebUI is on a different computer

When the browser tab is on a laptop but StreamMoE is on your desktop Mac, you
need to expose StreamMoE on your LAN.

1. In StreamMoE's menu bar icon: **Settings → Allow network access**.
2. In Open WebUI's connection URL, use this Mac's LAN IP:
   `http://10.0.0.42:11434` (replace with your actual IP).
3. Keep both devices on the same Wi-Fi / subnet.

## Keep your cloud connection too

Adding StreamMoE as a new connection doesn't remove any existing ones. You
can keep your Anthropic / OpenAI / Groq providers alongside StreamMoE — the
model dropdown will show everything, and you pick per chat.

## Troubleshooting

**"Server connection failed"** — StreamMoE isn't running. Open the menu bar
icon and check the header is green.

**Model doesn't appear in the dropdown** — Refresh the dropdown (click the
chevron), or reload the browser tab. The Ollama API reports models on demand,
so the list catches up after a few seconds.

**Requests hang for 30+ seconds on first prompt** — You didn't turn on
`--moe-eager-load` or `--streammoe-warmup`. Open StreamMoE's Settings and
flip those on; first-prompt TTFT drops from ~2.5s to ~0.25s.
