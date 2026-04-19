# Cursor + StreamMoE

Cursor supports custom OpenAI-compatible endpoints via **Settings → Models →
Add Model**. StreamMoE exposes the OpenAI API at `/v1/*`, so no translation
layer is needed.

## Setup

1. Open Cursor → **Settings** (⌘,) → **Models**.
2. Scroll to **OpenAI API Key** and paste any non-empty string, for example
   `sk-streammoe`. Cursor validates that the field isn't empty, but does not
   contact OpenAI once you add a custom base URL.
3. Click **Add Model** and paste the exact model id from StreamMoE. The id is
   displayed in the Connect-an-app pane — click the Copy button rather than
   typing, because Cursor requires a byte-exact match.
4. Under **OpenAI Base URL**, enter `http://localhost:11434/v1` and click
   **Verify**. Cursor will request `/v1/models` from StreamMoE and echo back
   a list of available IDs; confirm you see the one you just added.

## Chat vs Agent mode

Cursor routes "Chat" and "Agent" through the same model config, so once the
above works, both modes use StreamMoE. Agent mode tends to send longer
system prompts; if you notice TTFT slip under these longer prompts, turn on
`--streammoe-warmup` and optionally `--moe-keep-warm` in StreamMoE settings.

## "Model not found"

Almost always a case/whitespace mismatch between Cursor's cached display name
and the id StreamMoE actually advertises. Fix:

1. In Cursor's model config, delete the custom model.
2. Open StreamMoE's Connect-an-app → Cursor pane.
3. Click Copy next to **Model name**, then re-add in Cursor.

If the error persists, `curl http://localhost:11434/v1/models` from Terminal
and confirm the id shows up in the JSON response.
