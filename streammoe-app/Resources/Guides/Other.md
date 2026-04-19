# Other apps + StreamMoE

StreamMoE speaks two HTTP protocols on port 11434:

- **OpenAI-compatible** at `/v1/*` — chat completions, completions,
  embeddings, models list. Most modern tools support this.
- **Ollama-native** at `/api/*` — tags, generate, chat. Used by tools that
  were built specifically for Ollama and never added OpenAI support.

Both run on the same port and share the same model weights. You don't
configure which to expose — StreamMoE serves both at once.

## Which to pick

If the app's setup screen asks for "OpenAI API" or "OpenAI-compatible":
use `http://localhost:11434/v1`.

If it asks for "Ollama" or references pulling a model: use
`http://localhost:11434`.

If the app pre-populates an endpoint like `http://localhost:11434` and asks
you to pick a model, it's hardcoded against Ollama and already works — leave
the URL alone.

## API key field

Always put any non-empty string. `sk-streammoe` works. The value is
ignored; the field exists so apps don't fail their own validation.

## Apps we've tested

- **Open WebUI** — dedicated pane, tested on 0.3.x and 0.4.x.
- **Cursor** — dedicated pane, tested on 0.40+ builds.
- **LM Studio** — dedicated pane, tested on 0.3.x.
- **Continue.dev** — set `apiBase` to `http://localhost:11434/v1` and
  `model` to the StreamMoE id; both Chat and Autocomplete work.
- **Zed** — add a custom provider with `http://localhost:11434/v1`; works
  as of Zed 0.160.
- **Msty** — use the Ollama endpoint; works without configuration.
- **AnythingLLM** — use the Generic OpenAI provider.
- **LibreChat** — use the OpenAI provider with a custom base URL.

If your app isn't on this list and it supports either protocol, it almost
certainly works — tell us via the feedback button if it doesn't.
