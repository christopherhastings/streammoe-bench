# LM Studio + StreamMoE

LM Studio can either host its own models (port 1234) or talk to a remote
OpenAI-compatible endpoint. StreamMoE is the second kind.

## Setup

1. Open LM Studio → **Chat** tab.
2. Click the model dropdown at the top, then **Add Custom Endpoint** at the
   bottom of the list.
3. **Base URL**: `http://localhost:11434/v1`
4. **API Key**: leave blank, or paste `sk-streammoe` — StreamMoE accepts any
   non-empty value and ignores the content.
5. **Model name**: copy from StreamMoE's Connect-an-app pane.

## When to use which

**Use LM Studio's own hosting** for small models that fit entirely in RAM
and benefit from LM Studio's easy model downloader.

**Use StreamMoE** for large MoE models where you want SSD streaming — LM
Studio's built-in runtime doesn't do sidecar streaming, so a 35B MoE model
that StreamMoE handles in 6 GB resident would need 40+ GB in LM Studio.

## Running both simultaneously

LM Studio's server binds to 1234; StreamMoE binds to 11434. They don't
conflict. You can add LM Studio's endpoint as a second custom endpoint and
switch between them per chat.

## Troubleshooting

**"Cannot connect to endpoint"** — StreamMoE isn't running, or you typed
`localhost:11434` without the `http://` prefix. LM Studio is strict about
protocol scheme.

**Chat responses are blank** — Check StreamMoE's server log for errors; the
most common cause is a chat template that LM Studio sends which StreamMoE
doesn't recognize. Flip **Format**: Universal in LM Studio's endpoint config.
