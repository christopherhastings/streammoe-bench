# Claude Desktop + StreamMoE

Claude Desktop is Anthropic's official client and only connects to Anthropic's
hosted API. There is no "custom endpoint" setting. That means StreamMoE
cannot be wired into Claude Desktop via configuration alone — you need a
bridge.

## Option A — Use an MCP server

Claude Desktop supports **Model Context Protocol** (MCP) servers. You can
write (or install) an MCP server that forwards tool-use calls to StreamMoE's
OpenAI API and returns the result to Claude.

This won't replace Claude's own model — you're still talking to Claude. What
changes is that Claude can call *your local model* as a tool: for summarizing
private files, running through a draft that shouldn't go to the cloud, or
experimenting with MoE outputs.

Example MCP servers that wrap OpenAI endpoints:
- `mcp-openai-proxy` (GitHub, community)
- `openai-as-tool` (GitHub, community)

Point them at `http://localhost:11434/v1` and register the MCP in Claude
Desktop's **Settings → Developer → Edit Config**.

## Option B — Switch client

If your goal is "talk to my local MoE model in a nice chat UI," Claude
Desktop isn't the right client. Use Open WebUI, LM Studio, or Cursor —
those are one-click with StreamMoE.

## Option C — Ask Anthropic

Claude Desktop may add custom-endpoint support in the future. Keep an eye on
Claude Desktop's release notes.
