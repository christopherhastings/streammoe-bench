# StreamMoE Local Validation — Aider Implementation Guide

**Purpose:** Get `anemll-flash-llama.cpp` running a large MoE model with SSD expert streaming, connected to Open WebUI with RAG and tool calling. Prove the existing stack works before building anything new.

**Implementer:** Aider (AI pair programmer) with human oversight
**Hardware:**
- Mac Mini M4 Pro, 64GB unified RAM, 1TB SSD (primary dev machine)
- MacBook Air M3, 24GB unified RAM, 1TB SSD (secondary test machine)

**Aider setup:** Run Aider with Claude Sonnet against the cloned repos. Aider auto-commits every change, giving us a clean git history of the exploration.

```bash
# Install Aider (if not already present)
python -m pip install aider-install
aider-install

# Set API key
export ANTHROPIC_API_KEY=<your-key>
```

---

## Ground Rules for Aider

These instructions go in the Aider session so it understands the constraints.

1. **Read before writing.** Before modifying any file in the Anemll fork, read the relevant source files and READMEs. Use `/read` to add files to context.
2. **Commit messages must describe what was learned**, not just what changed. "Discovered sidecar expects manifest at ./experts/manifest.json" is better than "updated config."
3. **Do not modify the Anemll fork in Phase 1-3.** The first three phases are read-only exploration and infrastructure setup. We only touch Anemll code in Phase 4 if needed.
4. **Log everything.** Every command run, every output observed, every error hit. Redirect output to files.
5. **Test on the small model first.** Never go directly to the 397B model. Validate every step with Qwen3-0.6B or Qwen3.5-35B-A3B first.
6. **Stop and document when something doesn't work.** Write a findings note rather than hacking around the problem. The goal is to understand the system, not force it.

---

## Phase 1: Environment Setup

**Goal:** Both machines have the build toolchain, Aider, Docker, and the Anemll fork compiled.

### Step 1.1: Prerequisites (run on BOTH machines)

Open a terminal and run each check. Record the output.

```bash
# Create a work log
mkdir -p ~/streammoe/logs
LOG=~/streammoe/logs/setup_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== Machine Info ==="
sysctl hw.memsize          # RAM in bytes
system_profiler SPNVMeDataType | grep "Capacity"  # SSD size
sw_vers                    # macOS version
uname -m                   # Architecture (arm64)

echo "=== Toolchain ==="
xcode-select -p            # Xcode CLT path
xcrun -sdk macosx metal --version  # Metal compiler
cmake --version
ninja --version || echo "ninja not found"
python3 --version
python3.11 --version || echo "python3.11 not found"
git --version
docker --version || echo "docker not found"

echo "=== Disk Space ==="
df -h /
```

Install anything missing:

```bash
# Only run what's needed based on the checks above
xcode-select --install                    # If Xcode CLT missing
brew install cmake ninja python@3.11      # If any missing
brew install --cask docker                # If Docker missing
# Then launch Docker Desktop and wait for it to start
```

### Step 1.2: Clone and explore anemll-flash-llama.cpp

```bash
mkdir -p ~/streammoe
cd ~/streammoe

git clone https://github.com/Anemll/anemll-flash-llama.cpp.git
cd anemll-flash-llama.cpp

# Capture repo state
echo "=== Repo Info ==="
git log --oneline -10
git branch -a
git remote -v

# Read the README — this is the most important step
cat README.md | tee ~/streammoe/logs/anemll_readme.txt

# Look for model-specific branches
git branch -a | grep -i "glm\|qwen\|kimi\|gemma\|deepseek"

# Look for run scripts, examples, docs
find . -maxdepth 2 -name "*.md" -o -name "*.sh" -o -name "run*" -o -name "example*" | sort
find . -maxdepth 2 -name "*.py" | sort

# Look for sidecar/slot-bank related code
grep -rl "sidecar\|slot.bank\|expert.*stream\|pread\|moe.*ssd" --include="*.cpp" --include="*.h" --include="*.m" --include="*.mm" . | sort | tee ~/streammoe/logs/sidecar_files.txt
```

**Decision point after reading the README:** The README will tell us one of three things:
- (A) The fork has integrated SSD streaming into llama-server with specific flags → proceed to build
- (B) The fork requires a separate sidecar binary or script to be run alongside → note the architecture
- (C) The fork is a research repo that doesn't expose llama-server at all → we need to check if it even builds llama-server

Record which case it is before proceeding.

### Step 1.3: Build the fork

```bash
cd ~/streammoe/anemll-flash-llama.cpp

cmake -B build \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja \
  2>&1 | tee ~/streammoe/logs/cmake_configure.log

cmake --build build --target llama-server -j$(sysctl -n hw.physicalcpu) \
  2>&1 | tee ~/streammoe/logs/cmake_build.log

# Also build llama-cli for quick testing
cmake --build build --target llama-cli -j$(sysctl -n hw.physicalcpu) \
  2>&1 | tee -a ~/streammoe/logs/cmake_build.log

echo "=== Build Artifacts ==="
ls -lh build/bin/llama-server build/bin/llama-cli 2>/dev/null || echo "Build failed — check logs"
```

If the build fails:
- Save the full error log
- Check if the fork needs different CMake flags (read the README again)
- Check if there are Anemll-specific build targets (e.g., a sidecar binary)
- If the fork doesn't build llama-server at all, that changes the plan

### Step 1.4: Smoke test with a tiny model

```bash
# Download a tiny model for smoke testing
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q4_k_m.gguf \
  --local-dir ~/streammoe/models/

# Start the server
cd ~/streammoe/anemll-flash-llama.cpp
./build/bin/llama-server \
  -m ~/streammoe/models/qwen3-0.6b-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 2048 \
  --jinja \
  2>&1 | tee ~/streammoe/logs/smoke_test_server.log &
SERVER_PID=$!

# Wait for server to be ready
sleep 5
for i in {1..30}; do
  if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "Server ready after ${i} attempts"
    break
  fi
  sleep 2
done

# Test endpoints
echo "=== /v1/models ==="
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool | tee ~/streammoe/logs/smoke_models.json

echo "=== /v1/chat/completions ==="
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
    "max_tokens": 50,
    "temperature": 0
  }' | python3 -m json.tool | tee ~/streammoe/logs/smoke_chat.json

echo "=== Streaming test ==="
curl -s -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Count to 5."}],
    "max_tokens": 50,
    "stream": true
  }' 2>&1 | head -20 | tee ~/streammoe/logs/smoke_stream.txt

echo "=== Tool calling test ==="
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Austin?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 200
  }' | python3 -m json.tool | tee ~/streammoe/logs/smoke_tools.json

# Clean up
kill $SERVER_PID 2>/dev/null
```

**Record what works and what doesn't.** The smoke test validates: model listing, chat completions, streaming, tool calling. All four must work for Open WebUI integration.

---

## Phase 2: Open WebUI Setup

**Goal:** Open WebUI running in Docker, connected to the llama-server.

### Step 2.1: Start Open WebUI

```bash
docker pull ghcr.io/open-webui/open-webui:main

docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e WEBUI_AUTH=false \
  -v ~/streammoe/open-webui-data:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

# Wait for startup
echo "Waiting for Open WebUI..."
for i in {1..60}; do
  if curl -sf http://localhost:3000 > /dev/null 2>&1; then
    echo "Open WebUI ready after ${i} attempts"
    break
  fi
  sleep 2
done

docker logs open-webui 2>&1 | tail -20 | tee ~/streammoe/logs/openwebui_startup.log
```

### Step 2.2: Connect to llama-server

Make sure the llama-server from Phase 1.4 is still running (or restart it).

Then open http://localhost:3000 in a browser and:

1. Go to **Admin Panel** → **Settings** → **Connections**
2. Under **OpenAI API**, click **Add Connection** (or the + button)
3. Set **URL** to `http://host.docker.internal:8080/v1`
4. Leave **API Key** blank or type `none`
5. Click the check/verify button — it should show a green checkmark and list the model
6. Save

### Step 2.3: Validate chat

1. Start a new chat
2. Select the model from the dropdown
3. Send: "Hello, what model are you? Reply in one sentence."
4. Verify streaming response arrives

### Step 2.4: Validate RAG

1. Create a short test document — a text file with some unique facts:

```bash
cat > ~/streammoe/test_doc.txt << 'EOF'
StreamMoE Project Facts (Test Document)

The StreamMoE project was started in April 2026 by a developer in Austin, Texas.
The primary development machine is a Mac Mini M4 Pro with 64GB of unified memory.
The secondary test machine is a MacBook Air M3 with 24GB of unified memory.
The project uses the anemll-flash-llama.cpp fork for SSD expert streaming.
The target model for validation is Qwen3.5-397B-A17B with 512 experts per layer.
The Open WebUI interface runs on port 3000 via Docker.
The llama-server runs natively on port 8080.
EOF
```

2. In Open WebUI, click the attachment/upload button in the chat
3. Upload `test_doc.txt`
4. Ask: "According to the document, what machine has 64GB of memory?"
5. Verify the response references "Mac Mini M4 Pro" from the document

If RAG doesn't work:
- Check Admin Panel → Settings → Documents for embedding configuration
- Open WebUI may need to download a sentence-transformer model on first use
- Check `docker logs open-webui` for embedding-related errors

### Step 2.5: Validate tool calling in Open WebUI

1. Go to **Workspace** → **Tools** in Open WebUI
2. Check what built-in tools are available
3. Try asking the model a question that would trigger a tool (e.g., web search, if available)
4. Document what works

Note: Open WebUI's tool calling support depends on both the model and Open WebUI's tool infrastructure. Some tools are Open WebUI-native (they run inside Open WebUI, not via the model's tool_calls API). Document which kind you're seeing.

---

## Phase 3: MoE Model Exploration

**Goal:** Understand what it takes to run a large MoE model with SSD streaming on the Anemll fork.

### Step 3.1: Read the sidecar code

Start an Aider session in the fork repo to explore the codebase:

```bash
cd ~/streammoe/anemll-flash-llama.cpp
aider --model claude-sonnet-4-6 --api-key $ANTHROPIC_API_KEY
```

In Aider, ask it to help you understand the codebase:

```
/read README.md
/ask What does this README say about SSD expert streaming? What flags or configuration does it mention? Which models are supported?
```

Then explore the sidecar files identified in Step 1.2:

```
/read <files from sidecar_files.txt>
/ask How does this fork handle SSD expert streaming? What configuration does the sidecar runtime read? Where are model architecture constants defined? Are they hardcoded or configurable?
```

Key questions to answer:
- Does the fork read a manifest/config file, or are architecture constants hardcoded?
- What is the expert file format? (per-layer flat binary? single file? GGUF-native?)
- What CLI flags enable SSD streaming mode?
- Is there a packing/preparation tool included in the repo?
- Which models can it actually run today without modification?

### Step 3.2: Try the smallest supported MoE model

Based on what you learned in 3.1, try to run the smallest MoE model the fork supports. If Qwen3.5-35B-A3B is supported:

```bash
# Download (adjust based on what the README says)
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GGUF \
  --include "*Q4_K_M*" \
  --local-dir ~/streammoe/models/qwen35-35b/ \
  2>&1 | tee ~/streammoe/logs/download_qwen35_35b.log

# Run (adjust flags based on what you learned about the fork)
cd ~/streammoe/anemll-flash-llama.cpp
./build/bin/llama-server \
  -m ~/streammoe/models/qwen35-35b/<model_file>.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096 \
  --jinja \
  --flash-attn \
  2>&1 | tee ~/streammoe/logs/qwen35_35b_server.log
```

This model should fit entirely in 64GB RAM, so it tests the fork's llama-server path without needing SSD streaming. If it works, chat through Open WebUI and verify everything still works with a real MoE model.

### Step 3.3: Investigate the 397B model path

If Step 3.1 revealed how SSD streaming is configured, investigate what's needed for the 397B model:

```bash
# Check disk space
df -h /

# How much space does the 397B model need?
# The MLX 4-bit checkpoint is ~200GB
# Additional packed experts may be needed depending on the fork's approach
```

Use Aider to examine the model preparation scripts:

```
/ask Based on what I've read in this codebase, what steps are needed to prepare and run Qwen3.5-397B-A17B with SSD expert streaming? Is there a preparation/packing script? What files does the sidecar runtime expect to find on disk?
```

**Do NOT download the 397B model yet.** First understand the full preparation pipeline. The download alone is 200+GB and takes hours. Only proceed when you know exactly what files are needed and where they should go.

---

## Phase 4: Findings Document

**Goal:** Produce a clear picture of what works, what doesn't, and what's needed next.

After completing Phases 1-3, create this document:

```bash
cat > ~/streammoe/FINDINGS.md << TEMPLATE
# StreamMoE Validation Findings

Date: $(date +%Y-%m-%d)

## Environment

### Mac Mini M4 Pro (64GB)
- macOS version:
- Xcode CLT version:
- Metal compiler version:
- Disk free:

### MacBook Air M3 (24GB)
- macOS version:
- Disk free:
- Tested: yes/no

## anemll-flash-llama.cpp

### Build
- Commit hash:
- Branch:
- Build succeeded: yes/no
- Build time:
- Binary size:

### Supported models (from README/source):
-
-
-

### SSD streaming architecture:
- How it works (brief):
- Config format (manifest? hardcoded? CLI flags?):
- Packing tool included: yes/no
- Expert file format:

### Smoke test (Qwen3-0.6B):
- /v1/models: works/broken
- /v1/chat/completions: works/broken
- Streaming: works/broken
- Tool calling: works/broken

### MoE test (Qwen3.5-35B-A3B or other small MoE):
- Model loads: yes/no
- Chat works: yes/no
- tok/s observed:
- Memory usage:

### 397B model path:
- Download size:
- Preparation steps needed:
- Estimated disk space total:
- SSD streaming flags/config:
- Attempted: yes/no
- Result:

## Open WebUI

- Version:
- Docker image:
- Startup time:
- Connected to llama-server: yes/no
- Chat works: yes/no
- Streaming works: yes/no
- RAG (document upload): works/broken
- Tool calling: works/broken/not tested
- Image support: not available (text-only MoE models)

## Gaps to StreamMoE Goal

### What works today:
-

### What's missing for multi-model support:
-

### What's missing for image support:
-

### What's missing for production RAG:
-

### Recommended next steps (in priority order):
1.
2.
3.

## Raw Logs
All logs are in ~/streammoe/logs/
TEMPLATE
```

Fill this in as you complete each phase. This document is the single input for deciding what to build next.

---

## Aider Workflow Notes

### Starting a session

```bash
cd ~/streammoe/anemll-flash-llama.cpp
aider --model claude-sonnet-4-6 --api-key $ANTHROPIC_API_KEY
```

For exploring (read-only), use `/ask` mode. For making changes, use the default code mode.

### Useful Aider commands for this project

```
/read <file>          # Add file to context without editing
/ask <question>       # Ask about the codebase without making changes
/architect <request>  # Plan changes before implementing
/diff                 # Show uncommitted changes
/undo                 # Undo the last aider commit
/run <command>        # Run a shell command and see output
/lint                 # Run linter on changed files
/test <command>       # Run tests
```

### When working with C/C++ code in the fork

Add the relevant files to context before asking questions:

```
/read ggml/src/ggml-metal/ggml-metal-ops.cpp
/read src/llama-model.cpp
/ask How does this file handle MoE expert tensor loading?
```

### When building the Python tooling (future phases)

```bash
cd ~/streammoe/streammoe  # The Python tooling repo (created later)
aider --model claude-sonnet-4-6 --api-key $ANTHROPIC_API_KEY

# Add test file and source file together
aider src/streammoe/pack/gguf_reader.py tests/unit/test_gguf_reader.py
# Aider will write both the implementation and tests together
```

### Aider with LiteLLM (using your local LM Studio setup)

If you want to use Aider with a local model through your existing LiteLLM proxy:

```bash
aider --model openai/local-model \
  --openai-api-base https://llm.deepsignal.app/v1 \
  --openai-api-key sk-master-key-litellm
```

This routes through your existing Cloudflare tunnel → LiteLLM → LM Studio pipeline. Useful for quick tasks where you don't want to burn API credits, but Claude Sonnet will be more capable for complex codebase exploration.

---

## What NOT To Do

- **Do not start modifying the Anemll fork** until the findings document is complete. Understand what you have before changing it.
- **Do not download the 397B model** until you've confirmed the smaller models work end-to-end.
- **Do not build the Python packing tool** until you know what format the sidecar runtime expects. The Anemll fork may already have preparation scripts.
- **Do not try to add image/vision support.** The MoE models that work with SSD streaming are text-only. Image support is a separate project that requires a different model architecture.
- **Do not fight Docker networking.** If `host.docker.internal` doesn't work, use your Mac's actual LAN IP (`ipconfig getifaddr en0`) instead. Don't spend more than 10 minutes on Docker networking issues.
