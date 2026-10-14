# Hermes agent API: model routing and end-to-end verification

These scripts exercise the **Hermes agent API**, not a raw inference proxy. They use the same agent, tools, session store and profile-scoped credentials as the gateway. They create synthetic test conversations and small temporary tool-marker files; they do not upload private images.

## Routing

Use `hermes config set`, not hand-edited configuration. Keep `hermes-agent` out of `model_routes` so it resolves the configured default for new conversations. An explicit model selection still uses Hermes logic.

The tested deployment routes are:

| API model ID | Provider route | Backend model |
| --- | --- | --- |
| `hermes-agent` | configured default | current deployment: `gpt-6.1-sol` / `openai-codex` |
| `gpt-6.1-sol` | `openai-codex` | `gpt-6.1-sol` |
| `opus-5.5` | `nous` | `anthropic/claude-opus-5.5` |
| `swift-1.5-qwen3.8-27b-orcarouter` | saved local custom provider | `swift-1.5-qwen3.8-27b-orcarouter` |

Provider catalogs and model availability can change; resolve exact IDs against authenticated providers before applying this example elsewhere. The scripts' expected aliases/backends deliberately match this deployment and should be adjusted for a different deployment.

```bash
hermes config set model.provider openai-codex
hermes config set model.default gpt-6.1-sol
hermes config set gateway.api_server.enabled true
hermes config set gateway.api_server.host 127.0.0.1
hermes config set gateway.api_server.port 8642
hermes config set gateway.api_server.max_concurrent_runs 3
hermes config set gateway.api_server.model_routes '{"gpt-6.1-sol":{"model":"gpt-6.1-sol","provider":"openai-codex"},"opus-5.5":{"model":"anthropic/claude-opus-5.5","provider":"nous"},"swift-1.5-qwen3.8-27b-orcarouter":{"model":"swift-1.5-qwen3.8-27b-orcarouter","provider":"custom:NInfer (local GPU)"}}'
```

The local route requires an existing saved custom provider named `NInfer (local GPU)` with a working `base_url`, model ID and `api_mode: chat_completions`. Its `custom_providers` entry in this deployment uses `http://localhost:11434/v1`. The named provider inherits the transport correctly; generic `provider: custom` can auto-detect a Responses transport whose reasoning fields a local backend rejects. A route-level `api_mode` is not supported by all packaged route parsers.

Do not start competing local models just to populate discovery. Installed abliterated models whose inference server is disabled are **not** exposed by this example. The operator explicitly chose the already-running Swift backend to avoid disturbing occupied GPUs.

Authenticate the cloud providers separately with Hermes's supported login flows. Generate an API credential using a cryptographically secure generator and store `API_SERVER_KEY` privately through Hermes; do not commit the real `.env`, provider auth, tunnel token or API token. Configure a dedicated HTTPS tunnel to the loopback origin, with a 404 catch-all, and retain authentication at the API origin.

Model routes are snapshotted at gateway startup. Restart only the owning idle gateway under its existing supervisor; do not create another poller or stop another user's inference server. On Windows a login-triggered tunnel is not a pre-login service.

## Verify the actual launched source

On managed Windows installations, a source update can regenerate the launcher to load the source checkout rather than the prior packaged workspace. Before deployment and after updates, inspect:

```bash
hermes --print-runtime-command gateway run --external-supervisor
```

A terminal child importing a packaged Python environment does not prove its owning gateway imports that same source. Deploy to the actual launch tree and verify behavior through the real listener. Never mark an image fix active merely because edited files import successfully.

## Execute the acceptance scripts

Run with the **installed Hermes Python environment** (or a test environment containing `python-dotenv` and Pillow). Replace `python` below with that interpreter if necessary. `API_SERVER_KEY` may come from the environment or `<home>/.env`; the scripts never print the credential. `--base-url` takes a valid HTTPS origin **without** `/v1`, embedded credentials, query or fragment; only an optional trailing slash is accepted. Validation happens before reading the key. Authenticated requests use a local urllib opener that refuses cross-origin and HTTPS-to-HTTP redirects before forwarding credentials.

```bash
python scripts/api_acceptance/verify_api.py --home <home> --base-url https://your-api.example.net --output-dir <results>
python scripts/api_acceptance/verify_model_routes.py --home <home> --base-url https://your-api.example.net --output-dir <results>
python scripts/api_acceptance/verify_image_continuity.py --home <home> --base-url https://your-api.example.net --output-dir <results> --phase after --protocol chat --retention-only
python scripts/api_acceptance/verify_image_continuity.py --home <home> --base-url https://your-api.example.net --output-dir <results> --phase after --protocol responses --retention-only
```

Then restart the owning gateway, confirm its new PID and healthy API/other enabled channels, and run:

```bash
python scripts/api_acceptance/verify_image_continuity.py --home <home> --base-url https://your-api.example.net --output-dir <results> --phase restart --protocol chat --retention-only
python scripts/api_acceptance/verify_image_continuity.py --home <home> --base-url https://your-api.example.net --output-dir <results> --phase restart --protocol responses --retention-only
```

The fixture font defaults to Windows Arial Bold; on another OS pass `--font /path/to/a/TrueType-font.ttf`. Run fixture creation sequentially when using the same results directory. `--dns-override` is an **explicit opt-in** public DoH workaround for stale LAN NXDOMAIN; it preserves TLS Host/SNI verification and never changes system DNS/hosts files. Leave it off for ordinary DNS verification.

### What the scripts prove

- Unauthenticated and wrong-token requests are rejected.
- Chat, streaming, Responses and stored-response retrieval work over HTTPS.
- Discovery contains exactly the deployment's expected aliases, not additional backends. Every alias resolves the expected **provider and model** and executes a real Hermes terminal command. Missing provider provenance fails closed, including session-metadata fallback. The answer must match a random marker that is **not in the prompt**, and authenticated tool history must independently contain it.
- Each alias works with ordinary Chat/Responses requests without special `model_options`.
- Images in authenticated history decode to the exact uploaded PNG bytes, not an attachment flag or `[screenshot]` placeholder.
- Follow-up turns correctly read an unmentioned lower-left label without reupload. After a real gateway restart, they read a different unmentioned lower-right label. Before either question, authenticated assistant text/tool arguments/tool results must not already disclose its target; image data bytes are excluded. Missing or truncated history fails closed.
- Chat uses `X-Hermes-Session-Id`; Responses uses `previous_response_id`. Both retain renderable images in authenticated session history.

`--retention-only` isolates durable storage/replay from the separate safe-vision-reference and malformed-image ingress changes. It supplies a constant, **test-only** system instruction (Responses: `instructions`) to prefer native image reading rather than Hermes's usual vision-tool workflow; this instruction is identical across turns to preserve the prefix. It does not disable tools or change the gateway's normal policy. Follow-up/restart prompts explicitly prohibit tools, and authenticated history must show no new tool calls (fresh retention sessions must have none). Without it, the image harness additionally requires authenticated history pairing an actual `vision_analyze` call with an intact fixture data URL or managed attachment handle and a successful result, plus a `400` for malformed base64. Native `already_in_context` acknowledgments and successful native reinjection projections are accepted; failed or unpaired results are not. Do not describe those secondary behaviors as verified from retention-only results.

## Persistence and rollback

User image bytes live in the owning profile's content-addressed `images/` directory. The database stores versioned relative references and ordered model/display parts; searchable text retains `[screenshot]` without base64 in FTS. Back up **both** `state.db` and `images/`. Older screenshot-only rows cannot recover discarded pixels without reupload. Session deletion currently does not garbage-collect shared attachments.

Keep local rollback copies of deployed source files and an SQLite-consistent database backup. A rollback to pre-sidecar code keeps text but cannot interpret new image references. Do not publish private backups or runtime credentials with source changes.
