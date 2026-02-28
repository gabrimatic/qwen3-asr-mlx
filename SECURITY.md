# Security Policy

## Privacy by Design

Privacy is a core constraint, not a feature toggle.

- **All processing is local.** Inference runs entirely on your Mac via MLX.
- **No network calls** except to download model weights from HuggingFace Hub on first use.
- **No telemetry, no analytics, no cloud.** Zero data leaves your machine after model download.
- **Audio stays on device.** Input audio is never transmitted anywhere.

## Trust Boundaries

| Boundary | Trust Level | Notes |
|----------|-------------|-------|
| User audio | Trusted | Processed locally, never transmitted |
| Model weights | Trusted | Downloaded from HuggingFace Hub (Apache 2.0, by Qwen Team / Alibaba Cloud), cached locally |
| Config / tokenizer files | Trusted | Local filesystem, bundled with model |

No remote trust boundaries during inference. The only network access is the initial `huggingface_hub.snapshot_download` call to fetch model weights.

## Vulnerability Reporting

Report vulnerabilities responsibly:

1. **Do not open a public issue.** Vulnerabilities stay private until a fix ships.
2. Use [GitHub's private vulnerability reporting](https://github.com/gabrimatic/qwen3-asr-mlx/security/advisories/new) to submit.
3. Include:
   - Steps to reproduce
   - Demonstrated impact
   - Suggested fix (if any)

Reports without reproduction steps or demonstrated impact are deprioritized.

Expect acknowledgment within 48 hours.

## Out of Scope

These are not considered vulnerabilities:

- Issues requiring physical access to the machine
- Adversarial audio inputs designed to produce specific transcription outputs
- Model hallucinations or inaccurate transcriptions

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
