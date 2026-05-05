# Local Storage Layout

This document records the host storage contract for the local vLLM / Ollama lab.
The goal is simple: keep large model, WSL, and serving artifacts off the Windows
system drive.

## Current Layout

Snapshot date: 2026-05-05.

| Component | Location | Notes |
| --- | --- | --- |
| WSL Ubuntu VHD | `D:\WSL\Ubuntu\ext4.vhdx` | `Ubuntu` registry `BasePath` points at `D:\WSL\Ubuntu`. |
| vLLM workspace | `E:\GPTProject2\vLLM` | Accessed from WSL as `/mnt/e/GPTProject2/vLLM`. |
| Ollama models | `E:\ollama` | User and machine `OLLAMA_MODELS` both point here. |
| Ollama program files | `E:\Programs\Ollama` | Windows local-program path is a junction to this target. |
| Ollama runtime/config | `E:\OllamaRuntime` | `.ollama` and local Ollama app-data paths are junctions to this target. |

## Cleanup Performed

- Removed WSL package, temporary, pip, Hugging Face, and vLLM caches that can be
  regenerated.
- Kept the project virtual environment because it is required for the benchmark
  and serving scripts.
- Trimmed the WSL filesystem with `fstrim`.
- Removed Windows pip cache and old user temp files.
- Moved the WSL Ubuntu VHD away from `C:`.
- Moved Ollama program, model, and runtime/config writes away from `C:`.

Final verification snapshot:

| Drive | Free space |
| --- | ---: |
| `C:` | 169 GB |
| `D:` | 144 GB |
| `E:` | 70 GB |

WSL smoke check after the move:

```text
Ubuntu default user: non-root project user
WSL root usage: 12G used
vLLM virtualenv: 9.7G
WSL state after verification: Stopped
```

## Guardrails

- Do not store large model weights, Hugging Face caches, Ollama models, or WSL
  VHDs on `C:`.
- Shut down WSL before moving or maintaining the VHD:

```powershell
wsl.exe --shutdown
wsl.exe --manage Ubuntu --move D:\WSL\Ubuntu
```

- Keep `OLLAMA_MODELS` pinned to a non-`C:` drive:

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_MODELS', 'E:\ollama', 'User')
```

- Prefer project-local benchmark artifacts under `reports/benchmarks/`; do not
  write large ad hoc logs or model caches into `%USERPROFILE%`.
- The current WSL build rejected ordinary sparse VHD enablement and requested an
  unsafe override. Do not force `--allow-unsafe` on this machine unless we have a
  backup and a specific reason to test sparse VHD behavior.
