# GPU orchestration (RunPod, SSH-less)

Drives training on a rented RunPod GPU from a machine with no GPU. Code + boot script are
delivered via ntfy.sh attachments; progress/results stream back over an ntfy topic; the pod
self-terminates. No secrets live in these files — credentials are passed as env vars.

## Prerequisites
1. **Network egress allowlist** must permit `api.runpod.io` and `ntfy.sh` (and `kaggle.com`
   for the IEEE run). Set the environment's Network access to Full, or add those hosts.
2. Env vars (do NOT commit these):
   ```
   export RUNPOD_API_KEY=rpa_...      # RunPod account API key
   export KAGGLE_TOKEN=KGAT_...       # Kaggle API token (only for the IEEE run)
   ```

## Run the synthetic relational experiment (self-contained, no external data)
```
RUN_SCRIPT=run_synth.sh MAX_STEPS=6000 python scripts/gpu/launch_pod.py launch synth
# then poll the printed ntfy topic for RESULT_fusion_* / RESULT_memcsa_* / E2E_DONE
python scripts/gpu/launch_pod.py status      # pod desired/last status
python scripts/gpu/launch_pod.py terminate   # kill the pod
```
`launch_pod.py` builds `code.tgz` from the repo (`git archive`) — actually it expects a
`code.tgz` next to it; regenerate with `git archive --format=tar.gz -o scripts/gpu/code.tgz HEAD`.

## Scripts
- `launch_pod.py` — deploy/status/terminate a pod; uploads code+boot to ntfy, passes env knobs
  (`MAX_STEPS`, `VOLUME_ID`/`VOLUME_DC` for a network volume, `DATA_FILE`, `KAGGLE_TOKEN`).
- `run_synth.sh` — on-pod: generate synthetic data → encode → build memory → train the 2×2
  (per_card/relational × no-mem/memory-CSA) → probes. Curl/apt-free bootstrap; quiet retrying
  results channel.
- `run_ieee_mem.sh` — on-pod: fetch IEEE from Kaggle → parse/encode → entity-memory (addr1)
  relational experiment.

## Notes
- Prefer SECURE cloud + the fast-fail POD_UP check (a bad machine that can't reach ntfy is
  detected in ~2 min and killed for pennies).
- A network volume (`VOLUME_ID`) persists the processed data + checkpoints across pods and
  removes the re-download/re-process cost — pin the pod to the volume's datacenter.
