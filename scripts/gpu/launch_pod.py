#!/usr/bin/env python3
"""Deploy the pragma-ffm GPU E2E on a RunPod on-demand pod, driven from this sandbox.

No SSH: code + bootstrap are delivered via ntfy attachments, progress/results stream back
over an ntfy topic, and the pod self-terminates (trap + 4h watchdog + our API terminate).

  launch [nano|both]   upload files, deploy pod, print pod id + ntfy topic
  status               query pod desired/last status
  terminate            podTerminate now
State in scratchpad/gpu/pod_state.json.
"""
import json, os, sys, pathlib, secrets, subprocess, urllib.request

KEY = os.environ["RUNPOD_API_KEY"]
GQL = f"https://api.runpod.io/graphql?api_key={KEY}"
HERE = pathlib.Path(__file__).parent
STATE = HERE / "pod_state.json"
CODE = HERE / "code.tgz"
BOOT = HERE / os.environ.get("RUN_SCRIPT", "run_pod.sh")

# SECURE first: the COMMUNITY A100-80GB-PCIe machine (g1u1092cqu3r) has broken ntfy egress.
# Prefer cheaper/fast GPUs first — our model is <2GB VRAM so an L40S/A6000 is plenty.
GPU_CANDIDATES = ["NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA A100 80GB PCIe",
                  "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA GeForce RTX 4090"]
CLOUDS = ["SECURE", "COMMUNITY"]
IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"


def _st(**u):
    d = json.loads(STATE.read_text()) if STATE.exists() else {}
    if u: d.update(u); STATE.write_text(json.dumps(d, indent=2))
    return d


def _post(body: dict) -> dict:
    # curl (not urllib): RunPod's WAF 403s the default Python-urllib User-Agent.
    out = subprocess.run(
        ["curl", "-sS", "-m", "60", "-H", "Content-Type: application/json",
         "-A", "Mozilla/5.0", "-d", json.dumps(body), GQL],
        capture_output=True, text=True, timeout=90).stdout
    return json.loads(out)


def _ntfy_upload(path: pathlib.Path, name: str) -> str:
    topic = "pragma-file-" + secrets.token_hex(4)
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=path.read_bytes(),
                                 method="PUT", headers={"Filename": name})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["attachment"]["url"]


def launch():
    scope = sys.argv[2] if len(sys.argv) > 2 else "both"
    code_url = _ntfy_upload(CODE, "code.tgz")
    boot_url = _ntfy_upload(BOOT, "run_pod.sh")
    topic = "pragma-run-" + secrets.token_hex(5)
    print("results topic:", topic)
    print("code:", code_url, "\nboot:", boot_url)

    # curl/apt-free bootstrap: fetch boot script via the image's python (Mozilla UA dodges WAFs).
    pyget = ("import urllib.request as u;"
             f"r=u.Request('{boot_url}',headers={{'User-Agent':'Mozilla/5.0'}});"
             "open('/r.sh','wb').write(u.urlopen(r,timeout=120).read())")
    docker = f"bash -c \"python -c \\\"{pyget}\\\" && bash /r.sh\""
    env = [{"key": "NTFY_TOPIC", "value": topic}, {"key": "CODE_URL", "value": code_url},
           {"key": "SCOPE", "value": scope}, {"key": "RUNPOD_API_KEY", "value": KEY}]
    # optional pre-processed data payload (e.g. IEEE-CIS) shipped via ntfy attachment
    if os.environ.get("DATA_FILE"):
        data_url = _ntfy_upload(pathlib.Path(os.environ["DATA_FILE"]), "data.tgz")
        print("data:", data_url)
        env.append({"key": "DATA_URL", "value": data_url})
    for k in ("MAX_STEPS", "SEQLEN", "FIXED_BS", "FUSION", "MEM", "KAGGLE_TOKEN"):  # optional knobs / creds
        if os.environ.get(k):
            env.append({"key": k, "value": os.environ[k]})
    envstr = ",".join('{key:"%s",value:%s}' % (e["key"], json.dumps(e["value"])) for e in env)

    # optional network volume (persistent storage) -> pin to its datacenter, force SECURE
    vol_id = os.environ.get("VOLUME_ID")
    vol_dc = os.environ.get("VOLUME_DC", "US-WA-1")
    vol_frag = (f'networkVolumeId:{json.dumps(vol_id)},dataCenterId:{json.dumps(vol_dc)},'
                'volumeMountPath:"/workspace",' if vol_id else "")
    clouds = ["SECURE"] if vol_id else CLOUDS

    last = None
    for cloud in clouds:
        for gpu in GPU_CANDIDATES:
            mutation = (
                'mutation{podFindAndDeployOnDemand(input:{'
                f'cloudType:{cloud},gpuCount:1,volumeInGb:0,containerDiskInGb:40,'
                'minVcpuCount:4,minMemoryInGb:32,'
                f'{vol_frag}'
                f'gpuTypeId:{json.dumps(gpu)},name:"pragma-e2e",imageName:{json.dumps(IMAGE)},'
                f'dockerArgs:{json.dumps(docker)},ports:"",'
                f'env:[{envstr}]'
                '}){id imageName machineId}}')
            r = _post({"query": mutation})
            if not r.get("errors") and r.get("data", {}).get("podFindAndDeployOnDemand"):
                pod = r["data"]["podFindAndDeployOnDemand"]
                _st(pod_id=pod["id"], topic=topic, scope=scope, gpu=gpu, cloud=cloud)
                print(f"LAUNCHED pod {pod['id']} gpu='{gpu}' cloud={cloud} machine={pod.get('machineId')}")
                return
            last = r.get("errors")
            print(f"  no stock: {gpu} [{cloud}]")
    print("DEPLOY FAILED, last error:", json.dumps(last, indent=2)); sys.exit(1)


def status():
    pid = _st()["pod_id"]
    r = _post({"query": 'query{pod(input:{podId:"%s"}){id desiredStatus lastStatusChange '
                        'runtime{uptimeInSeconds}}}' % pid})
    print(json.dumps(r, indent=2))


def terminate():
    pid = _st().get("pod_id")
    if pid:
        print(_post({"query": 'mutation{podTerminate(input:{podId:"%s"})}' % pid}))


if __name__ == "__main__":
    {"launch": launch, "status": status, "terminate": terminate}[sys.argv[1]]()
