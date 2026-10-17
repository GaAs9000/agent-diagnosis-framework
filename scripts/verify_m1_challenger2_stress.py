#!/usr/bin/env python3
"""
scripts/verify_m1_challenger2_stress.py

Empirical stress verification harness for Milestone 1:
AML Server Resource Limits, GPU Stability, Process Lifecycle & Tunnel Resilience.
"""

import concurrent.futures
import json
import os
import subprocess
import time
import traceback
import urllib.request
import urllib.error
import sys

RECEIPT_PATH = "scripts/challenger2_stress_receipt.json"
SOCKET_PATH = "/Users/jiashen/.ssh/sockets/aml-b-8001.sock"
BASE_URL = "http://127.0.0.1:8001/v1"


def run_cmd(cmd, timeout=15):
    """Run a local shell command and return output."""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def run_ssh(host, remote_cmd, timeout=15):
    """Run an SSH command with BatchMode and ConnectTimeout."""
    full_cmd = f"ssh -o ConnectTimeout=8 -o BatchMode=yes {host} \"{remote_cmd}\""
    return run_cmd(full_cmd, timeout=timeout)


def audit_aml_b_gpu():
    print("[1/5] Auditing aml-b GPU 4 allocation & user GPU limits...")
    # 1. Query GPU memory
    rc, stdout, stderr = run_ssh("aml-b", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    assert rc == 0, f"nvidia-smi failed on aml-b: {stderr}"
    
    lines = stdout.splitlines()
    gpu_stats = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            idx = int(parts[0])
            gpu_stats[idx] = {
                "name": parts[1],
                "used_mib": int(parts[2]),
                "total_mib": int(parts[3]),
                "util_pct": int(parts[4]),
            }
    
    gpu4 = gpu_stats.get(4)
    assert gpu4 is not None, "GPU 4 not found in nvidia-smi output"
    print(f"  - GPU 4 ({gpu4['name']}): {gpu4['used_mib']} MiB / {gpu4['total_mib']} MiB used ({gpu4['util_pct']}% util)")
    assert 70000 <= gpu4["used_mib"] <= 95000, f"GPU 4 memory {gpu4['used_mib']} MiB out of expected ~81GB allocation"

    # 2. Query compute processes across all GPUs
    rc, stdout, _ = run_ssh("aml-b", "nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader")
    assert rc == 0
    compute_apps = stdout.splitlines()
    
    # 3. Query all processes of jiasheren2
    rc, stdout, _ = run_ssh("aml-b", "ps -u jiasheren2 -o pid,cmd")
    assert rc == 0
    jiasheren2_pids = set()
    for line in stdout.splitlines()[1:]:
        p = line.strip().split(maxsplit=1)
        if p and p[0].isdigit():
            jiasheren2_pids.add(int(p[0]))

    # Check which GPUs jiasheren2 processes occupy
    user_gpu_pids = []
    for line in compute_apps:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1].isdigit():
            pid = int(parts[1])
            if pid in jiasheren2_pids:
                user_gpu_pids.append({"pid": pid, "bus_id": parts[0], "process": parts[2]})

    print(f"  - jiasheren2 GPU processes on aml-b: {user_gpu_pids}")
    assert len(user_gpu_pids) == 1, f"Expected exactly 1 GPU process on aml-b for jiasheren2, found {len(user_gpu_pids)}"
    
    # Verify Slurm queue
    rc, stdout, _ = run_ssh("aml-b", "squeue -u jiasheren2 -h")
    assert rc == 0
    slurm_jobs = [l for l in stdout.splitlines() if l.strip()]
    print(f"  - aml-b Slurm active jobs: {len(slurm_jobs)}")
    assert len(slurm_jobs) == 0, f"Unexpected Slurm jobs found on aml-b: {slurm_jobs}"

    return {
        "gpu4_memory_used_mib": gpu4["used_mib"],
        "gpu4_memory_total_mib": gpu4["total_mib"],
        "gpu4_utilization_pct": gpu4["util_pct"],
        "user_gpu_processes": user_gpu_pids,
        "slurm_jobs_count": len(slurm_jobs),
    }


def audit_aml_a_isolation():
    print("[2/5] Auditing aml-a isolation (verify zero GPUs and zero Slurm jobs held)...")
    # 1. Check Slurm queue
    rc, stdout, stderr = run_ssh("aml-a", "squeue -u jiasheren2 -h")
    assert rc == 0, f"squeue failed on aml-a: {stderr}"
    slurm_jobs_a = [l for l in stdout.splitlines() if l.strip()]
    print(f"  - aml-a Slurm active jobs: {len(slurm_jobs_a)}")
    assert len(slurm_jobs_a) == 0, f"Unexpected Slurm jobs found on aml-a: {slurm_jobs_a}"

    # 2. Check GPU processes on aml-a
    rc, stdout, stderr = run_ssh("aml-a", "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    assert rc == 0, f"nvidia-smi failed on aml-a: {stderr}"
    rc2, stdout2, _ = run_ssh("aml-a", "ps -u jiasheren2 -o pid")
    assert rc2 == 0
    j_pids = {int(l.strip()) for l in stdout2.splitlines()[1:] if l.strip().isdigit()}

    aml_a_gpu_pids = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            pid = int(parts[0])
            if pid in j_pids:
                aml_a_gpu_pids.append(pid)

    print(f"  - aml-a jiasheren2 GPU processes: {aml_a_gpu_pids}")
    assert len(aml_a_gpu_pids) == 0, f"Forbidden: jiasheren2 is using GPUs on aml-a: {aml_a_gpu_pids}"

    return {
        "aml_a_slurm_jobs_count": len(slurm_jobs_a),
        "aml_a_gpu_pids_count": len(aml_a_gpu_pids),
    }


def audit_aml_b_system_load():
    print("[3/5] Auditing CPU thread limits, load average, and process lifecycle on aml-b...")
    # 1. System uptime & load
    rc, stdout, _ = run_ssh("aml-b", "uptime")
    assert rc == 0
    print(f"  - aml-b uptime & load: {stdout}")

    # 2. VLLM process hierarchy and thread limits
    rc, stdout, _ = run_ssh("aml-b", "cat /home/jiasheren2/logs/vllm-qwen36.pid")
    assert rc == 0
    vllm_pid = stdout.strip()
    assert vllm_pid.isdigit(), f"Invalid PID: {vllm_pid}"

    rc, stdout, _ = run_ssh("aml-b", f"ps -p {vllm_pid} -o pid,ppid,stat,%cpu,%mem,etime,cmd --no-headers")
    assert rc == 0 and stdout, f"PID {vllm_pid} is not running!"
    print(f"  - vLLM launcher process: {stdout}")

    # Check environment variables of the vLLM engine
    rc, stdout, _ = run_ssh("aml-b", f"tr '\\0' '\\n' < /proc/{vllm_pid}/environ | grep -E '^(OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|CUDA_VISIBLE_DEVICES)='")
    assert rc == 0
    env_vars = dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)
    print(f"  - Thread & Device env vars: {env_vars}")

    omp_t = int(env_vars.get("OMP_NUM_THREADS", 999))
    mkl_t = int(env_vars.get("MKL_NUM_THREADS", 999))
    openblas_t = int(env_vars.get("OPENBLAS_NUM_THREADS", 999))
    cuda_dev = env_vars.get("CUDA_VISIBLE_DEVICES")

    assert omp_t <= 64, f"OMP_NUM_THREADS ({omp_t}) exceeds limit of 64"
    assert mkl_t <= 64, f"MKL_NUM_THREADS ({mkl_t}) exceeds limit of 64"
    assert openblas_t <= 64, f"OPENBLAS_NUM_THREADS ({openblas_t}) exceeds limit of 64"
    assert cuda_dev == "4", f"CUDA_VISIBLE_DEVICES is '{cuda_dev}', expected '4'"

    # 3. Storage directory check
    rc, stdout, _ = run_ssh("aml-b", "ls -ld /home/jiasheren2/hf_cache /home/jiasheren2/logs")
    assert rc == 0
    print("  - Storage directories verified under /home/jiasheren2")

    return {
        "vllm_pid": int(vllm_pid),
        "env_threads": {
            "OMP_NUM_THREADS": omp_t,
            "MKL_NUM_THREADS": mkl_t,
            "OPENBLAS_NUM_THREADS": openblas_t,
        },
        "cuda_visible_devices": cuda_dev,
    }


def audit_tunnel_socket():
    print("[4/5] Auditing persistent SSH tunnel control socket...")
    assert os.path.exists(SOCKET_PATH), f"Tunnel socket does not exist at {SOCKET_PATH}"
    
    rc, stdout, stderr = run_cmd(f"ssh -S {SOCKET_PATH} -O check aml-b")
    combined = (stdout + " " + stderr).strip()
    assert rc == 0, f"Socket check failed: {stderr}"
    print(f"  - Socket check result: {combined}")
    assert "Master running" in combined, f"Unexpected socket status: {combined}"

    return {
        "socket_path": SOCKET_PATH,
        "socket_check": combined,
        "status": "HEALTHY",
    }


def stress_test_tunnel_concurrency():
    print("[5/5] Stress-testing tunnel under concurrent requests & checking post-stress stability...")
    
    def fetch_model():
        t0 = time.time()
        req = urllib.request.Request(f"{BASE_URL}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.time() - t0) * 1000
            return resp.status, data, elapsed_ms

    def send_chat():
        payload = {
            "model": "Qwen/Qwen3.6-27B",
            "messages": [
                {"role": "system", "content": "You are a concise calculator."},
                {"role": "user", "content": "What is 50 + 50? Answer with only the number."}
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        t0 = time.time()
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.time() - t0) * 1000
            return resp.status, data, elapsed_ms

    # 1. Run 10 parallel GET requests to /v1/models
    print("  - Running 10 concurrent /v1/models queries...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_model) for _ in range(10)]
        results_get = [f.result() for f in concurrent.futures.as_completed(futures)]

    for status, data, elapsed in results_get:
        assert status == 200, f"Expected 200, got {status}"
        model_ids = [m["id"] for m in data["data"]]
        assert "Qwen/Qwen3.6-27B" in model_ids

    latencies_get = [r[2] for r in results_get]
    print(f"    Passed 10 concurrent requests. Min: {min(latencies_get):.1f}ms, Max: {max(latencies_get):.1f}ms, Mean: {sum(latencies_get)/len(latencies_get):.1f}ms")

    # 2. Run 3 parallel completion requests
    print("  - Running 3 concurrent /v1/chat/completions queries...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(send_chat) for _ in range(3)]
        results_chat = [f.result() for f in concurrent.futures.as_completed(futures)]

    for status, data, elapsed in results_chat:
        assert status == 200, f"Expected 200, got {status}"
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        assert "100" in content, f"Unexpected response content: {content}"

    latencies_chat = [r[2] for r in results_chat]
    print(f"    Passed 3 concurrent chat completions. Mean latency: {sum(latencies_chat)/len(latencies_chat):.1f}ms")

    # 3. Re-verify socket health after stress
    rc, stdout, stderr = run_cmd(f"ssh -S {SOCKET_PATH} -O check aml-b")
    combined = (stdout + " " + stderr).strip()
    assert rc == 0, f"Socket degraded after load: {stderr}"
    print(f"  - Post-stress socket health: {combined}")

    return {
        "concurrent_get_count": 10,
        "concurrent_get_latencies_ms": latencies_get,
        "concurrent_chat_count": 3,
        "concurrent_chat_latencies_ms": latencies_chat,
        "post_stress_socket_status": combined,
    }


def main():
    print("================================================================")
    print("CHALLENGER 2: AML Server Resource & Tunnel Stress Verification")
    print("================================================================")
    
    t0 = time.time()
    try:
        gpu_audit = audit_aml_b_gpu()
        iso_audit = audit_aml_a_isolation()
        load_audit = audit_aml_b_system_load()
        sock_audit = audit_tunnel_socket()
        stress_audit = stress_test_tunnel_concurrency()

        receipt = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": "APPROVE",
            "total_duration_s": round(time.time() - t0, 2),
            "aml_b_gpu_audit": gpu_audit,
            "aml_a_isolation_audit": iso_audit,
            "system_load_audit": load_audit,
            "tunnel_socket_audit": sock_audit,
            "stress_test_audit": stress_audit,
        }

        with open(RECEIPT_PATH, "w") as f:
            json.dump(receipt, f, indent=2)

        print("================================================================")
        print("VERDICT: APPROVE")
        print(f"Saved receipt to {RECEIPT_PATH}")
        print("================================================================")
        sys.exit(0)

    except Exception as e:
        traceback.print_exc()
        print(f"\nCHALLENGE_FAILED: {e}")
        receipt = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": "CHALLENGE_FAILED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        with open(RECEIPT_PATH, "w") as f:
            json.dump(receipt, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
