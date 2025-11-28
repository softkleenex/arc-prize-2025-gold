"""Check Kaggle kernel status and wait for completion."""

import subprocess
import time
import sys

kernel_id = "softkleenex/arc-prize-2025-baseline-submission"
max_wait_minutes = 15
check_interval = 20  # seconds

print(f"Checking kernel status: {kernel_id}")
print(f"Will check every {check_interval} seconds for up to {max_wait_minutes} minutes")
print("=" * 70)

start_time = time.time()
max_wait_seconds = max_wait_minutes * 60

attempt = 0
while True:
    attempt += 1
    elapsed = int(time.time() - start_time)

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
            timeout=30
        )

        status_line = result.stdout.strip()
        print(f"[{attempt}] ({elapsed}s) {status_line}")

        if "complete" in status_line.lower():
            print("\n" + "=" * 70)
            print("✓ Kernel execution completed successfully!")
            print("=" * 70)
            sys.exit(0)
        elif "error" in status_line.lower() or "failed" in status_line.lower():
            print("\n" + "=" * 70)
            print("✗ Kernel execution failed!")
            print("=" * 70)
            print(f"Status: {status_line}")
            sys.exit(1)
        elif "running" in status_line.lower():
            print(f"   Still running... (elapsed: {elapsed}s)")

    except subprocess.TimeoutExpired:
        print(f"[{attempt}] Timeout checking status, retrying...")
    except Exception as e:
        print(f"[{attempt}] Error: {e}")

    if elapsed >= max_wait_seconds:
        print("\n" + "=" * 70)
        print(f"⚠ Timeout: Kernel did not complete within {max_wait_minutes} minutes")
        print("=" * 70)
        print("Check status manually at:")
        print(f"https://www.kaggle.com/code/{kernel_id}")
        sys.exit(2)

    time.sleep(check_interval)
