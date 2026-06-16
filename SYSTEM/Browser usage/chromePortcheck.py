import subprocess

def find_chrome_debug_port():
    # Get all listening ports
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True, text=True
    )

    # Get all chrome PIDs
    tasklist = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq chrome.exe"],
        capture_output=True, text=True
    )

    # Extract chrome PIDs
    chrome_pids = set()
    for line in tasklist.stdout.splitlines():
        parts = line.split()
        if "chrome.exe" in line and len(parts) >= 2:
            chrome_pids.add(parts[1])

    print(f"Chrome PIDs: {chrome_pids}")

    # Find ports those PIDs are listening on
    print("\nChrome listening ports:")
    found = False
    for line in result.stdout.splitlines():
        if "LISTENING" in line:
            pid = line.strip().split()[-1]
            if pid in chrome_pids:
                port = line.strip().split()[1].split(":")[-1]
                print(f"  PID {pid} → port {port}")
                found = True

    if not found:
        print("  No Chrome found listening on any port.")
        print("  (Normal Chrome doesn't expose a debug port — only CDP-launched Chrome does)")

if __name__ == "__main__":
    find_chrome_debug_port()