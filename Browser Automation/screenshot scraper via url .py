import asyncio
import re
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
from screeninfo import get_monitors
monitor = get_monitors()[0]
SCREEN_W = monitor.width
SCREEN_H = monitor.height
TASKBAR_H = 40
VIEWPORT_H = SCREEN_H - TASKBAR_H
print(f"Detected screen: {SCREEN_W}x{SCREEN_H}, viewport: {SCREEN_W}x{VIEWPORT_H}")
URL = input("Enter the URL: ").strip()
OUTPUT_DIR = Path("dataset")
TIMEOUT_MS = 15000
def url_to_filename(url: str, index: int) -> str:
    clean = re.sub(r"https?://", "", url)
    clean = re.sub(r"[^\w\-]", "_", clean)
    clean = clean.strip("_")
    return f"{index:03d}_{clean}.png"
async def screenshot_url(url: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(viewport=None, no_viewport=True)
        page = await context.new_page()
        try:
            filename = url_to_filename(url, 1)
            save_path = OUTPUT_DIR / filename

            await page.goto(url, timeout=TIMEOUT_MS, wait_until="networkidle")
            await page.wait_for_timeout(1500)

            actual_size = await page.evaluate("""() => ({
                width: window.outerWidth,
                height: window.outerHeight
            })""")
            print(f"Browser window: {actual_size['width']}x{actual_size['height']}")
            await page.screenshot(
                path=str(save_path),
                full_page=False,
                clip={
                    "x": 0,
                    "y": 0,
                    "width": actual_size["width"],
                    "height": actual_size["height"]
                }
            )
            print(f"Saved: {save_path} ({actual_size['width']}x{actual_size['height']})")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await page.close()
            await browser.close()
    manifest_path = OUTPUT_DIR / "manifest.txt"
    manifest_exists = manifest_path.exists()
    with open(manifest_path, "a", encoding="utf-8") as f:
        if not manifest_exists:
            f.write("=" * 50 + "\n")
            f.write("        SCREENSHOT MANIFEST\n")
            f.write("=" * 50 + "\n\n")
        f.write(f"Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  URL      : {url}\n")
        f.write(f"  File     : {filename}\n")
        f.write(f"  Size     : {actual_size['width']}x{actual_size['height']}\n")
        f.write("-" * 50 + "\n\n")
if __name__ == "__main__":
    asyncio.run(screenshot_url(URL))