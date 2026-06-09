import asyncio
import re
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
URL = input("Enter the URL: ").strip()
OUTPUT_DIR = Path("dataset")
VIEWPORT = {"width": 1280, "height": 800}
TIMEOUT_MS = 15000
FULL_PAGE = False
def url_to_filename(url: str, index: int) -> str:
    clean = re.sub(r"https?://", "", url)
    clean = re.sub(r"[^\w\-]", "_", clean)
    clean = clean.strip("_")
    return f"{index:03d}_{clean}.png"
async def screenshot_url(url: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()
        try:
            filename = url_to_filename(url, 1)
            save_path = OUTPUT_DIR / filename
            await page.goto(url, timeout=TIMEOUT_MS, wait_until="networkidle")
            await page.screenshot(path=str(save_path), full_page=FULL_PAGE)
            print(f"Screenshot saved to: {save_path}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await page.close()
            await browser.close()
    manifest_path = OUTPUT_DIR / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(
            f"Screenshot run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write(f"{filename} → {url}\n")
if __name__ == "__main__":
    asyncio.run(screenshot_url(URL))