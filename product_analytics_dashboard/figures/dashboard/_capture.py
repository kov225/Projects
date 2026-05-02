"""Capture Streamlit dashboard screenshots for the README.

Run while `streamlit run dashboard/app.py --server.port 8765` is up.
Writes one PNG per tab into figures/dashboard/.
"""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = "http://localhost:8765"
TABS = ["Overview", "Cohort analysis", "Experiments", "User segments"]
SLUGS = ["overview", "cohort", "experiments", "segments"]


WAIT_PER_TAB = {
    "Overview": 4_000,
    "Cohort analysis": 5_500,
    "Experiments": 5_500,
    "User segments": 60_000,
}


def wait_for_render(page, settle_ms: int) -> None:
    """Wait until Streamlit has finished running and DOM has settled."""
    deadline = time.time() + settle_ms / 1000.0
    last_height = -1
    while time.time() < deadline:
        page.wait_for_timeout(800)
        running = page.locator('[data-testid="stStatusWidget"]')
        try:
            visible = running.is_visible()
        except Exception:
            visible = False
        height = page.evaluate("document.body.scrollHeight")
        if not visible and height == last_height:
            return
        last_height = height


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1480, "height": 1000}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(4_000)

        for tab, slug in zip(TABS, SLUGS):
            tab_button = page.get_by_role("tab", name=tab)
            tab_button.click()
            wait_for_render(page, WAIT_PER_TAB.get(tab, 4_000))
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(800)
            target = OUT / f"{slug}.png"
            page.screenshot(path=str(target), full_page=True)
            print("wrote", target)

        browser.close()


if __name__ == "__main__":
    main()
