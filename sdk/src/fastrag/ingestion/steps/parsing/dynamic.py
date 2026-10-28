"""SVG rasterization for vision parsing."""

from fastrag.ingestion.steps.fetching._browser import _playwright


async def rasterize_svg(content: bytes, timeout: float) -> bytes:
    """Render SVG including foreignObject labels to PNG for vision models."""
    async with _playwright() as runtime:
        browser = await runtime.chromium.launch()
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(timeout * 1000)
            await page.set_content(content.decode("utf-8"))
            return await page.locator("svg").first.screenshot(type="png")
        finally:
            await browser.close()
