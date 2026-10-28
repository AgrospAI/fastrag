"""Optional Chromium rendering for JavaScript pages and SVG rasterization."""

from __future__ import annotations

import asyncio
import base64
import re
from typing import Any

from fastrag.exceptions import ConfigurationError, FetchError
from fastrag.ingestion.steps.fetching.http import FetchedPage, normalize_uri


def _playwright() -> Any:
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:
        raise ConfigurationError(
            "Dynamic rendering needs fastrag-ai[dynamic] and playwright install chromium"
        ) from error
    return async_playwright()


IDLE_GRACE_MS = 2000


class SharedBrowser:
    """One lazily launched Chromium process reused across pages.

    Callers open an isolated context per page and close only that context, so
    concurrent renders share the process start-up cost. ``aclose`` stops the
    browser and the Playwright driver and is idempotent.
    """

    def __init__(self) -> None:
        """Create an idle pool; nothing launches until the first ``get``."""
        self._lock = asyncio.Lock()
        self._runtime_manager: Any = None
        self._runtime: Any = None
        self._browser: Any = None

    async def get(self) -> Any:
        """Return a connected browser, launching or relaunching it if needed."""
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            if self._runtime is None:
                self._runtime_manager = _playwright()
                self._runtime = await self._runtime_manager.__aenter__()
            self._browser = await self._runtime.chromium.launch()
            return self._browser

    async def aclose(self) -> None:
        """Close the browser and driver. Safe to call repeatedly."""
        async with self._lock:
            browser, manager = self._browser, self._runtime_manager
            self._browser = self._runtime = self._runtime_manager = None
            try:
                if browser is not None:
                    await browser.close()
            finally:
                if manager is not None:
                    await manager.__aexit__(None, None, None)


async def wait_for_hydration(page: Any, content: bytes) -> None:
    """Wait until Docusaurus attaches handlers to server-rendered controls."""
    if re.search(rb"<meta\b[^>]*content=[\"']Docusaurus", content, re.I):
        await page.wait_for_selector('html[data-has-hydrated="true"]', state="attached")


async def render_html(
    source: FetchedPage,
    timeout: float,
    wait_selector: str | None,
    shared: SharedBrowser | None = None,
    wait_until: str = "networkidle",
) -> bytes:
    """Hydrate fetched HTML at its original URL, preserving the supplied bytes.

    An intercepted navigation avoids fetching a different version of the page;
    scripts and assets still resolve against the original URL. Wait for a site's
    explicit readiness selector when provided. Docusaurus hydration is detected
    automatically. Browser resources close even on cancellation or failure.
    With ``shared`` only this page's context closes; the browser stays up.
    """
    if shared is not None:
        browser = await shared.get()
        return await _render_in(
            browser, source, timeout, wait_selector, wait_until, close=False
        )
    async with _playwright() as runtime:
        browser = await runtime.chromium.launch()
        return await _render_in(
            browser, source, timeout, wait_selector, wait_until, close=True
        )


async def _render_in(
    browser: Any,
    source: FetchedPage,
    timeout: float,
    wait_selector: str | None,
    wait_until: str,
    *,
    close: bool,
) -> bytes:
    context = await browser.new_context(viewport={"width": 1440, "height": 1000})
    try:
        page = await context.new_page()
        page.set_default_timeout(timeout * 1000)
        if source.uri.startswith(("http://", "https://")):
            served = False

            async def serve(route: Any) -> None:
                nonlocal served
                if not served and normalize_uri(route.request.url) == normalize_uri(
                    source.uri
                ):
                    served = True
                    await route.fulfill(body=source.content, content_type="text/html")
                else:
                    await route.continue_()

            await context.route("**/*", serve)
            await page.goto(source.uri, wait_until=wait_until)
        else:
            await page.set_content(
                source.content.decode("utf-8"), wait_until=wait_until
            )
        if wait_until != "networkidle":
            # Best effort: let XHR-driven content land without letting a stalled
            # asset (which blocks ``load``/``networkidle``) hold up the page.
            try:
                await page.wait_for_load_state("networkidle", timeout=IDLE_GRACE_MS)
            except Exception:
                pass
        await wait_for_hydration(page, source.content)
        if wait_selector:
            await page.wait_for_selector(wait_selector, state="attached")
        await page.evaluate("""() => {
            for (const el of document.querySelectorAll('[role="tabpanel"], details')) {
                el.removeAttribute('hidden'); el.style.display = 'block';
                if (el.tagName === 'DETAILS') el.open = true;
            }
        }""")
        for svg in await page.locator(".markdown svg, article svg, main svg").all():
            # Ignore decorative icons; retain labeled diagrams and Mermaid.
            if not await svg.evaluate(
                "el => !!el.closest('.docusaurus-mermaid-container') || !!el.querySelector('text, foreignObject, title, desc')"
            ):
                continue
            png = await svg.screenshot(type="png")
            await svg.evaluate(
                '(el, data) => el.setAttribute("data-fastrag-png", data)',
                base64.b64encode(png).decode("ascii"),
            )
        return (await page.content()).encode("utf-8")
    except Exception as error:
        raise FetchError(
            f"Dynamic rendering failed for {source.uri}: {error}"
        ) from error
    finally:
        await context.close()
        if close:
            await browser.close()
