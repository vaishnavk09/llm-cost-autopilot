from __future__ import annotations

import asyncio
import logging
import os
import time

from app.db import init_db, pending_verifications
from app.pipeline import run_verification_job

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("autopilot.worker")


async def loop(poll_seconds: float = 2.0) -> None:
    init_db()
    log.info("verification worker started")
    while True:
        rows = pending_verifications(limit=5)
        if not rows:
            await asyncio.sleep(poll_seconds)
            continue
        for row in rows:
            started = time.perf_counter()
            try:
                await run_verification_job(dict(row))
                log.info("verified request %s in %.0fms", row["id"], (time.perf_counter() - started) * 1000)
            except Exception:
                log.exception("verification failed for request %s", row["id"])
        await asyncio.sleep(0.2)


def main() -> None:
    asyncio.run(loop())


if __name__ == "__main__":
    main()
