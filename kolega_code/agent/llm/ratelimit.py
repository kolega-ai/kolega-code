import asyncio
import time
from typing import List, Optional


class RateLimiter:
    """Rate limiter for API requests and tokens"""

    def __init__(self, requests_per_minute: Optional[int] = None, tokens_per_minute: Optional[int] = None):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps: List[float] = []
        self.token_timestamps: List[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: Optional[int] = None):
        """Acquire permission to make a request"""
        async with self.lock:
            current_time = time.time()

            # Clean up old timestamps
            self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
            self.token_timestamps = [ts for ts in self.token_timestamps if current_time - ts < 60]

            # Check request rate limit
            if self.requests_per_minute and len(self.request_timestamps) >= self.requests_per_minute:
                wait_time = 60 - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # Check token rate limit
            if self.tokens_per_minute and tokens and len(self.token_timestamps) >= self.tokens_per_minute:
                wait_time = 60 - (current_time - self.token_timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # Record new timestamps
            self.request_timestamps.append(current_time)
            if tokens:
                self.token_timestamps.extend([current_time] * tokens)
