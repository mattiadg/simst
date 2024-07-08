import asyncio
from queue import Queue
from typing import NamedTuple


class BiQueue(NamedTuple):
    send: Queue
    recv: Queue


async def queue_get(queue: Queue, blocking=True, timeout=None):
    task = asyncio.to_thread(queue.get, block=blocking, timeout=timeout)
    return await task


async def queue_put(queue: Queue, obj):
    task = asyncio.to_thread(queue.put, obj)
    await task


def health():
    return {"up": True}