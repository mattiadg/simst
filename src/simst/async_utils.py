import asyncio
from queue import Queue
from typing import NamedTuple


class BiQueue(NamedTuple):
    send: Queue
    recv: Queue


async def queue_get(queue: Queue):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.get)
    return await task


async def queue_put(queue: Queue, obj):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.put, obj)
    await task