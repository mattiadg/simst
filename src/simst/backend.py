import asyncio
import multiprocessing
import pathlib
import queue
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from faster_whisper import decode_audio

from src.simst.asr import ASR
from src.simst.async_utils import queue_put, queue_get
from src.simst.mt import MT, ReturnText


@dataclass
class Audio:
    data: np.ndarray
    sr: int


@dataclass
class STConfig:
    srclang: str
    tgtlang: str
    initial_delay: int
    step: int
    asr_port: int
    mt_port: int


class AudioStreamer:
    def __init__(self, audiofile: pathlib.Path, out_queue: queue.Queue, close_queue: queue.Queue):
        self.audio = Audio(decode_audio(audiofile), 16000)
        self.out_queue = out_queue
        self.close_queue = close_queue

    async def stream(self, delay: int, step: int):
        """
        Stream the wrapped audio in batches of `delay` ms
        :param delay: length of the audio stream at each batch
        :return:
        """
        i = 0
        batchlen = int(delay * self.audio.sr / 1000)
        await asyncio.sleep(delay / 1000)
        while i < len(self.audio.data):  # this may need to change when receiving actual streams
            try:
                if self.close_queue.get(block=False) is None:
                    self.close_queue.task_done()
                    await queue_put(self.out_queue, None)
                    break
            except queue.Empty:
                pass
            nexti = i + batchlen
            await queue_put(self.out_queue, self.audio.data[i:nexti])
            await asyncio.sleep(step / 1000)
            i = i + int(step * self.audio.sr / 1000)
        else:
            await queue_put(self.out_queue, None)
            await queue_get(self.close_queue)
            self.close_queue.task_done()


async def simultaneous_st(audio: pathlib.Path, config: STConfig) -> Iterator[ReturnText]:
    with multiprocessing.Manager() as manager:
        close_queue = manager.Queue()
        stream_queue = manager.Queue()
        asr_queue = manager.Queue()
        mt_queue = manager.Queue()
        streamer = AudioStreamer(audio, stream_queue, close_queue)
        asr = ASR(config.srclang, config.asr_port, stream_queue, asr_queue, close_queue)
        mt = MT(config.srclang, config.tgtlang, config.mt_port, asr_queue, mt_queue, close_queue)

        try:
            asyncio.create_task(streamer.stream(config.initial_delay, config.step))
            asyncio.create_task(asr.transcribe())
            asyncio.create_task(mt.translate())

            while True:
                try:
                    """ Unlock the driver every one second to allow cleanup in case of interruption """
                    data = await queue_get(mt_queue, timeout=1)
                    if data is not None:
                        yield data
                    else:
                        break
                except queue.Empty:
                    pass

        finally:
            await queue_put(close_queue, None)
            await queue_put(close_queue, None)
            await queue_put(close_queue, None)
            await asyncio.to_thread(close_queue.join)


async def start(audio_path):
    config = STConfig(
        srclang="en",
        tgtlang="it",
        initial_delay=2000,
        step=2000,
        asr_port=8000,
        mt_port=8001,
    )

    buffer1, buffer2 = "", ""
    last1, last2 = "", ""
    async for a, b, complete in simultaneous_st(audio_path, config):
        print(f"{a} --- {b}")
        if complete:
            buffer1 += " " + a
            buffer2 += " " + b
            if len(buffer1) > len(last1):
                yield buffer1, buffer2
        else:
            last1 = a
            last2 = b
            yield buffer1 + " " + a, buffer2 + " " + b


loop = asyncio.get_event_loop()


def main(file):
    task = start(file)
    while True:
        try:
            yield loop.run_until_complete(task.__anext__())
        except StopAsyncIteration:
            break


if __name__ == '__main__':
    file = pathlib.Path("C:/Users/matti/PycharmProjects/simst/simst/audio.wav")
    sys.exit(main(file))
