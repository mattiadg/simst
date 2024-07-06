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
from src.simst.mt import MT


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
    def __init__(self, audiofile: pathlib.Path, out_queue: queue.Queue):
        self.audio = Audio(decode_audio(audiofile), 16000)
        self.out_queue = out_queue

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
            print("STREAM: sending data")
            nexti = i + batchlen
            await queue_put(self.out_queue, self.audio.data[i:nexti])
            await asyncio.sleep(step / 1000)
            i = i + int(step * self.audio.sr / 1000)

        queue_put(self.out_queue, None)


async def simultaneous_st(audio: pathlib.Path, config: STConfig) -> Iterator[tuple[str, str]]:
    with multiprocessing.Manager() as manager:
        stream_queue = manager.Queue()
        asr_queue = manager.Queue()
        mt_queue = manager.Queue()
        streamer = AudioStreamer(audio, stream_queue)
        asr = ASR(config.srclang, config.asr_port, stream_queue, asr_queue)
        mt = MT(config.srclang, config.tgtlang, config.mt_port, asr_queue, mt_queue)

        task_stream = asyncio.create_task(streamer.stream(config.initial_delay, config.step))
        task_asr = asyncio.create_task(asr.transcribe())
        task_mt = asyncio.create_task(mt.translate())

        while (data := await queue_get(mt_queue)) is not None:
            yield data


async def main():
    config = STConfig(
        srclang="en",
        tgtlang="it",
        initial_delay=2000,
        step=2000,
        asr_port=8000,
        mt_port=8001,
    )
    audio_path = pathlib.Path("C:/Users/matti/PycharmProjects/simst/simst/audio.wav")
    async for a, b in simultaneous_st(audio_path, config):
        print(f"{a} --- {b}")


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
