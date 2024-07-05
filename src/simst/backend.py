import asyncio
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from faster_whisper import decode_audio

from src.simst.asr import ASR
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
    def __init__(self, audiofile: pathlib.Path):
        self.audio = Audio(decode_audio(audiofile), 16000)

    async def stream(self, delay: int, step: int) -> Iterator[np.ndarray]:
        """
        Stream the wrapped audio in batches of `delay` ms
        :param delay: length of the audio stream at each batch
        :return:
        """
        i = 0
        batchlen = int(delay * self.audio.sr / 1000)
        #await asyncio.sleep(delay / 1000)
        while i < len(self.audio.data):  # this may need to change when receiving actual streams
            await asyncio.sleep(1)
            nexti = i + batchlen
            yield self.audio.data[i:nexti]
            i = i + int(step * self.audio.sr / 1000)


async def simultaneous_st(audio: pathlib.Path, config: STConfig) -> Iterator[tuple[str, str]]:
    streamer = AudioStreamer(audio)
    asr = ASR(config.srclang, config.asr_port)
    mt = MT(config.srclang, config.tgtlang, config.mt_port)

    buffer = None
    transcriber = asr.transcribe(streamer.stream(config.initial_delay, config.step))
    while True:
        transcript = await transcriber.asend(buffer)

        to_trans = transcript.text
        if to_trans.endswith("."):
            to_trans = to_trans.rstrip(".")
        buffer = to_trans
        try:
            sentence_end = to_trans.index(".")
            if sentence_end > -1:
                trans1 = to_trans[:sentence_end+1]
                print("truncated = " + trans1)
                translation1 = await mt.translate(trans1, "")
                yield trans1, translation1.translation
                to_trans = to_trans[sentence_end+1:]
                buffer = to_trans
        except ValueError:
            translation = await mt.translate(to_trans, "")
            yield to_trans, translation.translation


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
