import asyncio
import pathlib
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import soundfile as sf


@dataclass
class Audio:
    data: np.ndarray
    sr: int


class AudioStreamer:
    def __init__(self, audio: Audio):
        self.audiodata = audio

    async def stream(self, delay: int) -> Iterator[np.ndarray]:
        """
        Stream the wrapped audio in batches of `delay` ms
        :param delay: length of the audio stream at each batch
        :return:
        """
        i = 0
        batchlen = int(delay * self.audiodata.sr)
        while i < len(self.audiodata.data):  # this may need to change when receiving actual streams
            await asyncio.sleep(delay / 1000)
            nexti = i + batchlen
            yield self.audiodata.data[i:nexti]
            i = nexti


def simultaneous_st(audio: Audio, config: STConfig) -> Iterator[tuple[str, str]]:
    streamer = AudioStreamer(audio)
    asr = Asr(config.srclang)
    mt = Mt(config.srclang, config.tgtlang)

    for batch in streamer.stream(config.delay):
        transcript = asr.recognize(batch)
        translation = mt.translate(transcript)

        yield transcript, translation
