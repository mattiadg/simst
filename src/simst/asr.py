import argparse
import asyncio
import base64
import json
import logging
import multiprocessing
import queue
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from queue import Queue

import httpx
import numpy as np
import uvicorn
import websockets
from deepmultilingualpunctuation import PunctuationModel
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.websockets import WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import BaseModel

from src.simst.async_utils import queue_get, queue_put, BiQueue, health

model_size = "whisper-medium-ct2-int"

AVAILABLE_LANGS = ["en", "de", "it"]


logger.setLevel(level=logging.DEBUG)


class AudioData(BaseModel):
    format: str
    samplerate: int
    channels: int
    language: str
    data: str
    start: float
    end: float
    prefix: str


class Segment(BaseModel):
    start: float
    end: float
    text: str


@dataclass
class ReturnTranscript:
    text: str
    complete: bool


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        return len(self.active_connections) - 1

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    def disconnect_all(self):
        self.active_connections.clear()


class ASR:
    def __init__(self, lang: str, port: int, in_queue: Queue, out_queue: Queue, close_queue: queue.Queue, timeout: int = 5, ):
        self.lang = lang
        self.port = port
        self.timeout = timeout
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.close_queue = close_queue

        self.server = f"localhost:{port}"

        health = httpx.get(f"http://{self.server}/health", timeout=timeout)
        if health.status_code != httpx.codes.ok:
            raise RuntimeError(f"Impossible to contact server at http://localhost:{port}")

    async def transcribe(self):
        print("ASR: start transcribe")
        json_data = {
            "format": "wav",
            "samplerate": 16000,
            "channels": 1,
            "language": "en",
        }
        prefix = ""
        sr = json_data["samplerate"]
        async with websockets.connect(f"ws://{self.server}/ws") as websocket:
            i = -1
            while (audio_data := await queue_get(self.in_queue)) is not None:
                try:
                    if self.close_queue.get(block=False) is None:
                        await queue_put(self.out_queue, None)
                        print("ASR: closing for close_queue")
                        self.close_queue.task_done()
                        break
                except queue.Empty:
                    pass
                try:
                    while (additional := await asyncio.to_thread(self.in_queue.get, block=False)) is not None:
                        audio_data = np.concatenate([audio_data, additional])
                    else:
                        await queue_put(self.in_queue, None)
                except queue.Empty:
                    pass
                i += 1
                json_data["data"] = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                json_data["start"] = i * len(audio_data) / sr
                json_data["end"] = (i + 1) * len(audio_data) / sr
                json_data["prefix"] = prefix
                # Send a POST request with JSON data
                await websocket.send(AudioData(**json_data).json())
                response = await websocket.recv()
                try:
                    response_dict = json.loads(response)
                    segment = Segment(**response_dict)
                except json.decoder.JSONDecodeError as e:
                    print(f"Response: {response}")
                    exit()
                except Exception as e:
                    print(response)
                    raise e
                to_trans = segment.text
                if to_trans.endswith("."):
                    to_trans = to_trans.rstrip(".")
                prefix = to_trans
                try:
                    sentence_end = to_trans.index(".")
                    if sentence_end > -1:
                        trans1 = to_trans[:sentence_end + 1]
                        await queue_put(self.out_queue, ReturnTranscript(trans1, complete=True))
                        to_trans = to_trans[sentence_end + 1:]
                        prefix = to_trans
                except ValueError:
                    await queue_put(self.out_queue, ReturnTranscript(to_trans, complete=False))
            else:
                await queue_put(self.out_queue, None)
                await queue_get(self.close_queue)
                self.close_queue.task_done()


class ASRModelsHandler:
    def __init__(self):
        self.max_workers = 1
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.device = "cuda"
        self.taskgen = parallel_transcription_generator(self.executor)
        self.queues: list[BiQueue] = []

    async def init(self):
        await anext(self.taskgen)
        for _ in range(self.max_workers):
            self.queues.append(await self.taskgen.asend(1))

    async def transcribe(self, data: AudioData):
        queues = self.queues[0]
        logger.log(logging.DEBUG, "Sending data")
        await queue_put(queues.send, data)
        logger.log(logging.DEBUG, "awaiting for answer")
        segment = await queue_get(queues.recv)
        queues.recv.task_done()
        logger.log(logging.DEBUG, segment)
        if segment is None:
            logger.log(logging.DEBUG, data.start, data.end)

        return segment

    def __del__(self):
        for queues in self.queues:
            queues.send.put_nowait(None)
        self.executor.shutdown(wait=True)
        self.taskgen.asend(None)
        return False


async def parallel_transcription_generator(executor_: ProcessPoolExecutor):
    with multiprocessing.Manager() as manager:
        i = yield
        loop = asyncio.get_running_loop()
        while i:
            send_queue = manager.Queue()
            recv_queue = manager.Queue()
            loop.run_in_executor(executor_, transcribe_audio, send_queue, recv_queue)
            i = yield BiQueue(send_queue, recv_queue)


def transcribe_audio(recv_queue: Queue, send_queue: Queue):
    try:
        logger.log(logging.DEBUG, "Loading model")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        punc_model = PunctuationModel(model="kredor/punctuate-all")
        logger.log(logging.DEBUG, "model loaded")

        while audiodata := recv_queue.get():
            sent = False
            logger.log(logging.DEBUG, "received audiodata")
            data = np.frombuffer(base64.b64decode(audiodata.data), dtype=np.float32)
            prefix = audiodata.prefix
            logger.log(logging.DEBUG, "start transcribe")
            segments, _ = model.transcribe(data, beam_size=5, language=audiodata.language, vad_filter=True, initial_prompt=prefix)
            recv_queue.task_done()
            logger.log(logging.DEBUG, "transcription complete")
            text = prefix
            for segment in segments:
                text += " " + segment.text
            if text != prefix:
                segdict = segment._asdict()
                del segdict["text"]
                segment_ = Segment(**segdict, text=punc_model.restore_punctuation(text))
                send_queue.put(segment_)
                sent = True
            if not sent:
                send_queue.put(None)
        else:
            recv_queue.task_done()
    except EOFError:
        logger.log(logging.DEBUG, "Data stream reached an end")


handler: ASRModelsHandler
manager: ConnectionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler, manager
    handler = ASRModelsHandler()
    manager = ConnectionManager()
    await handler.init()
    yield
    manager.disconnect_all()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def run_transcription_audio(websocket: WebSocket):
    await manager.connect(websocket)
    logger.log(logging.DEBUG, "received connection")

    try:
        while True:
            data = await websocket.receive_json()
            audiodata = AudioData(**data)
            if audiodata.language not in AVAILABLE_LANGS:
                await websocket.send_text(f"Not available language {audiodata.language}, expected one of {AVAILABLE_LANGS}")
                manager.disconnect(websocket)
                break

            segment = await handler.transcribe(audiodata)
            if not segment:
                await websocket.send_json(Segment(start=audiodata.start, end=audiodata.end, text="").dict())
            else:
                await websocket.send_json(Segment(start=audiodata.start, end=audiodata.end, text=segment.text).dict())
    except WebSocketDisconnect:
        manager.disconnect(websocket)

    logger.log(logging.DEBUG, "The show must go on")


app.get("/health")(health)


def start_server(args):
    uvicorn.run(app, host=args.address, port=args.port, log_level="debug")


def start_asr(args):
    ASR(args.lang, args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1", help="ip address")
    server_parser.add_argument("--port", type=int, default=8000, help="ip port")
    server_parser.set_defaults(func=start_server)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--lang", "-t", default="de")
    check_parser.add_argument("--port", type=int, default=8000, help="ip port")
    check_parser.set_defaults(func=start_asr)

    args = parser.parse_args()
    args.func(args)
