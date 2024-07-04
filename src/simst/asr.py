import argparse
import asyncio
import base64
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from queue import Queue

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.websockets import WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import BaseModel

from src.simst.async_utils import queue_get, queue_put, BiQueue


model_size = "whisper-large-v3-ct2-int"

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


class Segment(BaseModel):
    start: float
    end: float
    text: str


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        return len(self.active_connections) - 1

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)


class ASRModelsHandler:
    def __init__(self):
        self.max_workers = 1
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.device = "cpu"
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
        logger.log(logging.DEBUG, "model loaded")

        while audiodata := recv_queue.get():
            sent = False
            logger.log(logging.DEBUG, "received audiodata")
            data = np.frombuffer(base64.b64decode(audiodata.data), dtype=np.float32)
            logger.log(logging.DEBUG, "start transcribe")
            segments, _ = model.transcribe(data, beam_size=5, language=audiodata.language, vad_filter=True)
            recv_queue.task_done()
            logger.log(logging.DEBUG, "transcription complete")
            for segment in segments:
                send_queue.put(segment)
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
    manager.disconnect()


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


def start_server(args):
    uvicorn.run(app, host=args.address, port=args.port, log_level="debug")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1", help="ip address")
    server_parser.add_argument("--port", type=int, default=8000, help="ip port")
    server_parser.set_defaults(func=start_server)

    args = parser.parse_args()
    args.func(args)
