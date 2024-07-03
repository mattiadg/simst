import argparse
import asyncio
import base64
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from queue import Queue

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.websockets import WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from pydantic import BaseModel

app = FastAPI()


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


manager = ConnectionManager()

executor = ProcessPoolExecutor(max_workers=2)


def transcribe_audio(recv_queue: Queue, send_queue: Queue):
    try:
        logger.log(logging.DEBUG, "Loading model")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.log(logging.DEBUG, "model loaded")

        while audiodata := recv_queue.get():
            recv_queue.task_done()
            sent = False
            logger.log(logging.DEBUG, "received audiodata")
            data = np.frombuffer(base64.b64decode(audiodata.data), dtype=np.float32)
            logger.log(logging.DEBUG, "start transcribe")
            segments, _ = model.transcribe(data, beam_size=5, language=audiodata.language, vad_filter=True)
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


async def queue_get(queue: Queue):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.get)
    return await task


async def queue_put(queue: Queue, obj):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.put, obj)
    await task


async def run_generator_in_executor(data: AudioData, send_queue: Queue, recv_queue: Queue):
    logger.log(logging.DEBUG, "Sending data")
    await queue_put(send_queue, data)
    logger.log(logging.DEBUG, "awaiting for answer")
    segment = await queue_get(recv_queue)
    recv_queue.task_done()
    logger.log(logging.DEBUG, segment)
    if segment is None:
        logger.log(logging.DEBUG, data.start, data.end)
    return segment


@app.websocket("/ws")
async def run_transcription_audio(websocket: WebSocket):
    await manager.connect(websocket)
    logger.log(logging.DEBUG, "received connection")
    with multiprocessing.Manager() as mmanager:
        send_queue = mmanager.Queue()
        recv_queue = mmanager.Queue()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(executor, transcribe_audio, send_queue, recv_queue)
        try:
            while True:
                data = await websocket.receive_json()
                audiodata = AudioData(**data)
                if audiodata.language not in AVAILABLE_LANGS:
                    future.cancel()
                    await queue_put(send_queue, None)
                    await websocket.send_text(f"Not available language {audiodata.language}, expected one of {AVAILABLE_LANGS}")
                    manager.disconnect(websocket)
                    break

                segment = await run_generator_in_executor(audiodata, send_queue, recv_queue)
                if not segment:
                    await websocket.send_json(Segment(start=audiodata.start, end=audiodata.end, text="").dict())
                else:
                    await websocket.send_json(Segment(start=audiodata.start, end=audiodata.end, text=segment.text).dict())
        except WebSocketDisconnect:
            manager.disconnect(websocket)

        logger.log(logging.DEBUG, "The show must go on")
        send_queue.join()


def start_server(args):
    try:
        uvicorn.run(app, host=args.address, port=args.port, log_level="debug")
    finally:
        executor.shutdown(cancel_futures=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1", help="ip address")
    server_parser.add_argument("--port", type=int, default=8000, help="ip port")
    server_parser.set_defaults(func=start_server)

    args = parser.parse_args()
    args.func(args)
