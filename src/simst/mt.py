import argparse
import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from queue import Queue
from typing import NamedTuple

import ctranslate2
import sentencepiece as spm
import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseModel

AVAILABLE_LANGS = ["en", "de", "it"]


logger.setLevel(level=logging.DEBUG)


class TranslationRequest(BaseModel):
    src_sent: str
    prev_trans: str
    srclang: str
    tgtlang: str
    translation: str


models_paths = {
    "en-it": {
        "model": "mt_models/enit/",
        "source": "mt_models/enit/source.spm",
        "target": "mt_models/enit/target.spm",
    },
    "it-en": {
        "model": "mt_models/iten/",
        "source": "mt_models/iten/source.spm",
        "target": "mt_models/iten/target.spm",
    },
    "en-de": {
        "model": "mt_models/ende/",
        "source": "mt_models/ende/source.spm",
        "target": "mt_models/ende/target.spm",
    },
    "de-en": {
        "model": "mt_models/deen/",
        "source": "mt_models/deen/source.spm",
        "target": "mt_models/deen/target.spm",
    },
    "it-de": {
        "model": "mt_models/itde/",
        "source": "mt_models/itde/source.spm",
        "target": "mt_models/itde/target.spm",
    },
    "de-it": {
        "model": "mt_models/deit/",
        "source": "mt_models/deit/source.spm",
        "target": "mt_models/deit/target.spm",
    },
}

cuda = "cuda"
cpu = "cpu"


class BiQueue(NamedTuple):
    send: Queue
    recv: Queue


class MTModelsHandler:
    def __init__(self, device: ctranslate2.Device):
        self.executor = ProcessPoolExecutor(max_workers=6)
        self.device = device
        self.taskgen = parallel_translator_generator(self.executor)
        self.queues: dict[str, BiQueue] = {}

    async def init(self):
        await anext(self.taskgen)
        for lang in models_paths:
            print(lang)
            self.queues[lang] = await self.taskgen.asend((models_paths[lang], cpu))

    async def translate(self, source: str, langpair: str, prev: str):
        queues = self.queues[langpair]
        await queue_put(queues.send, (source, prev))
        translation = await queue_get(queues.recv)
        queues.recv.task_done()
        return translation

    def __del__(self):
        for queues in self.queues.values():
            queues.send.put_nowait(None)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.taskgen.asend(None)
        return False


async def parallel_translator_generator(executor_: ProcessPoolExecutor):
    with multiprocessing.Manager() as manager:
        model_path, device = yield
        loop = asyncio.get_running_loop()
        while model_path:
            send_queue = manager.Queue()
            recv_queue = manager.Queue()
            loop.run_in_executor(executor_, translate_task, model_path, device, send_queue, recv_queue)
            print("task submitted")
            model_path, device = yield BiQueue(send_queue, recv_queue)


def translate_task(model_path: dict, device: ctranslate2.Device, recv_queue: Queue, send_queue: Queue):
    ct_model_path = model_path["model"]
    sp_source_model_path = model_path["source"]
    sp_target_model_path = model_path["target"]

    print("loading model")
    ct2_translator = ctranslate2.Translator(ct_model_path, device)
    print("model loaded")
    print("loading vocabularies")
    sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
    sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)
    print("vocabularies loaded")

    while data := recv_queue.get():
        source, prev = data
        print(f"received text: {source}")
        source_tokenized = sp_source_model.encode(source, out_type=str)
        translation_objs = ct2_translator.translate_batch([source_tokenized])
        translations = translation_objs[0].hypotheses
        translations_detokenized = sp_target_model.decode(translations)
        translation = " ".join(translations_detokenized)
        print(f"produced translation: {translation}")

        recv_queue.task_done()
        send_queue.put(translation)

    else:
        recv_queue.task_done()
        send_queue.join()


async def queue_get(queue: Queue):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.get)
    return await task


async def queue_put(queue: Queue, obj):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.put, obj)
    await task


handler: MTModelsHandler


@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler
    print("init Handler")
    handler = MTModelsHandler(device=cpu)
    await handler.init()
    print("done")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/translate")
async def translate_service(request: TranslationRequest):
    print(f"received data: {request}")

    text = request.src_sent
    source_lang = request.srclang
    target_lang = request.tgtlang
    prev = request.prev_trans

    lang_pair = (
        f"{source_lang.split('-')[0].lower()}-{target_lang.split('-')[0].lower()}"
    )
    translated = await handler.translate(text, lang_pair, prev)
    request.translation = translated

    return request.json()


def start_server(args):
    uvicorn.run(app, host=args.address, port=args.port, log_level="debug")


def send_request(args):
    import requests
    address = f"http://{args.address}:{args.port}/translate"
    payload = TranslationRequest(
        src_sent=args.text, prev_trans="", srclang=args.srclang, tgtlang=args.tgtlang, translation=""
    )
    print(payload.json())
    response = requests.post(address, data=payload.json())
    print(response, response.json())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1", help="ip address")
    server_parser.add_argument("--port", type=int, default=8001, help="ip port")
    server_parser.set_defaults(func=start_server)

    client_parser = subparsers.add_parser("client")
    client_parser.add_argument("--address", type=str, default="127.0.0.1", help="ip address")
    client_parser.add_argument("--port", type=int, default=8001, help="ip port")
    client_parser.add_argument("--text", default="ciao")
    client_parser.add_argument("--srclang", "-s", default="it")
    client_parser.add_argument("--tgtlang", "-t", default="de")
    client_parser.set_defaults(func=send_request)

    args = parser.parse_args()
    args.func(args)
