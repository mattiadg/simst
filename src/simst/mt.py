import argparse
import logging
from collections import namedtuple

import ctranslate2
import sentencepiece as spm
import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseModel

app = FastAPI()


AVAILABLE_LANGS = ["en", "de", "it"]


logger.setLevel(level=logging.DEBUG)

#executor = ProcessPoolExecutor(max_workers=2)


class TranslationRequest(BaseModel):
    src_sent: str
    prev_trans: str
    srclang: str
    tgtlang: str
    translation: str


"""
def transcribe_audio(recv_queue: Queue, send_queue: Queue):
    try:
        print("Loading model")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("model loaded")

        while audiodata := recv_queue.get():
            recv_queue.task_done()
            sent = False
            print("received audiodata")
            data = np.frombuffer(base64.b64decode(audiodata.data), dtype=np.float32)
            print("start transcribe")
            segments, _ = model.transcribe(data, beam_size=5, language=audiodata.language, vad_filter=True)
            print("transcription complete")
            for segment in segments:
                send_queue.put(segment)
                sent = True
            if not sent:
                send_queue.put(None)
        else:
            recv_queue.task_done()
    except EOFError:
        print("Data stream reached an end")


async def queue_get(queue: Queue):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.get)
    return await task


async def queue_put(queue: Queue, obj):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, queue.put, obj)
    await task


async def run_generator_in_executor(data: AudioData, send_queue: Queue, recv_queue: Queue):
    print("Sending data")
    await queue_put(send_queue, data)
    print("awaiting for answer")
    segment = await queue_get(recv_queue)
    recv_queue.task_done()
    print(segment)
    if segment is None:
        print(data.start, data.end)
    return segment


@app.websocket("/ws")
async def upload_file(websocket: WebSocket):
    print("received a connection")
    await manager.connect(websocket)
    print("received connection")
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

        print("The show must go on")
        send_queue.join()
"""

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

Translator = namedtuple("Translator", ["engine", "source_spm", "target_spm"])

cuda = ctranslate2.Device.cuda
cpu = ctranslate2.Device.cpu

translators = {}
for lang in models_paths:
    ct_model_path = models_paths[lang]["model"]
    sp_source_model_path = models_paths[lang]["source"]
    sp_target_model_path = models_paths[lang]["target"]

    device = "cpu"
    translator = ctranslate2.Translator(ct_model_path, device)
    sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
    sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)
    translators[lang] = Translator(translator, sp_source_model, sp_target_model)


def translate(
    source: str,
    translator: ctranslate2.Translator,
    sp_source_model: spm.SentencePieceProcessor,
    sp_target_model: spm.SentencePieceProcessor,
):
    """
    Use CTranslate model to translate a sentence

    Args:
        source: Source sentences to translate
        translator: Object of Translator, with the CTranslate2 model
        sp_source_model: Object of SentencePieceProcessor, with the SentencePiece source model
        sp_target_model: Object of SentencePieceProcessor, with the SentencePiece target model
    Returns:
        Translation of the source text
    """
    
    print(f"received text: {source}")
    source_sentences = source
    source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
    print(source_tokenized)
    translation_objs = translator.translate_batch([source_tokenized])
    print(translation_objs)
    translations = translation_objs[0].hypotheses
    translations_detokenized = sp_target_model.decode(translations)
    translation = " ".join(translations_detokenized)
    print(f"produced translation: {translation}")

    return translation


@app.post("/translate")
def translate_service(request: TranslationRequest):
    print("processing translation request")
    # Get the JSON data from the request
    print(f"received data: {request}")

    print(request)
    text = request.src_sent
    source_lang = request.srclang
    target_lang = request.tgtlang

    lang_pair = (
        f"{source_lang.split('-')[0].lower()}-{target_lang.split('-')[0].lower()}"
    )
    translator = translators[lang_pair]
    print(translator)
    translated = translate(text, *translator)
    request.translation = translated

    # Return the processed JSON data
    return request.model_dump(mode="python")


def start_server(args):
    #try:
        uvicorn.run(app, host=args.address, port=args.port, log_level="debug")
    #finally:
        #executor.shutdown(cancel_futures=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    server_parser = subparsers.add_parser("server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1", help="ip address")
    server_parser.add_argument("--port", type=int, default=8001, help="ip port")
    server_parser.set_defaults(func=start_server)

    args = parser.parse_args()
    args.func(args)
