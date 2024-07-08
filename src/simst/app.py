import pathlib
import shutil
import sys

import gradio as gr

from src.simst.backend import main

UPLOAD_FOLDER = pathlib.Path("./demo_files")


def upload_file(file):
    if not UPLOAD_FOLDER.exists():
        UPLOAD_FOLDER.mkdir()
    shutil.copy(file, UPLOAD_FOLDER)
    gr.Info("file uploaded")


def main_app():
    running = False

    def wrap_main(file):
        global running
        running = True
        for x in main(file):
            if running:
                yield x
            else:
                return

    def stop():
        global running
        if not running:
            return
        running = False

    with gr.Blocks() as demo:
        gr.Markdown("""# My App""")
        with gr.Row(equal_height=True):
            with gr.Column():
                upload_button = gr.UploadButton("Upload")
                upload_button.upload(upload_file, upload_button)
                start_button = gr.Button()
                stop_button = gr.Button("Stop")
            transcription_text = gr.TextArea(label="Transcription")
            translation_text = gr.TextArea(label="Translation")

        start_button.click(wrap_main, [upload_button], [transcription_text, translation_text])
        stop_button.click(stop)

    demo.launch()


if __name__ == '__main__':
    sys.exit(main_app())
