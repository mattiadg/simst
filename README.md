# SimST - Simultaneuous Speech Translation

This repo is an experiment in setting up a gradio app for speech translation using Python's asyncio
api everywhere to reduce dead times.

It runs on the open source models Whisper for ASR and OpusMT for MT.

# Set up
Create a working environment using [pdm](https://pdm-project.org/en/latest/) with 
```bash
pdm install
```
in the root folder.

## Running it
```bash 
python src/simst/asr.py server
```
and in another terminal
```bash 
python src/simst/mt.py server
```

to start the servers, then finally run the [Gradio](https://www.gradio.app/) app:
```bash 
python src/simst/app.py
```

and then open your browser at http://127.0.0.1:7860