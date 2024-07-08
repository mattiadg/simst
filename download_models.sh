mkdir -p src/simst/mt_models/opus
wget https://object.pouta.csc.fi/OPUS-MT-models/en-it/opus-2019-12-04.zip -O opus/enit.zip
wget https://object.pouta.csc.fi/OPUS-MT-models/en-de/opus-2020-02-26.zip -O opus/ende.zip
wget https://object.pouta.csc.fi/OPUS-MT-models/de-en/opus-2020-02-26.zip -O opus/deen.zip
wget https://object.pouta.csc.fi/OPUS-MT-models/it-en/opus-2019-12-18.zip -O opus/iten.zip
wget https://object.pouta.csc.fi/OPUS-MT-models/it-de/opus-2020-01-20.zip -O opus/itde.zip
wget https://object.pouta.csc.fi/OPUS-MT-models/de-it/opus-2020-01-20.zip -O opus/deit.zip

pushd src/simst/mt_models/opus
for file in *.zip; do unzip $file -d $(basename $file .zip); done
popd

pushd src/simst/mt_models
for dir in enit ende iten deen itde deit; do
  ct2-opus-mt-converter --model_dir opus/$dir --output_dir $dir --quantization int8
  cp opus/source.spm $dir/
  cp opus/target.spm $dir/
done
popd

ct2-transformers-converter --model SYSTRAN/faster-whisper-medium --output_dir src/simst/whisper-medium-ct2-int8 --quantization int8