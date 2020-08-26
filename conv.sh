#!/bin/sh

rm -rf gest.tfmodel
tensorflowjs_converter --input_format tfjs_layers_model --output_format keras_saved_model gestures.tfjsmodel.json gest.tfmodel
tflite_convert --output_file gest.tflite --saved_model_dir gest.tfmodel/
hexdump -v -e '32/1 "%02x " 1/0 "\n"'  gest.tflite > gest.hex
