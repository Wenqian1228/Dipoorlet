python -m dipoorlet \
      -M resnet-18.onnx \
      -I cali_path \
      -N 30 \
      -A hist \
      -D trt \
      --bc \
      --adaround \
      # --brecq
