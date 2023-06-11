python onnx2trt.py --onnx adaround.onnx \
        --trt adaround.trt \
        --data /data/wqzhao/Datasets/imagenet \
        --clip-range-file trt_clip_val.json \
        --evaluate