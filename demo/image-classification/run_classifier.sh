export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=2,3,4

python -u img_classifier.py --dataset=indoor67 $@
