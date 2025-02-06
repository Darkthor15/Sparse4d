import os
import sys
import time
import warnings
import argparse
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import torch
from torch.profiler import profile, ProfilerActivity,  schedule, tensorboard_trace_handler
from torch2trt import TRTModule
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.parallel import scatter
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.runner import wrap_fp16_model

# Set globals
GPU_ID = 0
CONFIG = 'sparse4dv3_temporal_r50_1x8_bs6_256x704'
CKPT = "ckpt/sparse4dv3_r50.pth"
CFG = Config.fromfile(f"projects/configs/{CONFIG}.py")
FP16 = CFG.get("fp16", None)
WAIT = 2
WARMUP = 1
ACTIVE = 1
ACTIVITY = [ProfilerActivity.CPU, ProfilerActivity.CUDA]  
SCHEDULE = schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1)
TRACER = tensorboard_trace_handler('./log/sparse4dv3')

# Build dataloader
def dataset_build():
    dataset = build_dataset(CFG.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    data_iter = dataloader.__iter__()
    data = next(data_iter)
    data = scatter(data, [GPU_ID])[0]
    return data

# Build model
def model_build():
    model = build_detector(CFG.model, test_cfg=CFG.get("test_cfg"))
    if FP16 is not None:
        wrap_fp16_model(model)
    _ = model.load_state_dict(torch.load(CKPT)["state_dict"], strict=False)
    if CFG.use_tensorrt:
        if CFG.trt_paths["backbone"] is not None:
            model.img_backbone = TRTModule()
            model.img_backbone.load_state_dict(torch.load(CFG.trt_paths["backbone"]))
        if CFG.trt_paths["neck"] is not None:
            model.img_neck = TRTModule()
            model.img_neck.load_state_dict(torch.load(CFG.trt_paths["neck"]))
        if CFG.trt_paths["encoder"] is not None:
            model.head.anchor_encoder = TRTModule()
            model.head.anchor_encoder.load_state_dict(torch.load(CFG.trt_paths["encoder"]))
        if CFG.trt_paths["temp_encoder"] is not None:
            model.head.temp_anchor_encoder = TRTModule()
            model.head.temp_anchor_encoder.load_state_dict(torch.load(CFG.trt_paths["temp_encoder"]))
    model = model.cuda(GPU_ID).eval()
    assert model.use_deformable_func, "Please compile deformable aggregation first !!!"
    return model

def profiler_pytorch(data, model):
    with torch.no_grad():
        with profile(activities=ACTIVITY, schedule=SCHEDULE, on_trace_ready=TRACER, record_shapes=True) as prof:
            for step in range(WAIT+WARMUP+ACTIVE):
                if step < WAIT:
                    print(f"WAIT {step+1}")
                elif step >= WAIT and step < WAIT+WARMUP:
                    print(f"WARMUP {step-WAIT+1}")
                else:
                    print(f"ACTIVE {step-(WAIT+WARMUP)+1}")
                prof.step()
                out = model(return_loss=False, rescale=True, **data)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    with open('result_sparse4dv3_fp16.txt', 'w') as txt_file:
        txt_file.write(prof.key_averages().table())

def profiler_nsys(data, model):
    with torch.no_grad():
        with torch.cuda.profiler.profile():
            for step in range(2):
                if step == 1:
                    with torch.autograd.profiler.emit_nvtx():
                        out = model(return_loss=False, rescale=True, **data)
                else:
                    out = model(return_loss=False, rescale=True, **data)

def profiler_elapsed_time(data, model):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(f"Elapsed time: {elapsed_time:.3f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model")
    parser.add_argument("--choice", type=int, default=2, help="Profiler type.\n1. Pytorch\n2. Nsight")
    args = parser.parse_args()
    data = dataset_build()
    model = model_build()
    if args.choice == 1:
        profiler_pytorch(data, model)
    elif args.choice == 2:
        profiler_nsys(data, model)