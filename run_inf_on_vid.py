import argparse
import av
import torch
import pickle
import gc
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.structures.mask import encode_mask_results

def process_preds(result):
    result_keep_keys = ['pred_instances', 'pred_ds']
    instance_keep_keys = ['masks', 'bboxes', 'labels', 'scores']
    for k in result.keys():
        if k not in result_keep_keys:
            del result[k]

    result.pred_instances.masks = encode_mask_results(result.pred_instances.masks.detach().cpu())
    for k in result.pred_instances.keys():
        if k not in instance_keep_keys:
            del result.pred_instances[k]

    return result.to('cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', type=str, help='Path to video we want to process')
    parser.add_argument('-o', '--output_path', type=str, help='Path to dump results to')
    args = parser.parse_args()

    # Configuration
    config_file = 'configs/models/mask_rcnn/lg_ds_mask_rcnn_no_recon.py'  # Path to MMDetection config file
    checkpoint_file = 'weights/deepcvs2.pth'
    video_path = args.video_path # Path to input video
    output_pkl = args.output_path
    if not args.output_path.endswith('.pkl'):
        raise ValueError("Must write to pkl file")

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_options = dict(
        model = dict(
            detector=dict(
                test_cfg=dict(
                    rcnn=dict(score_thr=0.3),
                ),
            ),
        ),
    )
    model = init_detector(config_file, checkpoint_file, device=device, cfg_options=cfg_options)

    # Enable cuDNN benchmark mode for faster inference
    torch.backends.cudnn.benchmark = True

    # Open video using PyAV with GPU acceleration
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == 'video')
    total_frames = stream.frames

    # params, lists to store results
    batch_size = 16
    write_interval = 7500
    frame_results = []
    frame_buffer = []
    frame_id = 0

    with open(output_pkl, 'ab') as f:
        with tqdm(total=total_frames, desc='Processing frames') as pbar:
            for frame in container.decode(stream):
                img = frame.to_ndarray(format='bgr24')  # Convert to numpy array
                del frame  # Free decoded frame memory

                frame_buffer.append(img)

                # Process batch when full
                if len(frame_buffer) == batch_size:
                    with torch.no_grad():  # Disable gradient calculation
                        results = [process_preds(r) for r in inference_detector(model, frame_buffer)]  # Batch inference

                    frame_results.extend(results)  # Store results
                    frame_buffer.clear()  # Clear buffer

                    if len(frame_results) >= write_interval:
                        # Save batch results immediately to avoid memory overflow
                        pickle.dump(frame_results, f)
                        frame_results.clear()

                    # Free memory
                    torch.cuda.empty_cache()
                    gc.collect()

                frame_id += 1
                pbar.update(1)

        # Process remaining frames if any
        if frame_buffer:
            with torch.no_grad():
                results = [process_preds(r) for r in inference_detector(model, frame_buffer)]  # Batch inference
            frame_results.extend(results)
            pickle.dump(frame_results, f)

    # Close video container explicitly
    container.close()

    print(f'Results saved to {output_pkl}')
