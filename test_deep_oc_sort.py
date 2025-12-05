import pdb
import os
import shutil
import time

import torch
import cv2
import numpy as np

import dataset
import utils
from external.adaptors import detector
from trackers import integrated_ocsort_embedding as tracker_module
from external.utils.visualize import plot_tracking


def get_main_args():
    parser = tracker_module.args.make_parser()
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="run post-processing linear interpolation.",
    )
    parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument(
        "--alpha_fixed_emb",
        type=float,
        default=0.95,
        help="Alpha fixed for EMA embedding",
    )
    parser.add_argument("--emb_off", action="store_true")
    parser.add_argument("--cmc_off", action="store_true")
    parser.add_argument("--aw_off", action="store_true")
    parser.add_argument("--aw_param", type=float, default=0.5)
    parser.add_argument("--new_kf_off", action="store_true")
    parser.add_argument("--grid_off", action="store_true")
    parser.add_argument("--visual_off", action="store_true")
    
    args = parser.parse_args()

    args.visual_folder = ""
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
        args.visual_folder = "MOT17-val"
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
        args.visual_folder = "MOT20-val"
    elif args.dataset == "dance":
        args.result_folder = os.path.join(args.result_folder, "DANCE-val")
        args.visual_folder = "DANCE-val"
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args


def main():
    np.set_printoptions(suppress=True, precision=5)
    # Set dataset and detector
    args = get_main_args()

    if args.dataset == "mot17":
        if args.test_dataset:
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
        else:
            detector_path = "external/weights/bytetrack_ablation.pth.tar"
        size = (800, 1440)
    elif args.dataset == "mot20":
        if args.test_dataset:
            detector_path = "external/weights/bytetrack_x_mot20.tar"
            size = (896, 1600)
        else:
            # Just use the mot17 test model as the ablation model for 20
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
            size = (800, 1440)
    elif args.dataset == "dance":
        # Same model for test and validation
        detector_path = "external/weights/ocsort_dance_model.pth.tar"
        size = (800, 1440)
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")
    det = detector.Detector("yolox", detector_path, args.dataset)
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size)

    # Set up tracker
    oc_sort_args = dict(
        args=args,
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
        w_association_emb=args.w_assoc_emb,
        alpha_fixed_emb=args.alpha_fixed_emb,
        embedding_off=args.emb_off,
        cmc_off=args.cmc_off,
        aw_off=args.aw_off,
        aw_param=args.aw_param,
        new_kf_off=args.new_kf_off,
        grid_off=args.grid_off,
    )
    tracker = tracker_module.ocsort.OCSort(**oc_sort_args)
    results = {}
    frame_count = 0
    total_time = 0

    # 在循环开始前添加输出目录设置
    output_dir = f"/workspace/Deep-OC-SORT/results/visual/{args.visual_folder}/{args.exp_name}"  # 可以根据需要修改路径
    os.makedirs(output_dir, exist_ok=True)

    # See __getitem__ of dataset.MOTDataset
    for (img, np_img), label, info, idx in loader:
        # Frame info
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]

        # Hacky way to skip SDP and DPM when testing
        if "FRCNN" not in video_name and args.dataset == "mot17":
            continue
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []
        img = img.cuda()

        # Initialize tracker on first frame of a new video
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            tracker.dump_cache()
            tracker = tracker_module.ocsort.OCSort(**oc_sort_args)
            
            # 为每个视频创建单独的输出目录
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

        start_time = time.time()

        # Nx5 of (x1, y1, x2, y2, conf), pass in tag for caching
        pred = det(img, tag)
        if pred is None:
            continue
        # Nx5 of (x1, y1, x2, y2, ID)
        targets = tracker.update(pred, img, np_img[0].numpy(), tag)
        tlwhs, ids = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)

        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids))
        
        # ============ 新增：保存每一帧的可视化结果 ============
        if not args.visual_off:
            try:
                # 获取当前帧图像
                current_frame = np_img[0].numpy()
                
                # 如果图像格式需要转换 (例如从RGB到BGR)
                if current_frame.shape[0] == 3:  # 如果是CHW格式
                    current_frame = current_frame.transpose(1, 2, 0)  # 转为HWC
                if current_frame.dtype == np.float32:  # 如果是float格式
                    current_frame = (current_frame * 255).astype(np.uint8)
                
                # 绘制跟踪结果
                if len(tlwhs) > 0:
                    # 假设您有plot_tracking函数 (从之前的代码)
                    fps = frame_count / (total_time + 1e-9)
                    result_img = plot_tracking(current_frame, tlwhs, ids, 
                                            frame_id=frame_id, fps=fps)
                else:
                    result_img = current_frame.copy()
                    # 在没有目标时也显示帧信息
                    fps = frame_count / (total_time + 1e-9)
                    cv2.putText(result_img, f'frame: {frame_id} fps: {fps:.2f} num: 0',
                            (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                # 保存图像
                video_output_dir = os.path.join(output_dir, video_name)
                output_filename = f"frame_{frame_id:06d}.jpg"
                output_path = os.path.join(video_output_dir, output_filename)
                cv2.imwrite(output_path, result_img)

                    
            except Exception as e:
                print(f"Error saving frame {frame_id}: {e}")
            # ======================================================

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    # Save detector results
    det.dump_cache()
    tracker.dump_cache()

    # Save for all sequences
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    if args.post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        utils.dti(post_folder_data, post_folder_data)
        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")


def draw(name, pred, i):
    pred = pred.cpu().numpy()
    name = os.path.join("data/mot/train", name)
    img = cv2.imread(name)
    for s in pred:
        p = np.round(s[:4]).astype(np.int32)
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 3)
    for s in pred:
        p = np.round(s[:4]).astype(np.int32)
        cv2.putText(
            img,
            str(int(round(s[4], 2) * 100)),
            (p[0] + 20, p[1] + 20),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            thickness=3,
        )
    cv2.imwrite(f"debug/{i}.png", img)


if __name__ == "__main__":
    main()
