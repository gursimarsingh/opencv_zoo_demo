import cv2 as cv
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import argparse
from dexined import Dexined
import time
import re
opencv_python_version = lambda str_version: tuple(map(int, re.findall(r'\d+', str_version)))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from mp_handpose import MPHandPose
from facial_fer_model import FacialExpressionRecog
from yunet import YuNet
from vittrack import VitTrack
from mp_palmdet import MPPalmDet
from yolox import YoloX
from utils import *

class_names_yolo = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def get_args_parser(func_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)

    ### handpose
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                        help='''Choose one of the backend-target pair to run this demo:
                            {:d}: (default) OpenCV implementation + CPU,
                            {:d}: CUDA + GPU (CUDA),
                            {:d}: CUDA + GPU (CUDA FP16),
                            {:d}: TIM-VX + NPU,
                            {:d}: CANN + NPU
                        '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument('--confidence_hp', type=float, default=0.9,
                        help='Filter out hands of confidence < conf_threshold_hp')
    ###

    ### object detection
    parser.add_argument('--confidence_objDet', default=0.5, type=float,
                        help='Class confidence of object detection')
    parser.add_argument('--nms', default=0.5, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--obj', default=0.5, type=float,
                        help='Enter object threshold')

    ###
    args, _ = parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

# Add helper to compute dynamic text scale and thickness based on resolution
def get_text_params(img, base_scale=0.4, base_thickness=2, ref_width=640):
    """Compute font scale and thickness proportional to image width"""
    h, w = img.shape[:2]
    scale = base_scale * (w / ref_width)
    thickness = max(1, int(base_thickness * (w / ref_width)))
    return scale, thickness

def handpose_detection(frame, palm_detector, handpose_detector, paused=False):

    tm = cv.TickMeter()

    palms = palm_detector.infer(frame)
    hands = np.empty(shape=(0, 132))
    tm.start()
    for palm in palms:
        handpose = handpose_detector.infer(frame, palm)
        if handpose is not None:
            hands = np.vstack((hands, handpose))
    tm.stop()
    frame, view_3d = visualize(frame, hands)
    # Overlay mode and FPS dynamically, regardless of detection
    scale, thickness = get_text_params(frame)
    h, w = frame.shape[:2]
    margin = int(10 * (w / 640))
    texts = ['Mode: Handpose', f'FPS: {tm.getFPS():.2f}']
    y = margin
    for t in texts:
        (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += th
        cv.putText(frame, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)
        y += margin // 2
    tm.reset()

    # Show 'Paused' overlay in top-right corner if paused
    if paused:
        scale, thickness = get_text_params(frame)
        (tw, th), _ = cv.getTextSize('Paused', cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv.putText(frame, 'Paused', (frame.shape[1] - tw - 20, th + 10), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness)
    cv.imshow("OpenCV CVPR 2025", frame)
        
def edge_detection(frame,dexined, paused=False):
    tm = cv.TickMeter()

    tm.start()
    result = dexined.infer(frame)
    tm.stop()

    label = 'Inference time: {:.2f} ms, FPS: {:.2f}'.format(tm.getTimeMilli(), tm.getFPS())
    # Dynamic overlay of mode and inference stats
    scale, thickness = get_text_params(result)
    h, w = result.shape[:2]
    margin = int(10 * (w / 640))
    texts = ['Mode: Edge Detection', label]
    y = margin
    for t in texts:
        (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += th
        cv.putText(result, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), thickness)
        y += margin // 2
    
    tm.reset()

    # Show 'Paused' overlay in top-right corner if paused
    if paused:
        scale, thickness = get_text_params(result)
        (tw, th), _ = cv.getTextSize('Paused', cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv.putText(result, 'Paused', (result.shape[1] - tw - 20, th + 10), cv.FONT_HERSHEY_SIMPLEX, scale, (255, 0, 255), thickness)
    cv.imshow("OpenCV CVPR 2025", result)
    
def scale_coords(input_shape, coords, original_shape, scale):
    """
    input_shape: (H, W) of resized image
    coords: bounding box [x1, y1, x2, y2]
    original_shape: original frame shape (H, W)
    scale: float scaling factor used in letterbox()
    """
    coords = coords / scale
    coords = np.clip(coords, [0, 0, 0, 0], [original_shape[1], original_shape[0], original_shape[1], original_shape[0]])
    return coords.astype(int)

def object_detection(frame, model_net, class_names=class_names_yolo, paused=False):
    input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_blob, letterbox_scale = letterbox(input_blob)
    
    tm = cv.TickMeter()
    tm.start()
    preds = model_net.infer(input_blob)  # shape: (N, 6) → [x1, y1, x2, y2, conf, class_id]
    tm.stop()
    
    if paused:
        scale, thickness = get_text_params(frame)
        (tw, th), _ = cv.getTextSize('Paused', cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv.putText(frame, 'Paused', (frame.shape[1] - tw - 20, th + 10), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness)
    
    if len(preds) == 0:
        # No detections: dynamic overlay of mode and FPS
        scale, thickness = get_text_params(frame)
        h, w = frame.shape[:2]
        margin = int(10 * (w / 640))
        texts = ['Mode: Object Detection', f'FPS: {tm.getFPS():.2f}']
        y = margin
        for t in texts:
            (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
            y += th
            cv.putText(frame, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)
            y += margin // 2
        cv.imshow("OpenCV CVPR 2025", frame)
        return None, None

    preds = np.array(preds)
    
    # Separate humans (class_id == 0) and others
    human_preds = preds[preds[:, 5] == 0]
    if len(human_preds) > 0:
        best = max(human_preds, key=lambda x: x[4])
    else:
        best = max(preds, key=lambda x: x[4])

    class_id = int(best[5])
    class_name = class_names[class_id]

    # Scale bbox
    x1, y1, x2, y2 = scale_coords(input_blob.shape[1:3], best[:4].copy(), frame.shape[:2], letterbox_scale)

    # Convert to (x, y, w, h)
    w = x2 - x1
    h = y2 - y1
    roi_tracker = (x1, y1, w, h)

    # Draw detections and dynamic overlay
    img = vis(preds, frame, letterbox_scale)
    scale, thickness = get_text_params(img)
    h, w = img.shape[:2]
    margin = int(10 * (w / 640))
    texts = ['Mode: Object Detection', f'FPS: {tm.getFPS():.2f}']
    y = margin
    for t in texts:
        (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += th
        cv.putText(img, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)
        y += margin // 2
    cv.imshow("OpenCV CVPR 2025", img)
    return roi_tracker, class_name

def human_segmentation(frame,model,w,h, paused=False):
    _frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    _frame = cv.resize(_frame, dsize=(192, 192))
    tm = cv.TickMeter()
    # Inference
    tm.start()
    result = model.infer(_frame)
    tm.stop()
    result = cv.resize(result[0, :, :], dsize=(w, h), interpolation=cv.INTER_NEAREST)

    # Draw results on the input image
    frame = visualize_humanSeg(frame, result)
    # Dynamic overlay of mode and FPS
    scale, thickness = get_text_params(frame)
    h, w = frame.shape[:2]
    margin = int(10 * (w / 640))
    texts = ['Mode: Human Segmentation', f'FPS: {tm.getFPS():.2f}']
    y = margin
    for t in texts:
        (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += th
        cv.putText(frame, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)
        y += margin // 2

    # Show 'Paused' overlay in top-right corner if paused
    if paused:
        scale, thickness = get_text_params(frame)
        (tw, th), _ = cv.getTextSize('Paused', cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv.putText(frame, 'Paused', (frame.shape[1] - tw - 20, th + 10), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness)

    # Visualize results in a new window
    cv.imshow("OpenCV CVPR 2025", frame)
    tm.reset()

def facial_expression(frame,detect_model,fer_model, paused=False):
    tm = cv.TickMeter()
    tm.reset()
    # Inference
    tm.start()
    status, dets, fer_res = process(detect_model, fer_model, frame)
    tm.stop()
    if status:
        frame = visualize_face(frame, dets, fer_res)
    # Dynamic overlay of mode and FPS
    scale, thickness = get_text_params(frame)
    h, w = frame.shape[:2]
    margin = int(10 * (w / 640))
    texts = ['Mode: Facial Expression', f'FPS: {tm.getFPS():.2f}']
    y = margin
    for t in texts:
        (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += th
        cv.putText(frame, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)
        y += margin // 2

    # Show 'Paused' overlay in top-right corner if paused
    if paused:
        scale, thickness = get_text_params(frame)
        (tw, th), _ = cv.getTextSize('Paused', cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv.putText(frame, 'Paused', (frame.shape[1] - tw - 20, th + 10), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness)
    cv.imshow("OpenCV CVPR 2025", frame)
def tracker(frame,model,class_name, paused=False):
    tm = cv.TickMeter()
    # Inference
    tm.start()
    isLocated, bbox, score = model.infer(frame)
    tm.stop()
    # Visualize with dynamic overlay
    frame = visualize_tracker(frame, bbox, score, isLocated, class_name)
    scale, thickness = get_text_params(frame)
    h, w = frame.shape[:2]
    margin = int(10 * (w / 640))
    texts = ['Mode: Tracker', f'FPS: {tm.getFPS():.2f}']
    y = margin
    for t in texts:
        (tw, th), _ = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        y += th
        cv.putText(frame, t, (margin, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness)
        y += margin // 2
    tm.reset()

    # Show 'Paused' overlay in top-right corner if paused
    if paused:
        scale, thickness = get_text_params(frame)
        (tw, th), _ = cv.getTextSize('Paused', cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv.putText(frame, 'Paused', (w - tw - 20, th + 10), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness)
    cv.imshow("OpenCV CVPR 2025", frame)
    
def main(func_args=None):
    args = get_args_parser(func_args)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return

    print("Loading models...")
    # Load all models
    dexined = Dexined(modelPath='./models/edge_detection_dexined_2024sep.onnx')

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    palm_detector = MPPalmDet(
        modelPath='./models/palm_detection_mediapipe_2023feb.onnx',
        nmsThreshold=0.3,
        scoreThreshold=0.6,
        backendId=backend_id,
        targetId=target_id)

    handpose_detector = MPHandPose(
        modelPath='./models/handpose_estimation_mediapipe_2023feb.onnx',
        confThreshold=args.confidence_hp,
        backendId=backend_id,
        targetId=target_id)

    model_net = YoloX(
        modelPath='./models/object_detection_yolox_2022nov.onnx',
        confThreshold=args.confidence_objDet,
        nmsThreshold=args.nms,
        objThreshold=args.obj,
        backendId=backend_id,
        targetId=target_id)

    detect_model = YuNet(modelPath='./models/face_detection_yunet_2023mar.onnx')

    fer_model = FacialExpressionRecog(
        modelPath='./models/facial_expression_recognition_mobilefacenet_2022july.onnx',
        backendId=backend_id,
        targetId=target_id)

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
 
    tracker_model = VitTrack(
        model_path='./models/object_tracking_vittrack_2023sep.onnx',
        backend_id=backend_id,
        target_id=target_id)

    # Read model modes from 'model_modes.txt' if available, else use default modes
    modes_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_modes.txt')
    if os.path.isfile(modes_file):
        with open(modes_file, 'r') as f:
            model_modes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        model_modes = ['handpose', 'object_det', 'facial_expression', 'tracker']

    current_mode_index = 0
    current_mode = model_modes[current_mode_index]

    print("Press space to toggle between pause/resume, 'esc' to quit, and left/right arrow to switch between modes")

    start_time = time.time()
    paused = False
    tracker_toggle=True
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("No frame captured.")
            break

        key = cv.waitKey(1)
        #print(key)
        #exit(-1)
        if key == ord('q') or key == 27:
            break

        elif key == ord(' '):  # pause/unpause
            paused = not paused
            state = "Paused" if paused else "Resumed"
            print(f"{state} mode: {current_mode}")
            if not paused:
                start_time = time.time()  # reset timer on resume

        if paused:
            start_time = time.time()
        # Time-based auto-switch every 15s
        if time.time() - start_time >= 15:
            tracker_toggle = True
            current_mode_index = (current_mode_index + 1) % len(model_modes)
            current_mode = model_modes[current_mode_index]
            start_time = time.time()
        # Manual switch with right/left arrow (supports macOS codes)
        elif key in (83, 3):  # Right arrow key
            tracker_toggle = True
            start_time = time.time()
            current_mode_index = (current_mode_index + 1) % len(model_modes)
            current_mode = model_modes[current_mode_index]
            print(f"Switched to: {current_mode}")
            start_time = time.time()
        elif key in (81, 2):  # Left arrow key
            tracker_toggle = True
            start_time = time.time()
            current_mode_index = (current_mode_index - 1) % len(model_modes)
            current_mode = model_modes[current_mode_index]
            print(f"Switched to: {current_mode}")
            start_time = time.time()
        
        # Pass paused state to each mode function
        if current_mode == 'handpose':
            handpose_detection(frame, palm_detector, handpose_detector, paused)
        elif current_mode == 'edge':
            edge_detection(frame, dexined, paused)
        elif current_mode == 'object_det':
            object_detection(frame, model_net, paused=paused)
        elif current_mode == 'facial_expression':
            facial_expression(frame, detect_model, fer_model, paused)
        elif current_mode == 'tracker':
            if tracker_toggle:
                roi, class_name = object_detection(frame, model_net, class_names_yolo, paused=paused)
                if roi is not None:
                    tracker_model.init(frame, roi)
                    tracker_toggle = not tracker_toggle
            else:
                if roi is not None:
                    tracker(frame, tracker_model, class_name, paused=paused)
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
