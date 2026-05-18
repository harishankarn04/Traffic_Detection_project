import cv2
import numpy as np
import onnxruntime as ort
import os
import argparse

def run_inference(image, sess, class_names):
    img_h, img_w = image.shape[:2]
    
    # Preprocess
    input_size = (640, 640)
    resized = cv2.resize(image, input_size)
    blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis]
    
    # Inference
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: blob})[0]
    if len(out.shape) == 3:
        out = out[0].T
        
    boxes, scores, class_ids = [], [], []
    for row in out:
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        score = classes_scores[class_id]
        if score > 0.4:
            boxes.append(row[:4])
            scores.append(float(score))
            class_ids.append(class_id)
            
    # NMS
    output_image = resized.copy()
    if len(boxes) > 0:
        nms_boxes = []
        for b in boxes:
            cx, cy, bw, bh = b
            nms_boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
            
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, 0.4, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = nms_boxes[i]
                cid = class_ids[i]
                label = f"{class_names.get(cid, 'vehicle')} {scores[i]:.2f}"
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output_image, label, (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return output_image

def process_frame(frame, sess_base, sess_custom, coco_classes, custom_classes):
    res_base = run_inference(frame, sess_base, coco_classes)
    res_custom = run_inference(frame, sess_custom, custom_classes)

    cv2.putText(res_base, "BASE MODEL (COCO)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(res_custom, "FINE-TUNED (ATCS) - Detects Rickshaws!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    return np.hstack((res_base, res_custom))

def main():
    parser = argparse.ArgumentParser(description="Compare Base YOLO vs ATCS Fine-Tuned YOLO")
    parser.add_argument("image_path", help="Path to a local image file (e.g. an Indian traffic intersection)")
    args = parser.parse_args()

    input_path = args.image_path
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    # COCO classes for base model (2=car, 3=motorcycle, 5=bus, 7=truck)
    coco_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 0: 'person'}
    
    # Custom classes for fine-tuned model
    custom_classes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle', 4: 'auto_rickshaw', 5: 'ambulance', 6: 'fire_truck', 7: 'police'}

    print("Loading models...")
    sess_base = ort.InferenceSession("dl/yolov8n.onnx")
    sess_custom = ort.InferenceSession("dl/yolov8n_traffic.onnx")

    # Check if input is video or image
    is_video = any(input_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv'])

    if not is_video:
        original_img = cv2.imread(input_path)
        if original_img is None:
            print("Failed to load image.")
            return
        
        print("Processing image...")
        combined = process_frame(original_img, sess_base, sess_custom, coco_classes, custom_classes)
        out_path = "results_graphs/model_comparison.png"
        cv2.imwrite(out_path, combined)
        print(f"[SUCCESS] Saved side-by-side comparison to {out_path}")
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Failed to open video.")
            return
        
        print("Processing video... Press 'q' to stop early and save the snippet.")
        
        # Prepare video writer
        out_path = "results_graphs/model_comparison_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(out_path, fourcc, 15.0, (1280, 640)) # 640x640 * 2 horizontally

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            combined = process_frame(frame, sess_base, sess_custom, coco_classes, custom_classes)
            out_vid.write(combined)
            
            cv2.imshow("Model Comparison (Base vs Fine-Tuned)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out_vid.release()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] Saved side-by-side video comparison to {out_path}")

if __name__ == "__main__":
    main()
