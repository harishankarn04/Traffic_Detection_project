import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
import matplotlib.pyplot as plt

def run_inference(image, sess):
    input_size = (640, 640)
    resized = cv2.resize(image, input_size)
    blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis]
    
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
            
    final_class_ids = []
    final_class_ids, final_scores = [], []
    if len(boxes) > 0:
        nms_boxes = []
        for b in boxes:
            cx, cy, bw, bh = b
            nms_boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
            
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, 0.4, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                final_class_ids.append(class_ids[i])
                final_scores.append(scores[i])
                
    return final_class_ids, final_scores

def main():
    video_dir = "results_graphs/test_video"
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if not videos:
        print(f"No videos found in {video_dir}")
        return

    print("Loading models...")
    sess_base = ort.InferenceSession("dl/yolov8n.onnx")
    sess_custom = ort.InferenceSession("dl/yolov8n_traffic.onnx")

    # Process at 1.0 FPS for fast graph generation
    target_fps = 1.0

    # Cumulative stats
    total_base_veh = 0
    total_custom_veh = 0
    total_base_bike = 0
    total_custom_bike = 0
    total_base_auto = 0
    total_custom_auto = 0
    
    # Confidence trackers
    base_scores_list = []
    custom_scores_list = []

    times = []
    base_counts = []
    custom_counts = []
    
    global_time = 0

    print(f"Found {len(videos)} videos. Processing at {target_fps} FPS. This will take 20-30 minutes...")

    for vid in videos:
        print(f"\n--- Processing {os.path.basename(vid)} ---")
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            print(f"Failed to open {vid}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        skip_frames = max(1, int(round(fps / target_fps)))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % skip_frames == 0:
                # Base Model (2=car, 3=motorcycle, 5=bus, 7=truck)
                base_ids, base_conf = run_inference(frame, sess_base)
                base_valid = [cid for cid in base_ids if cid in [2, 3, 5, 7]]
                base_bikes = base_ids.count(3)
                if len(base_conf) > 0: base_scores_list.extend(base_conf)
                
                # Custom Model (0=car, 1=bus, 2=truck, 3=motorcycle, 4=auto_rickshaw)
                custom_ids, custom_conf = run_inference(frame, sess_custom)
                custom_valid = [cid for cid in custom_ids if cid in [0, 1, 2, 3, 4]]
                custom_bikes = custom_ids.count(3)
                custom_autos = custom_ids.count(4)
                if len(custom_conf) > 0: custom_scores_list.extend(custom_conf)

                total_base_veh += len(base_valid)
                total_custom_veh += len(custom_valid)
                total_base_bike += base_bikes
                total_custom_bike += custom_bikes
                total_custom_auto += custom_autos
                # base model doesn't have auto_rickshaw

                times.append(global_time)
                base_counts.append(len(base_valid))
                custom_counts.append(len(custom_valid))

                global_time += (1.0 / target_fps)
                
                if int(global_time) % 60 == 0 and int(global_time) > 0 and (global_time - int(global_time)) < 0.1:
                    print(f"Processed {int(global_time)} seconds of total footage...")

            frame_count += 1
        cap.release()

    print("\nFinished processing all videos! Generating aggregated graphs...")

    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Bar Chart Comparison (Total Detections Across All Frames)
    categories = ['Total Vehicles', 'Motorcycles', 'Auto-Rickshaws']
    base_stats = [total_base_veh, total_base_bike, total_base_auto]
    custom_stats = [total_custom_veh, total_custom_bike, total_custom_auto]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_stats, width, label='Base YOLOv8n', color='#e74c3c')
    rects2 = ax.bar(x + width/2, custom_stats, width, label='Fine-Tuned ATCS', color='#2ecc71')

    ax.set_ylabel('Total Bounding Boxes Detected (Millions)', fontsize=12)
    ax.set_title('Cumulative Detections Across All 3 Videos', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)

    # Format y-axis to millions for readability
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val/1000000:.1f}M" if val >= 1000000 else f"{val/1000:.0f}K"))

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(height):,}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results_graphs/batch_cumulative_bar_chart.png', dpi=300)
    plt.close()

    # 3. New Confidence Comparison Chart
    avg_base_conf = (sum(base_scores_list) / len(base_scores_list) * 100) if base_scores_list else 0
    avg_custom_conf = (sum(custom_scores_list) / len(custom_scores_list) * 100) if custom_scores_list else 0

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    bars = ax3.bar(['Base YOLOv8n', 'Fine-Tuned ATCS'], [avg_base_conf, avg_custom_conf], color=['#e74c3c', '#2ecc71'], width=0.5)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax3.set_ylabel('Average Detection Confidence (%)', fontsize=12)
    ax3.set_title('Overall Model Accuracy/Confidence Comparison', fontsize=14, pad=15)
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results_graphs/batch_confidence_comparison.png', dpi=300)
    plt.close()

    print("\nGraphs successfully generated and saved in 'results_graphs/' directory!")

    # 2. Line Graph (Time Series)
    def smooth(y, box_pts=24):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    plt.figure(figsize=(14, 6))
    plt.plot(times, smooth(custom_counts), label="Fine-Tuned ATCS Model", color='#2ecc71', linewidth=2)
    plt.plot(times, smooth(base_counts), label="Base YOLOv8n", color='#e74c3c', linewidth=2, linestyle='--')
    plt.title('Total Vehicles Detected Over Time (Across All Videos)', fontsize=14, pad=15)
    plt.xlabel('Cumulative Time (Seconds)', fontsize=12)
    plt.ylabel('Vehicles Detected Per Frame', fontsize=12)
    plt.legend(fontsize=12)
    plt.fill_between(times, smooth(custom_counts), smooth(base_counts), color='#2ecc71', alpha=0.1)
    
    out_line = "results_graphs/batch_time_series_line.png"
    plt.savefig(out_line, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Saved graphs to:")
    print(f"  1. results_graphs/batch_cumulative_bar_chart.png")
    print(f"  2. results_graphs/batch_time_series_line.png")
    print(f"  3. results_graphs/batch_confidence_comparison.png")

if __name__ == "__main__":
    main()
