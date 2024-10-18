import cv2
import easyocr
import matplotlib.pyplot as plt
import time

reader = easyocr.Reader(['en'], gpu=False)

video_path = 'images.png'
cap = cv2.VideoCapture(video_path)

frame_rate = 5   
frame_count = 0
total_text_detected = 0   
start_time = time.time()   

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  

     
    if frame_count % frame_rate == 0:
        results = reader.readtext(frame)
        detected_texts = 0

        
        for bbox, text, score in results:
            if score > 0.25:   
                detected_texts += 1
                 
                color = colors[detected_texts % len(colors)]   
                cv2.rectangle(frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), color, 2)
                cv2.putText(frame, f"{text} ({score:.2f})", tuple(map(int, bbox[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        total_text_detected += detected_texts   
        
         
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_count} - {detected_texts} Text(s) Detected")
        plt.axis('off')
        plt.show()

    frame_count += 1

cap.release()
 
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time if elapsed_time > 0 else 0

 
print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
print(f"FPS: {fps:.2f}")
print(f"Total Text Elements Detected: {total_text_detected}")