import cv2
from ultralytics import YOLO
import serial
import csv
import pynmea2

# Initialize YOLO model
model = YOLO('weights/rdd_ncnn_model')

# Initialize video capture
cap = cv2.VideoCapture('video_rdd.mp4')
if not cap.isOpened():
    print("Unable to read video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, (frame_width, frame_height))

# Open serial connection for GPS Module
ser = serial.Serial('COM3', 9600, timeout=1)

# Open CSV file to write GPS data
with open('rdd_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Latitude', 'Longitude', 'Object'])  # Write the header

    while True:
        ret, frame = cap.read()

        # Apply CLAHE to the frame
        clahe_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        clahe_frame[:, :, 0] = clahe.apply(clahe_frame[:, :, 0])
        clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_Lab2RGB)

        # Perform object detection
        results = model(clahe_frame)

        # Annotate frame with detection results
        annotated_frame = results[0].plot()

        # Write annotated frame to output qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq 
        output.write(annotated_frame)

        # Check if objects are detected
        if len(results[0].boxes) > 0:
            # Read GPS data from serial
            gps_data = ser.readline().decode('utf-8').strip()
            
            # Parse GPS data using pynmea2
            if gps_data.startswith('$GNGGA'):
                msg = pynmea2.parse(gps_data)
                latitude = str(round(msg.latitude, 7))
                longitude = str(round(msg.longitude, 7))
                
                # Extract object labels
                object_labels = [model.names[int(box.cls)] for box in results[0].boxes]
                
                # Write the data to CSV file
                for label in object_labels:
                    writer.writerow([latitude, longitude, label])

        # Display the annotated frame (optional)
        cv2.imshow("Inference", annotated_frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Release resources
cap.release()
output.release()
ser.close()
cv2.destroyAllWindows()