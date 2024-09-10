# csvs-btp-public
run this on terminal
expects video present on a folder named output
creates results in CSVS folder

docker run -it --rm=true -v ./:/app/csvs-btp-public/output/  --runtime=nvidia --gpus all vishaksagar/csvs:v8.0 python3 app.py input_file.json app/data/CSVS

# Files
detect_vehicles.py - Code to detect, track and determine turns of vehicles, takes input video and outputs csv of results

reid_vehicles.py - Code to reid vehicles based on detections, takes as input cam_id, the processed_csvs (internally) and outputs matrices and images in CSVS folder

app.py - Main function that puts everything together. Takes input.json and app/data/CSVS (path to output)

# Models used
yolov8s
Inception_v3 (weights called internally from pytorch in detect vehicles to extract features)

# Hardware requirements:
The above implementation is very light and can be run on any system with or without GPU.
## Benchmarks:
On submission system with specs (CPU - Core i9, GPU - RTX4090, RAM - 64GB, SATA - 500GB) ratio is 1:0.75