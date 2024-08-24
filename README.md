# csvs-btp-public
run this on terminal
expects video present on a folder named output
creates results in output

docker run -it --rm=true -v ./:/app/csvs-btp-public/output/  --runtime=nvidia --gpus all vishaksagar/csvs:v7.0 python3 app.py output/input_file.json output/output_file.json

# Files
detect_vehicles.py - Code to detect and track vehicles, takes input video and outputs csv of results

detect_turns.py - Code to determine turns based on detections, takes as input cam_id, the processed_csvs (internally) and outputs outputs.json

app.py - Main function that puts everything together. Takes input.json and outputs output.json

# Models used
yolov8s

# Hardware requirements:
The above implementation is very light and can be run on any system with or without GPU.
## Benchmarks:
On AMD Ryzen 5 system with integrated 1GB Radeon graphics, video processing takes time in the ratio 1:5

On AMD Ryzen 7 system with NVIDIA GTX 1080 (8GB), ratio is 1:1

On submission system with specs (CPU - Core i9, GPU - RTX4090, RAM - 64GB, SATA - 500GB) ratio is 1:0.5

The post processing of turns takes a few seconds at most.