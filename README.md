# csvs-btp-public
run this on terminal
expects video present on a folder named output
creates results in output

docker run -it --rm=true -v ./:/app/csvs-btp-public/output/  --runtime=nvidia --gpus all vishaksagar/csvs:v7.0 python3 app.py output/input_file.json output/output_file.json

# Files
detect_vehicles.py - Code to detect and track vehicles, takes input video and outputs csv of results
detect_turns.py

# Models used
yolov8s
