import subprocess
import sys
import json
from detect_vehicles import detect_vehicles
from detect_turns import detect_turns

def main():
    
    # Refreshing git instance
    subprocess.run(['git', 'pull'], capture_output=False, text=False)

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 app.py <input_file.json> <output_file.json>")
        sys.exit(1)

    # Get the arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, 'r') as file:
        input_data = json.load(file)
    cam_id_data = input_data["Cam_ID"]
    vid_1_path = cam_id_data["Vid_1"]
    vid_2_path = cam_id_data["Vid_2"]
    cam_id_1 = vid_1_path.split('/')[-1].split('.')[0][:-6]
    cam_id_2 = vid_2_path.split('/')[-1].split('.')[0][:-6]
    if cam_id_1 != cam_id_2:
        print('Print cam ids mismatch')
        print(cam_id_1)
        print(cam_id_2)
    else:
        cam_id = cam_id_1

    print("Processing started")
    # Print the arguments to verify
    print(f"Video file: {vid_1_path}")
    detect_vehicles(vid_1_path, 'vid_1.csv')
    print("Vid 1 processsed")
    print(f"Video file: {vid_2_path}")
    detect_vehicles(vid_2_path, 'vid_2.csv')
    print("Vid 2 processsed")

    detect_turns(cam_id, output_file)
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()
    print("Processing complete.")
