import os
import git
import sys
import json

def main():

    # Updating files
    git_dir = '../csvs-btp-public/'
    g = git.cmd.Git(git_dir)
    g.stash()
    g.pull()
    # print('git pull in app')

    from detect_vehicles import detect_vehicles
    from detect_turns import detect_turns

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 app.py <input_file.json> <output_file.json>")
        sys.exit(1)

    # Get the arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print(f"input file: {input_file}")
    print(f"output file: {output_file}")
    
    with open(input_file, 'r') as file:
        input_data = json.load(file)

    print(f"{input_data}")
    cam_id = list(input_data.keys())[0]
    cam_id_data = input_data[cam_id]
    vid_1_path = cam_id_data["Vid_1"]
    vid_2_path = cam_id_data["Vid_2"]

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
