import git
import sys
import json

def main():

    # Updating files
    git_dir = '/app/csvs-btp-public/'
    g = git.cmd.Git(git_dir)
    g.stash()
    g.pull()

    from detect_vehicles import detect_vehicles
    from reid_vehicles import reid_vehicles

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 app.py <input_file.json> <app/data/CSVS>")
        sys.exit(1)

    # Get the arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    print(f"input file: {input_file}")
    
    with open(input_file, 'r') as file:
        input_data = json.load(file)

    print(f"{input_data}")
    cam_ids = list(input_data.keys())
    videos = list(input_data.values())

    print("Processing started")
    # Print the arguments to verify
    for cam_id, video in zip(cam_ids, videos):
        print(f"cam id: {cam_id} video file: {video}")
        detect_vehicles(cam_id, video)
        print(f"{cam_id} processed")

    reid_vehicles(cam_ids, videos, output_dir)

if __name__ == "__main__":
    main()
    print("Processing complete.")
