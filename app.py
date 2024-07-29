from detect_vehicles import detect_vehicles
import sys

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 app.py <video_file> <csv_file>")
        sys.exit(1)

    # Get the arguments
    video_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    # Print the arguments to verify
    print(f"Video file: {video_file}")
    print(f"CSV file: {csv_file}")
    print("Processing started")

    detect_vehicles(video_file)

if __name__ == "__main__":
    main()
    print("Processing complete.")

