# csvs-btp-public
For IISC competition phase 1

# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install requirements.txt
pip install -r requirements.txt

# Run
python3 app.py <video_file> <csv_file>"
##Example:
python3 app.py Stn_HD_1_time_2024-05-18T07:30:02_004.mp4 counts.csv
