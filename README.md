# csvs-btp-public
run this on terminal
expects video present on a folder named output
creates results in output

docker run -it --rm=true -v ./:/app/csvs-btp-public/output/  --runtime=nvidia --gpus all vishaksagar/csvs:v6 python3 app.py output/myfile.mp4 output/counts.csv
