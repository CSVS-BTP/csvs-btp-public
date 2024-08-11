# csvs-btp-public
run this on terminal
expects video present on a folder named output
creates results in output

docker run -it --rm=true -v ./:/app/csvs-btp-public/output/  --runtime=nvidia --gpus all vishaksagar/csvs:v5.0 python3 app.py output/input_file.json output/output_file.json
