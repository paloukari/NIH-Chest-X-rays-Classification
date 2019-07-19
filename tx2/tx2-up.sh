#Install pythia tx2 Container
#docker build --no-cache -f ./Dockerfile.tx2_tensorflow . -t tensorflow_tx2

docker run \
--name tensorflow \
--privileged \
-v "$PWD":/content/project \
-p 8888:8888 \
--rm \
-ti tensorflow_tx2 \
jupyter notebook --no-browser --port 8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='root' --notebook-dir=/content/
