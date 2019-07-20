#Install TX2 Tensorflow Container
#docker build --no-cache -f ./Dockerfile.tx2_tensorflow . -t tensorflow_tx2

# Jupyter Notebook Execution
#docker run \
#--name tensorflow \
#--privileged \
#-v "$PWD":/content/project \
#-v "/media/brent/data":/content/data \
#-p 8888:8888 \
#--rm \
#-ti tensorflow_tx2 \
#jupyter notebook --no-browser --port 8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='root' --notebook-dir=/content/

# Command Line Execution
docker run \
--name tensorflow \
--privileged \
-v "$PWD":/content/project \
-v "/media/brent/data":/content/data \
-p 8888:8888 \
--rm \
-ti tensorflow_tx2 \
/bin/sh -c 'cd content/project; python3 ./inference.py'
