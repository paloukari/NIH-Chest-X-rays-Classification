ufw allow 32001/tcp

mkdir -m 777 /data
mkfs.ext4 /dev/xvdc
echo '/dev/xvdc /data                   ext4    defaults,noatime        0 0' >> /etc/fstab
mount /data

apt-get update
apt install python3-pip -y
apt install unzip -y

pip3 install kaggle

export KAGGLE_USERNAME=$1                                                                                                                               export KAGGLE_KEY=$2

kaggle datasets download -d nih-chest-xrays/data -p /data
unzip /data/data.zip -d /data
rm /data/data.zip
unzip '/data/*.zip' -d /data

docker build -t chest_x_rays_dev -f Dockerfile.dev .

