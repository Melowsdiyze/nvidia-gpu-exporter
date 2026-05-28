# Add docker repository
echo "Add docker repository"

# Install docker depedencies
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

# Add docker repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Updating Repository
apt update -y

# Installing docker
sudo apt install docker-ce -y

# Enable Docker
echo "Enabling Docker"
systemctl enable docker

# Restarting Docker
echo "Restarting Docker"
systemctl restart docker


# Set OS and Repo for nvidia repository
echo "Set OS and Repo for nvidia repository"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Updating repo
echo "Updating repo"
apt update -y

# Installing nvidia docker libs
echo "Installing nvidia docker libs"
apt install -y nvidia-docker2

# Restarting Docker
echo "Restarting Docker"
systemctl restart docker

# Running nvidia exporter as docker container
echo "Running nvidia exporter as docker container"
docker run --name nvidia-monitoring -d --gpus all -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:2.0.13-2.1.2-ubuntu20.04

# Updating configuration container into always restart if fail
echo "Updating configuration container into always restart if fail"
docker update --restart=always nvidia-monitoring