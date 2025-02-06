#!/bin/bash

read -p "input base image name: " base_image_name
read -p "input user add image name: " user_add_image_name
read -sp "input user password: " password
echo ""
read -sp "input user password again: " password2
echo ""

if [ ${password} != ${password2} ]; then
    echo "Sorry, passwords do not match"
    exit;
fi

echo "  base image     : " ${base_image_name}
echo "  user add image : " ${user_add_image_name}
echo "  user           : " ${USER}
echo "  uid            : " ${UID}
echo "  gid            : " ${GROUPS}

while true; do
    read -p "Is it OK?[Yes/no]: " answer
    case $answer in
        '' | [Yy]* )
            break;
            ;;
        [Nn]* )
            echo "building stopped."
            exit;
            ;;
        * )
            echo "Please answer YES or NO."
    esac
done;

# Search docker command
if which nvidia-docker >/dev/null 2>&1; then
    echo "nvidia-docker found."
    DOCKER=nvidia-docker
elif which docker >/dev/null 2>&1; then
    echo "docker found."
    DOCKER=docker
else
    echo "Any docker command is not found."
    exit 1
fi

# $DOCKER build -t ${base_image_name} -f Dockerfile .
$DOCKER build -t ${user_add_image_name} -f Dockerfile-add-user . \
              --build-arg base=${base_image_name} \
              --build-arg user=${USER} --build-arg uid=${UID} \
              --build-arg gid=${GROUPS} --build-arg pass=${password}
