#!/usr/bin/env bash

sudo apt update && sudo apt upgrade;

# install torchserve + dependencies
cd ~/samserve/serve;
python ts_scripts/install_dependencies.py --cuda=cu117; # vm actually has 12.0 ...
pip install torchserve torch-model-archiver torch-workflow-archiver;

# install segment-anything
cd ~/samserve/segment-anything;
pip install -e .;

# make model archive file
cd ~;
mkdir models && mkdir samweights;
curl -o samweights/sam_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth;
curl -o samweights/sam_l.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth;
curl -o samweights/sam_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth;

for m in {'sam_b','sam_l','sam_h'}; do
    for h in {'auto_maskgen','predict'}; do
        torch-model-archiver --model-name $m'_'$h \
            --serialized-file 'samweights/'$m'.pth' \
            --model-file 'samserve/'$m'.py' \
            --handler 'samserve/'$h'.py' \
            --export-path models -v 0.1 -f;
    done;
done

# # ssl stuff (note, requires interactive)
# cd ~;
# mkdir keys;
# openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout keys/mykey.key -out keys/mycert.pem;
# curl -4 ifconfig.co >> keys/ipaddress.txt;

# launch torchserve
cd ~;
torchserve --model-store models --start --models all --ts-config samserve/config.properties --no-config-snapshots;
# if you want specific models, e.g. only the _h ones:
# torchserve --model-store models --start \
#     --models sam_h_auto_maskgen=sam_h_auto_maskgen.mar sam_h_predict=sam_h_predict.mar  \
#     --ts-config samserve/config.properties --no-config-snapshots;