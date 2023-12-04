# OpenSeeD and ControlNet Worker
## Install
```sh
git clone https://github.com/IDEA-Research/OpenSeeD.git
cd OpenSeeD
mkdir serve
cp openseed_controlnet_worker.py openseed_controlnet_test_message.py serve/
mkdir ckpt
wget https://github.com/IDEA-Research/OpenSeeD/releases/download/ade20k_swinl/openseed_ade20k_swinl.pt
mv openseed_ade20k_swinl.pt ckpt/
```
## Run
```sh
#start worker
python serve/openseed_controlnet_worker.py
#test worker
python serve/openseed_controlnet_test_message.py
```