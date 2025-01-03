# Environment Setup 

```bash
conda create --name weaktr python=3.7
conda activate weaktr

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
pip install -r requirements.txt
```

Then, install [mmcv==1.4.0](https://github.com/open-mmlab/mmcv) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) following the official instruction.

```bash
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmsegmentation==0.30.0
```
And install `pydensecrf` from source.

```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```