# mmd-auto-trace-3

## 構築

conda create -n mat3 pip python=3.9
conda activate mat3
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

`xxxxx/Anaconda3/envs/mat3/Lib/site-packages/bezier/extra-dll/bezier-2a44d276.dll` を ひとつ上の `bezier` フォルダに配置する
`xxxxx/Anaconda3/envs/mat3/Lib/site-packages/bezier/bezier-2a44d276.dll`

git submodule add https://github.com/miu200521358/AlphaPose src/AlphaPose 

(mat3) C:\MMD\mmd-auto-trace-3\src\AlphaPose>python setup.py build install

opendr のインストールに失敗するけど、とりあえず無視

## 学習モデルDL

### detector > tracker

JDE-1088x608
https://github.com/Zhongdao/Towards-Realtime-MOT#pretrained-model-and-baseline-models

## 3D-Multi-Person-Pose

git submodule add https://github.com/miu200521358/3D-Multi-Person-Pose

