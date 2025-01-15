# Sparse4Dv3
`Sparse4Dv3`の`Jetson AGX Orin`向け高速化およびROS2化

## 環境構築
- 下記環境で動作を確認
```
OS             : Ubuntu 20.04 LTS
CUDA Version   : 11.4
Docker version : 24.0.7, build 24.0.7-0ubuntu2~20.04.1
Device         : Jetson AGX Orin 64GB
```

### Dockerセットアップ
#### ユーザをdockerグループへ追加
- sudoなしでdockerを実行した際に下記の様なエラーが出る場合は、ユーザをdockerグループへ追加
```sh
$ docker ps
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.39/containers/create: dial unix /var/run/docker.sock: connect: permission denied. See 'docker run –help’.
```
- dockerグループが存在するか確認
```sh
cat /etc/group | grep docker
```
- 存在しない場合、dockerグループの作成
```sh
sudo groupadd docker
```
- dockerグループにユーザを追加
  - userを任意のユーザ名に変更
```sh
sudo usermod -aG docker user
```

#### .bashrcの設定
- dockerコンテナ内外を識別するため、以下の内容を`~/.bashrc`に追加　
  - コンテナ内に入ると文字が白からオレンジに変わる
```sh
if [ -f /.dockerenv ]; then
            PS1='\[\033[01;33m\](container) '$PS1
fi
```
### Dockerイメージのビルド
- `sparse4dv3/`に移動し下記コマンドを実行
```bash
$ cd docker
$ docker build -t sparse4dv3_nuscenes:ros .
```

### Dockerコンテナの起動
- `$HOME`以外の場所にデータを配置してある場合はマウント(`-v`)が必要
  - 例えば、`/mnt/external`ディレクトリ内にNuScenesデータセットを格納している場合、`-v /mnt/external/data:/mnt/external/data`オプションを追加
- コンテナ内でNsight Systemsを利用する場合、Nsight Systemsのフォルダパスを追加
  - Jetson Orinでは`opt/nvidia/`にNsight Systemsのフォルダが存在するはず
```bash
$ docker run --rm -it --privileged --runtime nvidia --shm-size=16g \
             --net=host -e DISPLAY=$DISPLAY \
             -v /tmp/.x11-unix:/tmp/.x11-unix \
             -v $HOME:$HOME -v /mnt/external/data:/mnt/external/data \
             -v /opt/nvidia:/opt/nvidia \
             --name sparse4dv3_nuscenes sparse4dv3_nuscenes:ros bash
```
- **※ 以降の手順は全てコンテナ内で実行**

### deformable_aggregationのインストール(初回のみ)
- リポジトリの`sparse4dv3/projects/mmdet3d_plugin/ops/`へ移動し下記コマンドを実行

```bash
(container)$ sudo python3 setup.py develop --user
```

### checkpointsのダウンロード(初回のみ)
- [Sparse4D#Models](https://github.com/HorizonRobotics/Sparse4D?tab=readme-ov-file#nuscenes-benchmark)にあるcheckpointの内、`Sparse4Dv3 ResNet50`をダウンロード
  - リンク：[sparse4dv3_r50.pth](https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth)、[resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)

- ダウンロードしたものを`sparse4dv3/ckpt`以下に格納

## Dataの準備

### Datasetフォルダー構成

- Datasetフォルダー構成は以下を想定

```
sparse4dv3/data/nuscenes/
├── v1.0-mini/
│   ├── visibility.json
│   ├── sensor.json
│   ├── ...
├── sweeps/
│   ├── CAMERA_FRONT/
│   │   ├── ○○.png
│   │   ├── ...
│   ├── CAMERA_FRONT_RIGHT/
│   │   ├── ○○.png
│   │   ├── ...
│   ├── ...
├── samples/
│   ├── CAMERA_FRONT/
│   │   ├── ○○.png
│   │   ├── ...
│   ├── CAMERA_FRONT_RIGHT/
│   │   ├── ○○.png
│   │   ├── ...
│   ├── ...
├── maps/
├── LICENSE.txt

```
- `sparse4dv3/`で下記のコマンドを実行し、`sparse4dv3/data`以下にnuScenes-miniへのシンボリックリンクを作成
```bash
mkdir data && ln -s path/to/nuscenes-mini data/nuscenes
```

### データローダーファイルの作成(初回のみ)
- `sparse4dv3/`に移動し、nuScenes-miniの場合、下記コマンドを実行
```bash
$ python3 tools/nuscenes_converter.py --root_path path/to/nuscenes-mini --info_prefix nuscenes --version v1.0-mini
```
- 成功するとディレクトリに以下の2ファイルが作成される
  - `nuscenes_infos_val.pkl`
  - `nuscenes_infos_train.pkl`
- 作成された2ファイルを`sparse4dv3/data/nuscenes_annos_pkls/`以下に格納
- オプションの詳細は以下の通り

|オプション|説明|備考|
|---|---|---|
|`--root_path`|データセットディレクトリ|`Nuscenes/`へのパス|
|`--info_prefix`|出力ファイルのprefix|`○○_infos_train/val.pkl`の○○|
|`--version`|データセットのバージョン|`v1.0-mini` OR `v1.0-trainval,v1.0-test`|

### Anchorファイルの作成(初回のみ)
- `sparse4dv3/`に移動し下記コマンドを実行
```bash
$ python3 tools/anchor_generator.py --ann_file data/nuscenes_annos_pkls/nuscenes_infos_train.pkl --output_file_name nuscenes_kmeans900_train.npy
```
- 成功するとディレクトリに`nuscenes_kmeans900_train.npy`が作成される
- オプションの詳細は以下の通り

|オプション|説明|備考|
|---|---|---|
|`--ann_file`|データローダーファイルのパス|上記で作成されたデータローダーファイルへのパス|
|`--num_anchor`|anchor数|デフォルト:900。基本的に変更する必要はない|
|`--detection_range`|物体検出範囲|デフォルト:55[m]。基本的に変更する必要はない|
|`--output_file_name`|出力ファイルの名前||

## 推論の実行
- `sparse4dv3/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py`を編集
  - 306行目: `data_root` にNuscenesデータセットのパスに変更
  - 388行目: `version`にNuscenesデータセットのバージョンに変更
  - 117行目: `use_tensorrt`にTensorRTを使うかによって`True/False`の設定。デフォルトは`False`
  - 119行目: `trt_paths`にそれぞれのTensortRTモデルへのパスを追加。新規作成の場合、それぞれを`None`にする

- `sparse4dv3/`に移動し下記コマンドを実行
```bash
(container)$ bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704 ckpt/sparse4dv3_r50.pth
```
### ROS Nodeの実行

#### 1. ROS2 BAGの作成

- `nuscenes2bag`を使用して作成した`ROS1 BAG`から`ROS2 BAG`を作成
- `sparse4dv3/`に移動し下記コマンドを実行
```bash
bash convert_to_ros2.sh /path/to/ROS1_BAGS_FOLDER/
```

#### 2. Sparse4Dv3 ROS2 Nodeのインストール

- Sparse4Dv3向けのROS2 NodeとMsgファイルをインストール
- `sparse4dv3/`に移動し下記コマンドを実行
```bash
colcon build --symlink-install --packages-select sparse_msgs sparse4dv3
source install/setup.bash
```
**※インストール中に警告やエラーメッセージが出ますが、`sparse_msgs`と`sparse4dv3`の後に`finished`が出ていれば問題ありません**
- 以下のコマンドを実行、エラーなくメッセージファイルの内容が出ていればインストールは成功
```bash
ros2 interface show sparse_msgs/msg/CustomTFMessage
```

#### 3. nuScenesでROS Nodeを実行

● 手法1 (Bash Script)
- `sparse4dv3/`に移動し下記コマンドを実行、複数ROSBAGを実行可能
```bash
bash run_ros_nuscenes.sh /path/to/NuScenes/ROS2_BAG_01 [/path/to/NuScenes/ROS2_BAG_02 ...]
```

● 手法2 (Launch file)
- `sparse4dv3/`に移動し下記コマンドを実行
```bash
ros2 launch launch/sparse_node_launch.py
```
- 同じDocker環境内で別の端末を用意し、上記を実行した端末で`Node Ready`が表示されたら、以下を実行
```bash
ros2 bag play -r 0.3 /path/to/NuScenes/ROS2_BAG/
```
- ROSBAGの再生が終了した場合、必要に応じて次のROSBAGを再生ください

**※注意**
- nuScenesの評価方法
  - `sparse4dv3/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py`を編集
  - 88行目の`ros_eval`で`True`を設定するとROS実行時にnuScenes評価を実施 (デフォルトは`False`)
  - nuScenes-miniで評価する場合、`nuscenes-devkit`は`scene-0103`+`scene-0916`を想定しているため、次のように実行する
```bash
bash run_ros_nuscenes.sh data/NuScenes-v1.0-mini-scene-0103/ data/NuScenes-v1.0-mini-scene-0916/
```
- Sparse4Dv3のNodeに30秒以上topicが送信されないと自動で終了します
