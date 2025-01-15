# nuscenes2bag

> _[nuScenes](https://www.nuscenes.org/)-mini data を ROS1 [bag](http://wiki.ros.org/rosbag) に変換するツール_

## Dataの準備
- 以下のデータを [nuScenes](https://nuscenes.org/nuscenes) からダウンロード、`unzip`する
  - nuScenes-mini (v1.0-mini.zip)
  - CAN bus expansion pack (can_bus.zip)
  - Map expansion pack v1.3 (nuScenes-map-expansion-v1.3.zip)

### Datasetフォルダー構成

- Datasetフォルダー構成は以下を想定

```
nuscenes2bag/
├── docker/
├── data/
│   ├── v1.0-mini/
│   │   ├── visibility.json
│   │   ├── sensor.json
│   │   ├── ...
│   ├── sweeps/
│   │   ├── CAMERA_FRONT/
│   │   │   ├── ○○.png
│   │   │   ├── ...
│   │   ├── CAMERA_FRONT_RIGHT/
│   │   │   ├── ○○.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── samples/
│   │   ├── CAMERA_FRONT/
│   │   │   ├── ○○.png
│   │   │   ├── ...
│   │   ├── CAMERA_FRONT_RIGHT/
│   │   │   ├── ○○.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── maps/
│   │   ├── basemap/
│   │   │   ├── ○○.png
│   │   │   ├── ...
│   │   ├── expansion/
│   │   │   ├── ○○.json
│   │   │   ├── ...
│   │   ├── prediction/
│   │   │   ├── ○○.json
│   │   ├── ○○.png
│   │   ├── ...
│   ├── can_bus/
│   │   ├── scene-○○.json
│   │   ├── ...
│   ├── LICENSE.txt
├── nuscenes2bag.py
├── README.jp.md
```

- 上記の構成にするため、まずは`expansion pack`を`nuScenes-mini`以下に移動
- 次に`nuscenes2bag/`で下記コマンドを実行し、`nuscenes2bag/data`以下に`nuScenes-mini`へのシンボリックリンクを作成
```bash
mkdir data && ln -s path/to/nuscenes-mini data/nuscenes
```

## 環境構築
- 下記環境で動作を確認
```
OS             : Ubuntu 20.04 LTS
CUDA Version   : 11.4
Docker version : 24.0.7, build 24.0.7-ubuntu2~20.04.1
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
- `nuscenes2bag/`に移動し下記コマンドを実行
```bash
$ cd docker
$ docker build -t ros_noetic:base .
```
- `./docker_setup.sh` を起動して、ユーザ入りのDockerイメージを作成
  - `user` の部分は自分のユーザ名に置き換える
```bash
$ ./docker_setup.sh
input base image name: ros_noetic:base
input user add image name: ros_noetic:user
input user password:
input user password again:
  base image     :  ros_noetic:base
  user add image :  ros_noetic:user
  user           :  user
  uid            :  1001
  gid            :  1001
Is it OK?[Yes/no]: Yes
```

### Dockerコンテナの起動
- `$HOME`以外の場所にデータを配置してある場合はマウント(`-v`)が必要
- 例えば、`/mnt/external/data`ディレクトリ内にNuScenesデータセットを格納している場合、`-v /mnt/external/data:/mnt/external/data`オプションを追加
```bash
$ docker run --rm -it --privileged --runtime nvidia --shm-size=16g \
             --net=host -e DISPLAY=$DISPLAY \
             -v /tmp/.x11-unix:/tmp/.x11-unix \
             -v $HOME:$HOME -v /mnt/external/data:/mnt/external/data \
             --name ros_noetic ros_noetic:user bash
```

## 実行
- `nuscenes2bag/` に移動し、以下のコマンドを実行
```
python3 nuscenes2bag.py
```

- `NuScenes-v1.0-mini-scene-0103.bag`のようなbagファイルが生成されていれば成功

## License

nuscenes2bag is licensed under [MIT License](https://opensource.org/licenses/MIT).
