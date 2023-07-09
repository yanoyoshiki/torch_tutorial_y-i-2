# torch_tutorial_y_-2
The repository for M1 tutorial. 

The administers are YANO and NAKAMURA. 

このリポジトリは「課題2: ハイパーパラメータチューニングに慣れる」に関するリポジトリです.

# 課題
以下の1.-8.を実施する．

1. CIFAR-10とMNISTを落としてくる
2. Tensorboardをつかうので，知らない人は調べる
3. CNNでネットワークを作る(各種パラメータは調整するので変更が容易な形で実装)
4. MNISTを使って，ネットワークの深さ，学習率，バッチサイズなどのハイパーパラメータを変化させながら
モデルの性能を評価
5. CIFAR-10でも同様の作業を行う
6. 各ハイパーパラメータが結果にどのように影響するか比較，考察
7. どのように最適なパラメータを探すかについて調査
8. (できる人だけ) Optunaなどのハイパーパラメータチューニングライブラリによる自動化


# Quick Start
## SSH接続(SSH接続でリモートPCへ接続する場合のみlocal terminalにて実行が必要)
```
 ssh -L 63322:localhost:63322 -L 6006:localhost:6006 -L 6007:localhost:6007  <username>@<remotePC IP>
```
接続完了後， 環境構築へ
## nativeの場合
環境構築へ
## 環境設定
### 初回
任意のディレクトリにて， githubからリポジトリのクローンを作成する．
```
git clone https://github.com/Issa-N/torch_tutorial_y-i-3.git
```
次に，Dockerで仮想環境を作る．
```
cd torch_tutorial_y-i-3/docker

bash build.sh
bash run.sh
```
### 2回目以降
以下のコマンドで，dockerのコンテナへの接続を行う．
```
docker exec -it torch_tutorial_y-i　bash
```
一時的に抜ける時は 「controll + P + Q」

## Jupiter notebookを開く時
```
jupyter-notebook --ip 0.0.0.0 --port 63322 --allow-root
```



