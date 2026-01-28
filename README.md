# モデル学習・評価 実行手順書

本プロジェクトは、モデルの学習から結果の集計までを行うためのスクリプト群です。
計算負荷が高いため、GPUサーバー上での実行を想定しています。

## 1. 準備：サーバーへのファイル転送

ローカルPCにあるデータセット(`archive`)とソースコード(`graduation`)をサーバーへ転送します。
※ `.venv` (仮想環境) は転送せず、サーバー上で新規作成することを推奨します。

```bash
# ローカルPCのターミナルで実行
scp -r archive graduation {your_username}@{your_server_ip}:~/
```

## 2. 環境構築 (サーバー上)

サーバーにログインし、Python環境をセットアップします。

```bash
ssh {your_username}@{your_server_ip}

# 仮想環境の作成と有効化 (例: .venv)
python3 -m venv .venv
source .venv/bin/activate

# ライブラリのインストール
pip install -r graduation/undergraduate/requirements.txt
```

## 3. モデルの学習 (サーバー上)

学習スクリプトをバックグラウンドで実行します。
`nohup`を使用することで、ログアウト後も処理を継続させます。

```bash
# GPU指定が必要な場合の例 (GPU 0番を使用)
# export CUDA_VISIBLE_DEVICES=0

# 実行 (timeで時間を計測、ログを保存)
time nohup python graduation/undergraduate/{ファイル名} > output.log 2> error.log &
```

- **進捗確認**: `tail -f output.log` (終了は `Ctrl + C`)
- **エラー確認**: `cat error.log`
- **終了処理**: `deactivate` で仮想環境を無効化し、`exit` でログアウト。

## 4. 結果の取得 (ローカルPC)

学習完了後、サーバー上の `mcoutput` フォルダをローカルにダウンロードします。

```bash
# ローカルPCのターミナルで実行
scp -r {your_username}@{your_server_ip}:~/mcoutput .
```

## 5. 結果の集計 (ローカルPC)

ダウンロードした結果を集計します。

```bash
# 仮想環境を有効化
source .venv/bin/activate

# 集計スクリプトの実行
python {集計スクリプトファイル名}
```
