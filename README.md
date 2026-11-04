# Lightweight-TSC

時系列分類（TSC）向け軽量モデルの研究用リポジトリです。
モデルの学習から結果の集計までを行うためのスクリプト群を、論文・提出先ごとにまとめています。

## ドキュメント一覧

| ファイル | 役割 |
|---|---|
| `CLAUDE.md` | Claude Codeに渡すプロジェクトの背景情報・作業方針。新しいチャットで作業を始める際はまずこれが読まれる。 |
| `TODO.md` | 現在進行中の研究タスクと優先順位。最も頻繁に更新する現状把握の一次情報源。 |
| `RESEARCH_LOG.md` | ゼミ発表資料から再構成した研究の経緯（モデルの変遷・意思決定の背景・実験結果）。個人メモのためgitignore対象。 |
| `HISTORY.md` | リポジトリのディレクトリ構成・論文投稿先の変遷。 |

## ディレクトリ構成

```
.
├── data/                   # データセット（.gitignore対象）
│   └── UCR_TS_Archive_2015/
├── experiments/            # 論文・提出先ごとの実験一式
│   ├── ieee_smc2026/
│   │   ├── runs/           # 学習スクリプト＋実行結果（model.h5, training.log等）
│   │   └── analysis/       # 結果の集計・統計スクリプトとcsv/png
│   ├── iiai_aai2026/
│   │   ├── runs/
│   │   └── analysis/
│   ├── undergraduate/      # 学部の実験
│   │   ├── runs/
│   │   │   ├── origins/    # 最初期のベースライン実装
│   │   │   ├── main/       # 検証本体（旧 valid/）
│   │   │   └── further/    # 追加検証
│   │   └── analysis/
│   └── masters/            # 修士の実験（未着手）
├── wip/                     # まだ論文単位に整理していない最新の作業中コード
│   ├── runs/
│   └── analysis/
├── archive/                 # 過去に破棄・置き換えたコード・ログ
├── papers/                  # 論文執筆物（tex/bib/doc/fig等）
│   └── ieee_smc2026/
├── CLAUDE.md                # Claude Codeに渡すプロジェクトの背景情報
└── requirements.txt
```

`experiments/*/runs`、`wip/runs` 配下に保存されている `model.h5` や `training.log` 等は過去の実行結果なので削除しないでください。
集計用の csv や Excel（`analysis/` 配下）は整理・統合して構いません。

## スクリプトの実行に関する注意

学習スクリプトは `UCR_TS_Archive_2015`（データセット）や出力フォルダ（`mcloutput5` など）を
**スクリプトの場所からの相対パスではなく、実行時のカレントディレクトリからの相対パス**で参照します。
そのため実行する際は、対象スクリプトが置かれているフォルダに `cd` してから実行してください。

```bash
cd experiments/ieee_smc2026/runs/mc_mhlite/mc_mhlite_train
# データセットをこのフォルダから見える場所に用意する（コピー or シンボリックリンク）
ln -s ../../../../../data/UCR_TS_Archive_2015 UCR_TS_Archive_2015
python mc_mhlite_train.py
```

## 1. 準備：サーバーへのファイル転送

```bash
# ローカルPCのターミナルで実行
scp -r data experiments {your_username}@{your_server_ip}:~/
```

※ `.venv`（仮想環境）は転送せず、サーバー上で新規作成することを推奨します。

## 2. 環境構築（サーバー上）

```bash
ssh {your_username}@{your_server_ip}

# 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate

# ライブラリのインストール
pip install -r requirements.txt
```

## 3. モデルの学習（サーバー上）

```bash
cd {対象スクリプトのフォルダ}
# GPU指定が必要な場合の例（GPU 0番を使用）
# export CUDA_VISIBLE_DEVICES=0

time nohup python {ファイル名} > output.log 2> error.log &
```

- **進捗確認**: `tail -f output.log`（終了は `Ctrl + C`）
- **エラー確認**: `cat error.log`
- **終了処理**: `deactivate` で仮想環境を無効化し、`exit` でログアウト。

## 4. 結果の取得（ローカルPC）

```bash
# ローカルPCのターミナルで実行
scp -r {your_username}@{your_server_ip}:~/{出力フォルダ} .
```

## 5. 結果の集計（ローカルPC）

```bash
source .venv/bin/activate
python {analysis配下の集計スクリプトファイル名}
```
