# UCR_TS_Archive_2015 カテゴリ別データセット特徴

`data/UCR_TS_Archive_2015/` に含まれる85データセットを、分野（ドメイン）ごとに7カテゴリ
（Device / ECG / Image / Motion / Sensor / Simulated / Spectro）へ分類し、各カテゴリの特徴を
まとめたもの。この分類は `papers/ieee_smc2026/category.csv` 等、本リポジトリの分析スクリプトで
実際に使われている区分と一致する。

統計値（系列長・クラス数・train/test件数）は `data/UCR_TS_Archive_2015/UCR_TS_Archive_2015/` 配下の
`*_TRAIN` / `*_TEST` を実際に読み込んで集計したもの（2026-11-04時点）。

## カテゴリ概要

| カテゴリ | 件数 | ドメイン概要 | 系列長 (min–max / avg) | クラス数 (min–max / avg) | train件数 (min–max / avg) | test件数 (min–max / avg) |
|---|---|---|---|---|---|---|
| Device | 6 | 家電・電力センサーの電力消費波形（UK-DALE/REDD系）。Computers, ElectricDevices, LargeKitchenAppliances, RefrigerationDevices, ScreenType, SmallKitchenAppliances。 | 96–720 / 616.0 | 2–7 / 3.5 | 250–8926 / 1779.3 | 250–7711 / 1576.8 |
| ECG | 6 | 心電図波形。心拍1拍分の分類（ECG200/ECGFiveDays/TwoLeadECG）から、胎児心電図の多クラス分類（NonInvasiveFatalECG_Thorax1/2、42クラス）まで幅がある。 | 82–750 / 325.7 | 2–42 / 15.8 | 23–1800 / 707.7 | 100–4500 / 1755.0 |
| Image | 29 | 画像の輪郭（アウトライン）を1次元系列化したデータ。葉・魚・矢じり・骨のX線輪郭（Phalanx系）・顔・単語の輪郭など、形状分類タスクが中心。85データセット中最多の29件を占める。 | 80–2709 / 324.9 | 2–60 / 10.3 | 16–1800 / 301.9 | 20–3000 / 615.4 |
| Motion | 14 | モーションキャプチャ・加速度センサーによる動作・ジェスチャ系列。クリケット審判の身振り（Cricket_X/Y/Z）、ガンポイント動作、uWaveジェスチャ、線虫(Worms)の動きなど。 | 150–1882 / 595.3 | 2–12 / 6.6 | 36–896 / 377.8 | 130–3582 / 1230.4 |
| Sensor | 18 | 上記以外の物理・環境センサー計測値全般（温度・光量・音響・振動・天体輝度・エンジン音など）。ドメインが雑多で、データセット間のばらつきが大きい最も異種混合なカテゴリ。 | 24–1639 / 442.7 | 2–39 / 5.6 | 20–1320 / 318.8 | 60–8236 / 1960.5 |
| Simulated | 5 | 特定の形状・パターン検出能力を検証するための人工合成データ（CBF, MALLAT, ShapeletSim, Two_Patterns, synthetic_control）。ノイズやクラス境界を制御できるベンチマーク用途。 | 60–1024 / 368.0 | 2–8 / 4.6 | 20–1000 / 281.0 | 180–4000 / 1545.0 |
| Spectro | 7 | 食品の分光（スペクトル）データ（近赤外分光等）による品種・産地判別（Beef, Coffee, Ham, Meat, OliveOil, Strawberry, Wine）。サンプル数が小さい（train平均約98件）のが特徴。 | 234–570 / 382.0 | 2–5 / 2.9 | 28–370 / 97.7 | 28–613 / 131.4 |

全85データセットの合計: Device 6 + ECG 6 + Image 29 + Motion 14 + Sensor 18 + Simulated 5 + Spectro 7 = 85。

## カテゴリ別の傾向

- **Image** が最大カテゴリ（29件、全体の約34%）であり、系列長・クラス数のレンジが広い
  （系列長80〜2709、クラス数2〜60）。Phalanx系（Distal/Middle/Proximal × OutlineAgeGroup/OutlineCorrect/TW）
  のような「同一ドメインの派生データセット」が多く含まれる。
- **Spectro** は7データセットすべてで train件数が30〜370と非常に少なく、サンプル数の少なさが
  学習・検証（特に交差検定の分割数）に影響しやすい。
- **Device** は「1系列=1日分の電力消費（96点 or 720点）」という時間刻みの系列長がほぼ固定で、
  カテゴリ内の均質性が高い。
- **ECG** は NonInvasiveFatalECG_Thorax1/2 のみクラス数42と極端に多く、他4件（2〜5クラス）と
  性質が大きく異なる。カテゴリ平均（クラス数avg 15.8）はこの2件に強く引かれている。
- **Sensor** は最も異種混合（温度・光・音・振動・天体輝度・エンジン音など出自が揃っていない）で、
  系列長・クラス数・サンプル数のいずれも分散が大きい。
- **Simulated** は人工生成データのため、実データに比べてクラス境界やノイズ特性が既知・制御可能。
  モデルの基礎的な形状認識能力の検証に向く。

## データセット別詳細

各カテゴリ内はデータセット名の昇順。列は「クラス数 / 系列長 / train件数 / test件数」。

### Device (6)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| Computers | 2 | 720 | 250 | 250 |
| ElectricDevices | 7 | 96 | 8926 | 7711 |
| LargeKitchenAppliances | 3 | 720 | 375 | 375 |
| RefrigerationDevices | 3 | 720 | 375 | 375 |
| ScreenType | 3 | 720 | 375 | 375 |
| SmallKitchenAppliances | 3 | 720 | 375 | 375 |

### ECG (6)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| ECG200 | 2 | 96 | 100 | 100 |
| ECG5000 | 5 | 140 | 500 | 4500 |
| ECGFiveDays | 2 | 136 | 23 | 861 |
| NonInvasiveFatalECG_Thorax1 | 42 | 750 | 1800 | 1965 |
| NonInvasiveFatalECG_Thorax2 | 42 | 750 | 1800 | 1965 |
| TwoLeadECG | 2 | 82 | 23 | 1139 |

### Image (29)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| 50words | 50 | 270 | 450 | 455 |
| Adiac | 37 | 176 | 390 | 391 |
| ArrowHead | 3 | 251 | 36 | 175 |
| BeetleFly | 2 | 512 | 20 | 20 |
| BirdChicken | 2 | 512 | 20 | 20 |
| DiatomSizeReduction | 4 | 345 | 16 | 306 |
| DistalPhalanxOutlineAgeGroup | 3 | 80 | 139 | 400 |
| DistalPhalanxOutlineCorrect | 2 | 80 | 276 | 600 |
| DistalPhalanxTW | 6 | 80 | 139 | 400 |
| FISH | 7 | 463 | 175 | 175 |
| FaceAll | 14 | 131 | 560 | 1690 |
| FaceFour | 4 | 350 | 24 | 88 |
| FacesUCR | 14 | 131 | 200 | 2050 |
| HandOutlines | 2 | 2709 | 370 | 1000 |
| Herring | 2 | 512 | 64 | 64 |
| MedicalImages | 10 | 99 | 381 | 760 |
| MiddlePhalanxOutlineAgeGroup | 3 | 80 | 154 | 400 |
| MiddlePhalanxOutlineCorrect | 2 | 80 | 291 | 600 |
| MiddlePhalanxTW | 6 | 80 | 154 | 399 |
| OSULeaf | 6 | 427 | 200 | 242 |
| PhalangesOutlinesCorrect | 2 | 80 | 1800 | 858 |
| ProximalPhalanxOutlineAgeGroup | 3 | 80 | 400 | 205 |
| ProximalPhalanxOutlineCorrect | 2 | 80 | 600 | 291 |
| ProximalPhalanxTW | 6 | 80 | 205 | 400 |
| ShapesAll | 60 | 512 | 600 | 600 |
| SwedishLeaf | 15 | 128 | 500 | 625 |
| Symbols | 6 | 398 | 25 | 995 |
| WordsSynonyms | 25 | 270 | 267 | 638 |
| yoga | 2 | 426 | 300 | 3000 |

### Motion (14)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| Cricket_X | 12 | 300 | 390 | 390 |
| Cricket_Y | 12 | 300 | 390 | 390 |
| Cricket_Z | 12 | 300 | 390 | 390 |
| Gun_Point | 2 | 150 | 50 | 150 |
| Haptics | 5 | 1092 | 155 | 308 |
| InlineSkate | 7 | 1882 | 100 | 550 |
| ToeSegmentation1 | 2 | 277 | 40 | 228 |
| ToeSegmentation2 | 2 | 343 | 36 | 130 |
| UWaveGestureLibraryAll | 8 | 945 | 896 | 3582 |
| Worms | 5 | 900 | 77 | 181 |
| WormsTwoClass | 2 | 900 | 77 | 181 |
| uWaveGestureLibrary_X | 8 | 315 | 896 | 3582 |
| uWaveGestureLibrary_Y | 8 | 315 | 896 | 3582 |
| uWaveGestureLibrary_Z | 8 | 315 | 896 | 3582 |

### Sensor (18)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| Car | 4 | 577 | 60 | 60 |
| ChlorineConcentration | 3 | 166 | 467 | 3840 |
| CinC_ECG_torso | 4 | 1639 | 40 | 1380 |
| Earthquakes | 2 | 512 | 139 | 322 |
| FordA | 2 | 500 | 1320 | 3601 |
| FordB | 2 | 500 | 810 | 3636 |
| InsectWingbeatSound | 11 | 256 | 220 | 1980 |
| ItalyPowerDemand | 2 | 24 | 67 | 1029 |
| Lighting2 | 2 | 637 | 60 | 61 |
| Lighting7 | 7 | 319 | 70 | 73 |
| MoteStrain | 2 | 84 | 20 | 1252 |
| Phoneme | 39 | 1024 | 214 | 1896 |
| Plane | 7 | 144 | 105 | 105 |
| SonyAIBORobotSurface | 2 | 70 | 20 | 601 |
| SonyAIBORobotSurfaceII | 2 | 65 | 27 | 953 |
| StarLightCurves | 3 | 1024 | 1000 | 8236 |
| Trace | 4 | 275 | 100 | 100 |
| wafer | 2 | 152 | 1000 | 6164 |

### Simulated (5)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| CBF | 3 | 128 | 30 | 900 |
| MALLAT | 8 | 1024 | 55 | 2345 |
| ShapeletSim | 2 | 500 | 20 | 180 |
| Two_Patterns | 4 | 128 | 1000 | 4000 |
| synthetic_control | 6 | 60 | 300 | 300 |

### Spectro (7)

| dataset | classes | length | train | test |
|---|---|---|---|---|
| Beef | 5 | 470 | 30 | 30 |
| Coffee | 2 | 286 | 28 | 28 |
| Ham | 2 | 431 | 109 | 105 |
| Meat | 3 | 448 | 60 | 60 |
| OliveOil | 4 | 570 | 30 | 30 |
| Strawberry | 2 | 235 | 370 | 613 |
| Wine | 2 | 234 | 57 | 54 |

## 備考

- カテゴリ区分自体は `papers/ieee_smc2026/category.csv`（および同内容の
  `experiments/ieee_smc2026/analysis/overall_macd/category.csv` 等）に合わせたもので、
  本ファイルは分類の由来・ドメイン説明・構造的統計量を独立に整理した参照資料。
- 系列長・クラス数・train/test件数は `data/UCR_TS_Archive_2015/UCR_TS_Archive_2015/<dataset>/<dataset>_{TRAIN,TEST}`
  を実際に読み込んで算出したもので、`experiments/*/analysis/` 側のモデル精度等とは独立。
- データセット本体（`data/UCR_TS_Archive_2015/`）はgitignore対象だが、このファイルは参照資料として
  `.gitignore` に例外登録してgit管理する。
