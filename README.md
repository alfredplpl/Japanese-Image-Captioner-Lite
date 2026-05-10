# Japanese Image Captioner Lite

軽量な日本語画像キャプションモデルを作るための、LLaVA風の最小学習基盤です。

構成はシンプルです。

- vision tower: `google/siglip2-so400m-patch16-512`
- language model: `sbintuitions/sarashina2.2-1b-instruct-v0.1`
- image tokens: SigLIP2の1024 patch tokensをそのままLMへ渡す
- trainable layers: vision特徴をLM埋め込みへ写す2層MLP projectorと、LMのLoRA adapter

デフォルト設定はCUDA GPU前提です。CPUで短い動作確認だけする場合は `configs/lite_llava_caption_cpu_test.yaml` を使い、推論では `--device cpu --allow-cpu` を指定します。

## データ形式

学習データは、画像ファイルと日本語キャプションを対応させたJSONLです。
1行が1サンプルで、少なくとも画像パスとキャプション文字列が必要です。

```jsonl
{"image": "images/000001.jpg", "caption": "白い皿に盛られたカレーライス。"}
{"image": "/abs/path/000002.jpg", "caption": "駅のホームに電車が停車している。"}
```

デフォルトでは、画像パスは `image`、キャプションは `caption` というキーから読みます。
キー名を変えたい場合は `configs/lite_llava_caption.yaml` の `data.image_key` と `data.caption_key` を変更します。

```yaml
data:
  train_jsonl: data/train.jsonl
  image_root: data
  image_key: image
  caption_key: caption
```

画像パスの扱い:

- 絶対パス: そのまま読みます。
- 相対パス: `data.image_root` が設定されていれば、そこからの相対パスとして解決します。
- `data.image_root` が未指定の場合: JSONLファイルが置かれているディレクトリからの相対パスとして解決します。

推奨ディレクトリ構成:

```text
data/
  train.jsonl
  val.jsonl
  images/
    000001.jpg
    000002.jpg
```

この場合の `train.jsonl` は次のように書けます。

```jsonl
{"image": "images/000001.jpg", "caption": "白い皿に盛られたカレーライス。"}
{"image": "images/000002.jpg", "caption": "駅のホームに電車が停車している。"}
```

キャプションの書き方:

- 日本語で、画像に実際に写っている内容を書く。
- 1文から2文程度の短い説明から始める。
- 人物、物体、場所、状態、動作、色、数など、画像から判断できる情報を入れる。
- 推測しすぎた情報、画像外の背景説明、ファイル名由来の情報は入れない。
- 同じ定型文ばかりにしない。モデルがその文体だけを覚えやすくなります。

学習時には、各キャプション末尾にtokenizerのEOS tokenを自動で追加します。これにより、推論時にキャプションが不自然に続いたり同じ文を反復したりする問題を抑えます。

よい例:

```jsonl
{"image": "images/cat_001.jpg", "caption": "窓辺に座った茶色の猫が外を見ている。"}
{"image": "images/food_001.jpg", "caption": "木の皿に焼き魚と野菜が盛り付けられている。"}
{"image": "images/street_001.jpg", "caption": "雨の道路を傘を差した人たちが歩いている。"}
```

避けたい例:

```jsonl
{"image": "images/cat_001.jpg", "caption": "かわいい画像。"}
{"image": "images/food_001.jpg", "caption": "これは2024年に東京の有名店で撮影された高級料理です。"}
{"image": "images/street_001.jpg", "caption": ""}
```

画像の条件:

- JPEG、PNGなど、Pillowで読める画像を使います。
- 壊れた画像、極端に小さい画像、真っ黒や真っ白だけの画像は除外します。
- 縦横比は混在していても構いません。image processor がモデル入力サイズに変換します。
- 重複画像やほぼ同じ画像が多すぎると、モデルが偏ります。

データ量の目安:

- 動作確認: 20から100件程度でも可能です。
- 小さなドメイン適応: 数百から数千件。
- 汎用的な日本語キャプション品質を狙う場合: 数万件以上が望ましいです。

学習前の確認:

```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image

jsonl = Path("data/train.jsonl")
image_root = jsonl.parent

for line_no, line in enumerate(jsonl.open(encoding="utf-8"), start=1):
    item = json.loads(line)
    image_path = Path(item["image"])
    if not image_path.is_absolute():
        image_path = image_root / image_path
    caption = item["caption"]
    if not caption.strip():
        raise SystemExit(f"{jsonl}:{line_no}: empty caption")
    with Image.open(image_path) as image:
        image.verify()

print("ok")
PY
```

検証用データを分ける場合は `data.val_jsonl` に指定できます。ただし現在の最小学習ループは `train_jsonl` の学習を主目的にしており、検証ループはまだ実装していません。

## セットアップ

推奨は `uv` です。仮想環境作成、editable install、依存関係の導入をまとめて実行できます。

`uv` が未導入の場合:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

初回セットアップ:

```bash
./scripts/setup_uv.sh
```

このコマンドは `.venv` を作成し、`uv.lock` に基づいて依存関係を同期します。デフォルトではLoRA用の `peft` も含めます。
LoRAを使わない場合:

```bash
./scripts/setup_uv.sh --no-lora
```

Pythonのバージョンを指定する場合:

```bash
./scripts/setup_uv.sh --python 3.11
```

環境を作り直す場合:

```bash
./scripts/setup_uv.sh --recreate
```

手動で実行する場合は次のコマンドでも同じです。

```bash
uv venv --python 3.11
uv sync --extra lora
```

8bit optimizerを試す場合だけ、追加で次を使えます。

```bash
uv sync --extra bnb
```

以降のコマンドは `uv run ...` で実行します。`.venv` を直接有効化したい場合は `source .venv/bin/activate` も使えます。

## 学習

デフォルトの主な学習設定:

```yaml
model:
  language_model: sbintuitions/sarashina2.2-1b-instruct-v0.1
  torch_dtype: bf16
  num_image_tokens: 1024
  freeze_vision: true
  freeze_language_model: true
  use_lora: true

train:
  optimizer: adamw
  learning_rate: 0.0001
  lr_scheduler: constant_with_warmup
  warmup_ratio: 0.03
  mixed_precision: bf16
```

```bash
uv run jicl-train --config configs/lite_llava_caption.yaml
```

または:

```bash
uv run python -m jicl.train --config configs/lite_llava_caption.yaml
```

GPU学習用スクリプト:

```bash
uv run ./scripts/train_gpu.sh configs/lite_llava_caption.yaml
```

複数GPUを使う場合:

```bash
NUM_PROCESSES=2 uv run ./scripts/train_gpu.sh configs/lite_llava_caption.yaml
```

CPUで配線だけ確認する場合:

```bash
uv run jicl-train --config configs/lite_llava_caption_cpu_test.yaml
```

## 推論

学習後の出力先には、`config.yaml`、`tokenizer/`、`projector.pt` が保存されます。`model.use_lora: true` の場合は `lora/` も保存されます。
推論では同じ vision tower / language model を読み込み、保存済みprojectorとLoRA adapterを適用して画像キャプションを生成します。

GPU推論用スクリプト:

```bash
uv run ./scripts/generate_gpu.sh \
  --checkpoint outputs/lite-captioner \
  --image path/to/image.jpg
```

プロンプトや生成長を指定する場合:

```bash
uv run ./scripts/generate_gpu.sh \
  --checkpoint outputs/lite-captioner \
  --image path/to/image.jpg \
  --prompt "画像を日本語で詳しく説明してください。" \
  --max-new-tokens 96
```

dtypeを明示する場合:

```bash
DTYPE=bf16 uv run ./scripts/generate_gpu.sh \
  --checkpoint outputs/lite-captioner \
  --image path/to/image.jpg
```

CPUで動作確認だけする場合は、明示的にCPU許可が必要です。

```bash
uv run ./scripts/generate_gpu.sh \
  --checkpoint outputs/lite-captioner \
  --image path/to/image.jpg \
  --device cpu \
  --allow-cpu
```

entrypointを直接使う場合:

```bash
uv run jicl-generate \
  --checkpoint outputs/lite-captioner \
  --image path/to/image.jpg \
  --prompt "画像を日本語で簡潔に説明してください。" \
  --device cuda \
  --dtype auto
```

`--prompt` を省略すると、checkpoint内の `config.yaml` に保存された `data.prompt` を使います。
デフォルトではCUDA推論です。CUDAがない環境で `--device cuda` のまま実行するとエラーで停止します。

## メモリをさらに削る設定

- `train.batch_size` を下げ、`gradient_accumulation_steps` を上げる
- `model.num_image_tokens` を下げる。ただし現在の実装では先頭から指定数の画像トークンを使います
- `train.mixed_precision` をGPUに合わせて `bf16` または `fp16` にする
- `model.use_lora: false` のままprojectorだけ学習する

## LoRA

デフォルト設定ではLoRAを使います。

```yaml
model:
  freeze_language_model: true
  use_lora: true
```

この場合もベースLM全体は保存せず、projectorとLoRA adapterだけ保存します。推論時はcheckpoint内の `config.yaml` を見て、`lora/` を自動で読み込みます。
