# CowFaceDetection

牛脸检测 + 个体识别：YOLOv8 定位牛脸，EfficientNet-B0 判断是哪一头牛，Flask 提供网页界面。

![检测样例](docs/preview.png)

## 结果

同一个权重，在两个划分上的表现：

| 评测 | 数据 | mAP50 | mAP50-95 | P | R |
| --- | --- | --- | --- | --- | --- |
| 项目自带 test | 107 张，与 train 同源 | 0.941 | 0.520 | 0.936 | 0.934 |
| 跨数据集零样本 | OpenCows2020 真实帧 3,707 张 / 6,914 个框 | 0.055 | 0.013 | 0.166 | 0.103 |

个体识别在项目自带的 10 头牛、110 张 test 上 top-1 100%，CPU 单张检测 26 ms。

两个数字差 17 倍，原因不只是域偏移（不同牧场、相机、品种），更主要的是自带划分本身有泄漏：

- 检测的 test 只有 40 段视频帧，其中 39 段同时出现在 train —— Roboflow 按增强图随机划分，不是按帧划分，同一条视频的两帧会分处 train 与 test。
- 分类的 test 与 train 是同一批牛、同一场景的照片；dHash 抽查显示 14% 的 test 图与 train 某张的 Hamming 距离 ≤ 4。
- 有 2 头牛（`Cow (4)`、`Cow (8)`）没进训练集，模型仍然把它们判成已知个体：平均最大概率 0.90 与 0.57，没有 unknown 拒识。

复现：

```bash
python scripts/evaluate.py                        # 自带划分：检测 + 分类 + open-set
python scripts/evaluate_opencows.py --zip OpenCows2020.zip --workdir opencows_eval   # 跨数据集
```

### 换成无泄漏划分重训

上表的 0.941 之所以不能当能力看，是因为划分本身有泄漏。要一个能对外说的数字，就得换一套按身份分组的划分：OpenCows2020 自带了官方的 10 折检测划分（`10-fold-CV.json`，按牛只身份分组）和 9 个识别划分（`75-25.json` 等，每折给出哪些牛当已知类、哪些只在测试出现）。

| 任务 | 划分 | 训练 / 测试 | 结果 |
| --- | --- | --- | --- |
| 检测 | `10-fold-CV` fold 0 | 6,338 / 705 张 | mAP50 **0.974**、mAP50-95 0.761、P 0.945、R 0.949 |
| 识别 · 闭集 | `75-25` fold 0，35 头已知 | 3,124 / 367 张 | top-1 **0.967** |
| 识别 · 开集 | 同上，11 头只在测试出现 | 129 张 | AUROC **0.912**，误拒 5% 时拒识率 0.574 |
| 识别 · 闭集 | `50-50` fold 0，23 头已知 | 2,122 / 249 张 | top-1 0.956 |
| 识别 · 开集 | 同上，23 头只在测试出现 | 247 张 | AUROC 0.896，拒识率 0.583 |

反向也测了一次：把 OpenCows2020 重训出来的检测器拿回本仓库的 test（107 张）跑，mAP50 0.355。也就是说域偏移是双向的 —— 自带权重去 OpenCows 只有 0.055，OpenCows 权重回来也只有 0.355。这些数字说明模型记住的是"哪个牧场"，不是"牛脸长什么样"。

检测 30 轮在 RTX 4060 Ti 上约 16 分钟，识别 20 轮约 6 分钟：

```bash
python scripts/train_detect_opencows.py --zip OpenCows2020.zip --workdir opencows_work --fold 0 --epochs 30 --device 0
python scripts/train_ident_opencows.py  --zip OpenCows2020.zip --workdir opencows_work --split 75-25 --fold 0 --epochs 20 --device cuda
```

识别脚本会把 checkpoint 存到 `outputs/`，里面的 `classes` 列表就是这轮训练的类别顺序，换数据集时不必再靠 `sorted(os.listdir())` 猜。

## 数据集

`ml_models.zip`（457 MB，见 [releases](https://github.com/summeriver13/cowface-detection/releases/tag/ml-models)）：

| 目录 | 内容 |
| --- | --- |
| `yolo-models/` | 微调后的 YOLOv8n 检测器 6 MB、EfficientNet-B0 分类器 18 MB |
| `conference-paper-preparations/` | 检测数据集，YOLO 格式：train 2238 / valid 213 / test 107 张 |
| `dataset_for_classification/` | 分类数据集，按牛分目录：train 420 / test 110 张，10 头牛 |
| `pretrained_models/` | YOLOv8n、YOLOv8s 预训练权重 |

真实样本量由视频帧数决定：train 56 帧、valid 48 帧、test 40 帧，其余都是增强副本。

### 外部数据集

自带数据只有 12 头牛、56 段训练帧，想验证泛化或者换个场景训练，用这两个公开数据集：

| 数据集 | 规模 | 内容 | 许可 |
| --- | --- | --- | --- |
| [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17) | 2.1 GiB | 7,043 张检测帧（darknet + VOC 标签，003707 之后为合成图）、46 头牛的识别图 | Non-Commercial Government Licence |
| [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7) | 17.6 GiB | 检测（COCO json）、识别（182 头牛、21k 图）、官方权重 | Non-Commercial Government Licence |

两个都来自布里斯托大学同一个课题组，检测标签的类别都是"牛脸"，与这里的单类检测直接对齐；识别部分是分牛目录，可以直接换成 `dataset_for_classification` 的格式。非商业许可意味着可以用于研究和演示，不能商用。

```bash
curl -L -o OpenCows2020.zip https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip
curl -L -o Cows2021.zip    https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip
```

## 快速开始

### 1. 准备模型

在 [releases](https://github.com/summeriver13/cowface-detection/releases/tag/ml-models) 下载 `ml_models.zip`，解压后把 `ml_models/` 放到项目根目录。

### 2. 装依赖

```bash
uv venv && uv pip install -r requirements.txt
# 或者
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

### 3. 启动

```bash
python run.py                  # http://127.0.0.1:5000
FLASK_DEBUG=1 python run.py    # 带调试与自动重载
```

默认 SQLite（`instance/cow_face.db`），首次启动自动建表，不需要额外装数据库。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SECRET_KEY` | `dev-only-insecure-key` | session 签名密钥，部署前必须替换 |
| `DATABASE_URL` | `sqlite:///instance/cow_face.db` | 换 MySQL：`mysql+pymysql://user:pass@host/cow_face_db` |
| `ML_MODELS_DIR` | `./ml_models` | 权重与数据集根目录 |
| `FLASK_DEBUG` | `0` | 设为 `1` 打开调试 |

## 功能

- 注册 / 登录 / 登出
- 上传图片 → 检测牛脸 → 识别个体 → 返回画框图与每条检测的置信度，结果入库
- 牛只管理：增删改查、每只牛最近 5 条识别记录、完整历史
- 在网页里发起 YOLOv8 微调训练（模型规格、epochs、batch size、学习率），后台线程执行，状态页轮询数据库

## HTTP 接口

页面与数据接口分两组蓝图：页面在根路径（`web.*`），JSON 接口统一在 `/api` 前缀下（`api.*`）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/login`、`/register`、`/logout` | 认证页面 |
| GET | `/dashboard` | 首页，含牛只/记录/训练计数 |
| GET | `/cattle`、`/cattle/<id>` | 牛只列表、详情 |
| GET | `/detect`、`/train`、`/train/status` | 检测、训练、训练状态页 |
| POST | `/api/cattle` | 新建牛只（form 或 JSON） |
| PATCH | `/api/cattle/<id>` | 改名 |
| DELETE | `/api/cattle/<id>` | 删除牛只与其识别记录 |
| POST | `/api/detect` | 上传图片，返回标注图 base64 与检测列表 |
| GET | `/api/train` | 训练任务列表（状态页每 3 秒轮询） |
| POST | `/api/train` | 启动训练，返回 202 |
| DELETE | `/api/train/<id>` | 删除训练记录 |

约定：成功 `{"ok": true, "data": ...}`，失败 `{"ok": false, "error": {"code", "message"}}` 并带上对应 HTTP 状态码；所有写请求要带 `X-CSRF-Token`（页面从 `<meta name="csrf-token">` 取）或表单里的 `csrf_token` 字段。

```bash
python scripts/smoke_test.py    # 22 项端到端检查：页面、接口、CSRF、权限
```

## 目录结构

```
run.py                        入口
scripts/evaluate.py           自带划分上的检测 / 识别 / open-set 评测
scripts/evaluate_opencows.py  跨数据集评测（OpenCows2020）
scripts/smoke_test.py         端到端冒烟测试
app/__init__.py               create_app：配置、蓝图注册、建表、错误处理
app/config.py                 路径、环境变量、模型规格
app/security.py               CSRF token 生成与校验
app/routes/                   页面蓝图 web.*：auth / dashboard / cattle / detect / train
app/api/                      JSON 蓝图 api.*（前缀 /api）：cattle / detect / train
app/models/                   ORM：User / Cattle / RecognitionRecord / ModelTraining
app/controllers/              业务逻辑：识别器、数据集 yaml、训练
app/templates/                Jinja2 页面，全部继承 base.html
app/static/                   css/app.css、js/{api,cattle,detect,train,training_status}.js
docs/preview.png              README 里的检测样例
```

## 已知限制

- 分类器输出层的类别顺序由训练时的数据集目录决定，首次加载会快照到 `ml_models/yolo-models/*.classes.json`；换数据集重训后要删掉这个文件。
- 只有 10 头牛的闭集识别，没有 unknown 拒识：陌生牛会被硬判成最像的已知个体。
- 训练跑在 Web 进程的后台线程里，只适合单机演示：服务重启会丢任务，没有并发保护，也不能中途取消。
- 推理在请求线程内同步执行，没有队列与超时控制。
