# herdface

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white">
  <img alt="YOLOv8" src="https://img.shields.io/badge/detection-YOLOv8-FF6F00">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="CI" src="https://github.com/summeriver13/herdface/actions/workflows/ci.yml/badge.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

<p align="center">牛脸检测 + 个体识别：YOLOv8 定位牛脸，EfficientNet-B0 判断是哪一头牛，Flask 提供网页界面。<br>
检测 mAP50 <b>0.974</b>（无泄漏划分）· 已知牛识别 top-1 <b>0.967</b> · 陌生牛拒识 AUROC <b>0.912</b></p>

<p align="center"><img src="docs/preview.webp" width="440" alt="检测样例"></p>

## 目录

- [结果速览](#结果速览)
- [实验记录](#实验记录)
- [快速开始](#快速开始)
- [功能与接口](#功能与接口)
- [项目结构](#项目结构)
- [环境变量](#环境变量)
- [已知限制](#已知限制)
- [数据来源](#数据来源)
- [许可](#许可)

## 结果速览

| 任务 | 划分 | 数据 | 指标 |
| --- | --- | --- | --- |
| 检测牛脸 | OpenCows2020 官方 10 折 fold 0 | 6,338 训练 / 705 测试 | mAP50 **0.974**、mAP50-95 0.761、P 0.945、R 0.949 |
| 检测牛脸 | 自带 `test`（与 train 同源） | 107 张 | mAP50 0.941、mAP50-95 0.520 |
| 检测牛脸 | 零样本跨牧场 | OpenCows2020 真实帧 3,707 张 | mAP50 0.055 |
| 识别已知牛 | OpenCows `75-25` fold 0 | 35 头 / 367 张 | top-1 **0.967** |
| 识别陌生牛 | 同上（11 头只在测试出现） | 129 张 | AUROC **0.912**，误拒 10% 时拒识 80.6% |
| 识别已知牛 | 自带 10 头牛 | 110 张 | top-1 1.000（同牛同场景，含泄漏） |

<p align="center"><img src="docs/domain_gap.png" width="620" alt="域偏移"></p>

同一个模型换个牧场就崩：本项目学到的很大一部分是那个牧场的光照、相机和栏杆，不只是牛脸。自带划分的 test 40 个视频帧里有 39 个同时出现在 train，所以 0.941 只能当上限看。

## 实验记录

[`docs/experiments.md`](docs/experiments.md) 是完整的研究记录：数据诊断（泄漏证据、真实样本量）、跨数据集零样本、无泄漏重训、开集拒识曲线、近重复检查、双向域偏移，以及每条结论对应的复现命令。

<p align="center"><img src="docs/openset_curve.png" width="560" alt="开集拒识曲线"></p>

一句话版本：闭集识别已经可用；开集拒识只到「误拒两成、拒识八成八」的水平，再往上要靠 metric learning 换打分方式。

## 快速开始

```bash
git clone https://github.com/summeriver13/herdface.git && cd herdface
```

### 1. 准备模型

在 [releases](https://github.com/summeriver13/herdface/releases/tag/ml-models) 下载 `ml_models.zip`（457 MB），解压后把 `ml_models/` 放到项目根目录。

### 2. 装依赖

```bash
uv venv && uv pip install -r requirements.txt
# 或者
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

`requirements.txt` 会装 torch 与 ultralytics（数 GB）。只想先看网页骨架、不跑推理，用 `requirements-web.txt`：应用照常启动，只有上传图片的接口会报依赖缺失。

### 3. 启动

```bash
python run.py                  # http://127.0.0.1:5000
FLASK_DEBUG=1 python run.py    # 带调试与自动重载
```

默认 SQLite（`instance/cow_face.db`），首次启动自动建表，不需要额外装数据库。

## 功能与接口

- 注册 / 登录 / 登出
- 上传图片 → 检测牛脸 → 识别个体 → 返回画框图与逐条置信度，结果入库
- 牛只管理：增删改查、每只牛最近 5 条识别记录、完整历史
- 网页里发起 YOLOv8 微调（模型规格、epochs、batch、学习率），后台线程执行，状态页轮询

页面在根路径（`web.*` 蓝图），JSON 接口统一在 `/api` 前缀下（`api.*` 蓝图）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/login`、`/register`、`/logout` | 认证页面 |
| GET | `/dashboard` | 首页，牛只 / 记录 / 训练计数 |
| GET | `/cattle`、`/cattle/<id>` | 牛只列表、详情 |
| GET | `/detect`、`/train`、`/train/status` | 检测、训练、训练状态 |
| POST | `/api/cattle` | 新建牛只 |
| PATCH | `/api/cattle/<id>` | 改名 |
| DELETE | `/api/cattle/<id>` | 删除牛只与其识别记录 |
| POST | `/api/detect` | 上传图片，返回标注图 base64 与检测列表 |
| GET | `/api/train` | 训练任务列表（状态页每 3 秒轮询） |
| POST | `/api/train` | 启动训练，返回 202 |
| DELETE | `/api/train/<id>` | 删除训练记录 |

约定：成功 `{"ok": true, "data": ...}`，失败 `{"ok": false, "error": {"code", "message"}}` 并带对应 HTTP 状态码；写请求需带 `X-CSRF-Token`（页面从 `<meta name="csrf-token">` 取）或表单字段 `csrf_token`。

```bash
python scripts/smoke_test.py    # 22 项端到端检查：页面、接口、CSRF、权限
```

## 项目结构

```
run.py                         入口
app/__init__.py                create_app：配置、蓝图注册、建表、错误处理
app/config.py                  路径、环境变量、模型规格
app/security.py                CSRF token 生成与校验
app/routes/                    页面蓝图 web.*：auth / dashboard / cattle / detect / train
app/api/                       JSON 蓝图 api.*（前缀 /api）：cattle / detect / train
app/models/                    ORM：User / Cattle / RecognitionRecord / ModelTraining
app/controllers/               业务逻辑：识别器、数据集 yaml、训练
app/templates/                 Jinja2 页面，全部继承 base.html
app/static/                    css/app.css、js/{api,cattle,detect,train,training_status}.js
scripts/evaluate.py            自带划分上的检测 / 识别 / 开集评测
scripts/evaluate_opencows.py   跨数据集零样本评测
scripts/train_detect_opencows.py   官方 10 折重训检测
scripts/train_ident_opencows.py    闭集识别 + 开集拒识
scripts/opencows.py            OpenCows2020 解包与划分读取
scripts/smoke_test.py          端到端冒烟测试
scripts/plot_experiments.py    重画实验图
requirements-web.txt           仅 Web 依赖（CI 用，不装 torch）
.github/workflows/ci.yml       CI：ruff + 惰性导入检查 + 冒烟测试
docs/experiments.md            实验记录
docs/preview.webp              检测样例（原 PNG 1.25 MB → WebP 103 KB）
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SECRET_KEY` | `dev-only-insecure-key` | session 签名密钥，部署前必须替换 |
| `DATABASE_URL` | `sqlite:///instance/cow_face.db` | 换 MySQL：`mysql+pymysql://user:pass@host/cow_face_db` |
| `ML_MODELS_DIR` | `./ml_models` | 权重与数据集根目录 |
| `FLASK_DEBUG` | `0` | 设为 `1` 打开调试 |

## 已知限制

- 分类器类别顺序由训练时目录决定，首次加载快照到 `ml_models/yolo-models/*.classes.json`；换数据集重训后要删掉该文件。
- 网页里跑的是自带 10 头牛的闭集分类器，没有 unknown 拒识；开集能力见 [`docs/experiments.md`](docs/experiments.md)，尚未接进界面。
- 训练跑在 Web 进程的后台线程：服务重启会丢任务，无并发保护，不能中途取消。
- 推理在请求线程内同步执行，没有队列与超时控制。
- 跨牧场不成立（0.055），要用在真实牛场必须用自己的数据重训。

## 数据来源

自带数据来自本项目作者采集的 12 头牛。实验部分用到的公开数据集：

| 数据集 | 规模 | 许可 |
| --- | --- | --- |
| [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17) | 2.1 GiB：7,043 张检测帧、46 头牛识别图 | Non-Commercial Government Licence |
| [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7) | 17.6 GiB：检测、182 头牛识别、官方权重 | Non-Commercial Government Licence |

两者都来自布里斯托大学的同一课题组，非商业许可：可用于研究与演示，不能商用。参考文献：*Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning*（[arXiv:2006.09205](https://arxiv.org/abs/2006.09205)）。

## 许可

本仓库代码以 [MIT](LICENSE) 发布。两处例外：

- 数据集与预训练权重不随仓库分发；`ml_models.zip` 里的数据由作者自采，OpenCows2020 / Cows2021 是 Non-Commercial Government Licence。
- 依赖中的 Ultralytics YOLOv8 是 AGPL-3.0：把推理能力作为网络服务对外提供时，需遵守 AGPL 的源码开放要求，或购买 Ultralytics 商业许可。
