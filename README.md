# Qwen3-Reranker 高通 NPU 部署（QNN / HTP）

在高通 **QCS8550** 开发板（Ubuntu，aarch64）的 Hexagon NPU 上，以 fp16 精度运行
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)，支持 **4096 token 上下文**。

本仓库包含两部分：
- **转换流水线**：ONNX 导出 → 精度验证 → Qualcomm AI Hub 编译 → 下载 → 打包；
- **板端推理代码**：C++ QNN 运行时 + Python API。

| 项目 | QCS8550 开发板实测 |
|---|---|
| 上下文长度 | 4096 token（含 prompt 模板） |
| 精度 | NPU 上 fp16；logit 差值和 fp32 PyTorch 的误差在 0.05 以内 |
| 延迟 | 满长度时每个 query–文档对 **4.56 秒**（4 段 × 约 1.1–1.2 秒） |
| 加载时间 | 约 4 秒，只需一次（4 个 context binary 常驻） |
| CPU 侧开销 | 约 13 毫秒（分词、查 embedding、数据拷贝） |

## 原理

4096 token 下，Hugging Face 原版模型无法直接放进 NPU。因此先把它改写成静态的、数值等价的计算图（`export/modeling.py`）：

- **静态形状** `[1, 4096]`，左侧补齐，int32 输入；prompt 模板在主机端拼好
- **embedding 在 CPU 上查表**（fp16 词表，mmap 加载）：4096 长度下，带 embedding 查表的图在 link（HTP 图准备）时 7 次有 6 次失败，去掉 embedding 后 4 次全部成功
- **28 层 decoder 切成 4 个依次执行的图**，每个图一个 QNN context binary
- **按 query 分块的注意力**：query 每 512 个一块，每块只和因果上可见的那段 key 计算。最大的注意力张量从 512 MB 降到 64 MB（不分块的版本在真机上会报 `QNN_COMMON_ERROR_MEM_ALLOC`）
- mask 在图内由 `attention_mask` 生成；`lm_head` 只保留 `yes`/`no` 两行，输出 `logits [1,2]` 和 `P(yes) [1]`

板子上，`device/` 把 4 个 context binary 作为一个 HTP context 组加载（放不进一个进程域时，QNN 会自动启用第二个），依次执行，段与段之间传递 hidden state。

```
文本 ──分词/拼 prompt/补齐──▶ ids ──查 embedding（CPU）──▶ hidden ──▶ 第1段 ─▶ 第2段 ─▶ 第3段 ─▶ 第4段 ──▶ logit(yes) − logit(no)
                              （Python）                              （NPU，QNN context binary）
```

## 目录结构

```
export/                 主机端（x86_64 Linux）
  modeling.py           面向 NPU 的模型改写（分块注意力、分段）
  export_onnx.py        1. 导出 ONNX 分段 + fp16 embedding 表 + manifest.json
  verify_onnx.py        2. onnxruntime 与 PyTorch 对比（含一条满长度文档）
  aihub_compile.py      3. AI Hub：上传、编译 fp16 QNN、link 成 context binary、性能测试
  aihub_eval.py            （可选）通过 AI Hub 在真机上串联各段，验证端到端精度
  aihub_download.py     4. 下载 context binary（分块、可断点续传）
  make_deploy.py        5. 组装板端部署包（从你本地的 SDK 拷贝 QNN 运行库）
device/                 板端（aarch64 Linux）
  src/qnn_reranker.cpp  基于 QNN C API 的 C++ 运行时（C ABI，通过 ctypes 调用）
  qwen3_reranker.py     Python API：分词、拼 prompt、补齐、查 embedding、排序
  run_reranker.py       命令行：info / selftest / bench / rerank
  serve_reranker.py     rerank HTTP 服务（FastAPI，Jina/Cohere 兼容接口）
  setup_env.sh          设置 LD_LIBRARY_PATH / ADSP_LIBRARY_PATH
  tools/diagnose_npu.sh FastRPC / NPU 访问诊断
scripts/run_pipeline.sh 一键跑完上述全部步骤
qairt/                  放置 QAIRT SDK 的目录（SDK 需自行下载，见其中的 README.md）
docs/NOTES.md           实测数据与踩坑记录
```

## 环境要求

- **主机**：x86_64 Linux，Python 3.10+，`pip install -r requirements-export.txt`（CPU 版 torch 即可），
  内存约 16 GB，磁盘约 15 GB。
- **Qualcomm AI Hub** 账号（[workbench.aihub.qualcomm.com](https://workbench.aihub.qualcomm.com)）：
  执行 `qai-hub configure --api_token <你的 token>`。
- **QAIRT SDK**（[Qualcomm AI Runtime](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)），
  版本必须**和 AI Hub 编译时用的一致**（本项目用的是 2.50.0）。SDK 里的运行库、头文件和工具受高通许可证约束，
  **不包含在本仓库中**：请下载后解压到仓库的 [`qairt/`](qairt/README.md) 目录，由 `make_deploy.py` 从中拷贝需要的文件。
- **开发板**：QCS8550，Ubuntu 22.04 及以上（glibc ≥ 2.34），有高通 FastRPC（`libcdsprpc.so`、`/dev/adsprpc-smd`），
  Python 3 并安装 `pip install -r requirements-device.txt`。

## 生成模型部署包

```bash
pip install -r requirements-export.txt
qai-hub configure --api_token <你的 AI Hub token>
# 把 QAIRT SDK 解压到 qairt/ 目录（见 qairt/README.md），然后：
bash scripts/run_pipeline.sh
```

这条命令会依次完成：下载模型；导出并验证 ONNX 分段（`max |diff|` 约 1e-4）；在 AI Hub 上编译（每段 link 约 10 分钟）；
下载 binary（约 1.2 GB）；生成 `dist/deploy/`。每一步也可以单独运行，参见 `scripts/run_pipeline.sh`。
可调参数：`SEQ_LEN`、`PARTS`、`ATTN_CHUNK`、`DEVICE`、`QNN_TARGET`、`HTP_ARCH`、`EVAL=1`。

## 在开发板上运行

```bash
# 把 dist/deploy 拷到板子上，然后：
cd deploy
pip install -r requirements-device.txt       # numpy、tokenizers、fastapi、uvicorn
make                                         # 仅当 lib/libqnn_reranker.so 没有预先交叉编译时需要
source setup_env.sh
python3 run_reranker.py info                 # 加载 4 个 context binary，打印输入输出张量
python3 run_reranker.py selftest             # 与 fp32 参考结果对比 -> PASS
python3 run_reranker.py bench -n 5
python3 run_reranker.py rerank -q "中国的首都是哪里？" -d "北京是中华人民共和国的首都。" -d "今天天气很好。"
```

```python
from qwen3_reranker import Qwen3Reranker

rr = Qwen3Reranker("/path/to/deploy")        # 加载一次（约 4 秒），之后常驻复用
for r in rr.rerank("中国的首都是哪里？", ["北京是中华人民共和国的首都。", "今天天气很好。"]):
    print(f"{r.margin:+.3f}  {r.score:.4f}  {r.text}")
```

- **按 `margin` 排序**（= logit(yes) − logit(no)）。`score` = P(yes)，在 fp16 下接近 1 时会饱和，可能出现平分。
- `instruction=` 用来设置任务指令（默认是网页搜索），和官方用法一致。
- 超长输入会从**文档末尾**截断。更长的文档可以切成多个窗口分别打分，取最高的 margin。

## 部署为 rerank 服务（OpenAI 风格接口）

`device/serve_reranker.py` 基于 FastAPI 提供 HTTP 服务。OpenAI 官方 API 没有 rerank 接口，这里实现的是业界通用的
**Jina / Cohere 格式**（vLLM、Xinference、LocalAI 等采用的格式），Dify、FastGPT、LangChain 等应用可以直接接入。

```bash
source setup_env.sh
python3 serve_reranker.py --host 0.0.0.0 --port 8000 --api-key sk-你的密钥    # 不设 --api-key 则无需鉴权
```

| 接口 | 说明 |
|---|---|
| `POST /v1/rerank`（也支持 `/rerank`、`/v2/rerank`） | 重排序 |
| `GET /v1/models` | OpenAI 格式的模型列表 |
| `GET /health` | 健康检查（无需鉴权） |
| `GET /docs` | 交互式接口文档 |

```bash
curl http://<板子IP>:8000/v1/rerank \
  -H "Authorization: Bearer sk-你的密钥" -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Reranker-0.6B", "query": "中国的首都是哪里？",
       "documents": ["北京是中华人民共和国的首都。", "今天天气很好。"]}'
```

QCS8550 开发板上的实际返回：

```json
{"id": "rerank-dd8c83928486490b97b190d6967751cc", "object": "rerank", "model": "Qwen3-Reranker-0.6B",
 "results": [{"index": 0, "relevance_score": 0.9982453872602942, "margin": 6.343750953674316,
              "document": {"text": "北京是中华人民共和国的首都。"}},
             {"index": 1, "relevance_score": 0.000012768001378939149, "margin": -11.268555641174316,
              "document": {"text": "今天天气很好。"}}],
 "usage": {"total_tokens": 168}, "meta": {"elapsed_s": 8.846, "documents": 2}}
```

图的形状固定为 4096，所以短文本也按满长度计算，每篇约 4.4–4.6 秒。

- 请求字段：`query`、`documents`（字符串，或 `{"text": ...}` 对象）、`top_n`、`return_documents`（默认 true），
  以及扩展字段 `instruction`（任务指令）。
- `relevance_score` 是 P(yes)，范围 0–1。结果按 `margin` 排序，额外返回的 `margin` 字段不影响兼容性。
- NPU 同一时间只处理一对 query–文档：请求会异步接收、排队串行推理，排队期间服务仍能正常响应。
  **只能启动一个服务进程**，每个进程都会加载一份模型，多开会耗尽 NPU 内存。
- 满长度时每篇文档约 4.6 秒，20 篇约 90 秒。请调大客户端超时，或者先用向量检索缩小候选集。
  单次请求的文档数上限由 `--max-documents` 控制（默认 64）。
- 接入 Dify 等应用时，选择支持 Jina/Cohere 格式 rerank 的供应商（例如 Dify 的「OpenAI-API-compatible」，模型类型选 Rerank），
  把 API 地址填为 `http://<板子IP>:8000/v1`。服务同时响应 `/rerank` 和 `/v1/rerank`，两种路径拼法都能用。
  具体配置项以各应用的版本为准。

开机自启（systemd）示例，把路径和用户换成你自己的：

```ini
# /etc/systemd/system/qwen3-reranker.service
[Unit]
Description=Qwen3-Reranker on QNN NPU
After=network.target

[Service]
# 该用户需要在 system 组中，才能访问 /dev/adsprpc-smd
User=<你的用户>
WorkingDirectory=/path/to/deploy
Environment=RERANKER_API_KEY=sk-你的密钥
ExecStart=/bin/bash -c 'source setup_env.sh && exec python3 serve_reranker.py --host 0.0.0.0 --port 8000'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now qwen3-reranker
```

如果 `info` 失败，运行 `bash tools/diagnose_npu.sh`，并参考 [docs/NOTES.md](docs/NOTES.md) 里的排错表。
最常见的两个原因：QNN 运行库和芯片不匹配；当前用户没有 `/dev/adsprpc-smd` 的访问权限。

## 其他设备 / 其他长度

流水线是参数化的，但只有默认配置（QCS8550、4096 token、4 段、fp16）做过端到端验证。
换其他骁龙芯片时，需要设置 `DEVICE`、`HTP_ARCH` 和 `QNN_TARGET`（参考 QAIRT 文档里的 "supported Snapdragon devices" 表）。
更短的上下文需要的段数更少。作为参考，我们用另一种导出方式（embedding 在图内、常量 mask）测过 2048 token 单图版本，QCS8550 上约 1.8 秒；本流水线的 `SEQ_LEN=2048 PARTS=1` 配置没有验证过。

## 许可证

本仓库代码采用 Apache-2.0 许可证。Qwen3-Reranker 由 Qwen 团队以 Apache-2.0 许可证发布。
Qualcomm AI Hub、QAIRT SDK 及其运行库遵循高通自己的条款。
