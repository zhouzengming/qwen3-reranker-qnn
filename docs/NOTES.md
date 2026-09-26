# 实测数据与踩坑记录

以下数据均基于 QAIRT 2.50.0、Qualcomm AI Hub（QCS8550 Proxy）和一块运行 Ubuntu 22.04 的 QCS8550 开发板。

## 实测数据

| 配置（未注明的均为 fp16） | QCS8550 上的结果 |
|---|---|
| L=512，单图 | 201 ms，1669 个算子全部在 NPU 上；正负样本排序与 fp32 完全一致 |
| L=512，w8a16（AI Hub 量化，400 条校准样本） | 210 ms（没有变快），与 fp32 的 Spearman 相关系数 0.90：**得不偿失** |
| L=2048，单图 | 1.8 s（长度 ×4 → 耗时 ×9，注意力是平方复杂度） |
| L=4096，单图 | 能编译，link 约 2 小时，运行时报 `QNN_COMMON_ERROR_MEM_ALLOC` |
| **L=4096，4 段 + 分块注意力 + embedding 放 CPU** | **每段 1.10 / 1.12 / 1.12 / 1.21 s，合计 4.56 s** |

在真机上串联各段做端到端验证（AI Hub inference job，8 条样本，171–4096 token）：logit margin 平均误差 0.019，
最大 0.036，正负样本顺序 100% 保持。在开发板上运行 `selftest`，最大误差 0.045。

**排序请用 logit margin，不要用 P(yes)。** fp16 下 P(yes) 接近 1 时的最小刻度约 0.0005–0.002，会出现平分
（L=512 时 60 对里有 4 对打平）；而 margin 能正确排序（60/60）。

## 为什么必须改写模型

1. **注意力的内存占用。** L=4096 时一层的注意力分数是 16 头 × 4096 × 4096 × 2 字节 = 512 MB，加 mask 和 softmax
   时还会同时存在好几个这么大的张量。把 query 按 512 分块（每块只和位置 `< end` 的 key 计算），就能把它压到 64 MB。
   数学上完全等价，而且跳过了被因果 mask 掉的 key，注意力的计算量还少了约一半。
2. **4096 下图里带 embedding，link 基本都会失败。** 带 `input_ids → Gather` embedding 的 4096 图一共试了 7 次，
   6 次在 link 时报 `graph_prepare.cc: Unable to find op in specified graph context`。失败的几次包括常量 mask、
   分块注意力、7 层分段，以及只有 1 层 decoder 的探针模型。唯一成功的一次是最早那个 28 层、不分块、arange mask 的版本，
   原因不明，也没能复现。同样结构的图去掉 embedding（直接输入 hidden state）后，4 次 link 全部成功。
   所以 embedding 改在主机端查表。
3. mask 在图内由两个位置向量比较生成，不存 L×L 的常量。4096 下用常量 mask 的版本也 link 失败了，但它同样带着
   embedding，无法确定是哪个原因导致的。
4. mask 值用有限的 −100，而不是 `finfo.min`：结果完全一样，对 fp16 和量化更友好。

## 设备上的 HTP 内存

- L=4096 时，每个 7 层的 context 在 HTP 上约占 1 GB，其中约 400 MB 是 spill-fill 临时缓冲区
  （可以从 binary 元数据的 `QnnHtpSystemContext_GraphBlobInfo.spillFillBufferSize` 读到）。4 个放不进一个
  HTP 进程域（PD）：报 `Failed to find available PD ... context size estimate`。
- **把 4 个图 link 进一个 context binary 没有用**：AI Hub 会以权重共享的方式 link，但每个图的临时内存并不共享，
  预估占用 3.9 GB。
- `QnnContext_createFromBinaryListAsync` + `QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES`（Genie 加载分段 LLM
  的方式）在 QCS8550 的 Linux 后端上**不被支持**：`Backend does not support shared resources enabled optimization`。
- **实测可行的方案**：用 `QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS` 把这 4 个 context 注册为同一组
  （`maxSpillFillBuffer` 取各段最大值），加载成功，用时约 4 秒。QNN 日志显示第 4 段被放进了另一个 PD（`pdId 2`），
  执行时没有可测量的额外开销。日志里同时有 `This option (2) is only for create context from binary use case` 的警告，
  所以无法确定真正起作用的是分组共享 spill-fill，还是 QNN 自动启用第二个 PD。
- **x86 HTP 模拟器**不支持权重共享布局的多图 context binary、多 context 分组和 list-async 接口。
  单图 binary 可以在模拟器上运行（很慢：一个 7 层的段约 19 分钟），适合用来检查主机端代码。

## Linux 上的 QNN 运行库

- SoC 型号从 `/sys/devices/soc0/soc_id` 读取（QCS8550 = 603 → QNN SoC 编号 66）。QAIRT 2.50 中只有
  **`aarch64-oe-linux-gcc11.2`** 这套库支持 QCS/QCM8550（见 QAIRT 文档的 "supported Snapdragon devices" 表）。
  实测 `gcc9.3` 这套会在后端初始化时报 `Dsp startup: Unsupported SoC model (SnapdragonModel): 66`，
  通过设备配置指定 SoC 型号也无效，因为这个检查发生在 `deviceCreate` 之前。`ubuntu-gcc9.4` 这套没有 V73 stub，
  用不了 v73 NPU（没有实测）。
- gcc11.2 这套库要求 glibc ≥ 2.34、GLIBCXX_3.4.29，即 Ubuntu 22.04 及以上。
- AI Hub 为 "QCS8550 (Proxy)" 编译的 binary，目标其实是 SM8550（SoC 编号 43，同样是 v73 NPU），
  在真正的 QCS8550 上可以正常加载。

## FastRPC / 权限

- `/dev/adsprpc-smd` 的权限通常是 `crw-rw---- system system`。没有访问权限时，QNN 会先报
  `createUnsignedPD unsigned PD or DSPRPC_GET_DSP_INFO not supported`，然后退回签名进程域，
  接着报 `openSessionForPriority failed ... status=0x00000200` 失败（SDK 里的 skel 是未签名的）。
  修复：`sudo usermod -aG system $USER`，然后重新登录。可以用 `qnn-platform-validator --backend dsp --testBackend`
  确认：root 下通过、普通用户下失败，就是这个问题。
- `ADSP_LIBRARY_PATH` 必须包含 `libQnnHtpV73Skel.so` 所在的目录（`setup_env.sh` 已经设好）。

## Qualcomm AI Hub 使用技巧

- `--target_runtime qnn_context_binary` 已经没有了，改用 `submit_compile_and_link_jobs`（先编译成 `qnn_dlc`，再 link）。
- 超过 2 GB 的 ONNX 模型要以目录形式上传：`name.onnx/{model.onnx, model.data}`。
- 编译后的 context binary 会把输出重命名为 `output_0`、`output_1`……；输入名保留，但顺序可能变化
  （`attention_mask` 排在了前面），所以要按名字匹配输入输出张量。
- 对多图 context 做性能测试，需要加 `--qnn_options context_enable_graphs=<单个图名>`。
- 客户端的读超时只有约 3 秒，长时间轮询任务要加重试。
- qai_hub 通过 S3 传输大文件（上传模型、推理任务的输入输出、性能报告、模型下载）时，用的是 256 MiB 分片，而且没有总超时。
  经过代理时，连接可能半死不活：每隔一会儿漏一点数据，读超时永远不会触发，调用就一直挂着。实测下载推理输出时就卡住过。
  本仓库的做法：所有传输数据的调用都放在带总时限的线程里执行，超时就重试（`aihub_compile.retry(..., timeout=)`）；
  模型下载改用 16 MiB 分片，每片都有总时限，可以断点续传（`aihub_download.py`）。
- 4096 长度的图 link 耗时：7 层分块版约 10 分钟，28 层不分块版约 2 小时。

## 开发板排错

| 现象 | 原因 / 解决方法 |
|---|---|
| `Dsp startup: Unsupported SoC model (SnapdragonModel): 66` → `deviceCreate failed` | QNN 运行库版本不对；QCS8550 需要 `aarch64-oe-linux-gcc11.2`（`make_deploy.py --qnn-target`） |
| `createUnsignedPD ... not supported`，接着 `openSessionForPriority failed ... 0x00000200` | 没有 `/dev/adsprpc-smd` 的访问权限：把用户加入它所属的组（`system`） |
| `Failed to find available PD ... context size estimate` | HTP 内存不足；要用 4 个独立 binary 分组加载（默认方式），不能用 4 个图 link 成一个 binary 的版本 |
| `dlopen ... libcdsprpc.so` 失败 | BSP 里缺少 FastRPC 用户态库，或者不在 `LD_LIBRARY_PATH` 中 |
| 其他 `contextCreateFromBinary ... failed` | QNN 运行库版本必须和 AI Hub 编译时用的 SDK 版本一致 |
| 每段明显慢于约 1.2 秒 | 默认已开启高性能模式（`--no-burst` 可关闭）；检查是否过热降频 |
