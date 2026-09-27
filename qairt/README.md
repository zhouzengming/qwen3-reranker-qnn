# 请把 QAIRT SDK 放在这个目录下

本项目的转换流水线需要高通的 **QAIRT SDK**（Qualcomm AI Runtime，原 Qualcomm AI Engine Direct / QNN SDK）。
受高通许可证约束，SDK **不能随本仓库分发**，需要你自行下载后放到这个 `qairt/` 目录中。

## 1. 下载 SDK

本项目基于 **QAIRT 2.50.0.260828** 构建和验证，直接下载（约 2.6 GB）：

- **https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip**

也可以在 [Qualcomm Software Center](https://softwarecenter.qualcomm.com/)（搜索 "Qualcomm AI Runtime"）或
[产品页面](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) 下载。
使用前请阅读并遵守 SDK 附带的许可协议（`LICENSE.pdf`）。

- **版本必须和 Qualcomm AI Hub 编译模型时使用的版本一致。** 版本不一致时，开发板加载 context binary 可能失败。
  可以在 AI Hub 编译任务的日志里确认版本：日志中会出现 `/qairt_sdk/default/2.50.0/...` 这样的路径。
  AI Hub 升级默认 SDK 后，请下载对应版本（把链接中的两处版本号换成新的完整版本号）。

## 2. 解压到本目录

把压缩包直接解压到这里。解压后的目录结构应类似于下面两种之一，脚本都能识别：

```
qairt/
├── README.md                 ← 本文件
└── 2.50.0.260828/            ← SDK 根目录（名称随版本号变化）
    ├── bin/
    ├── include/QNN/
    ├── lib/
    │   ├── aarch64-oe-linux-gcc11.2/
    │   ├── hexagon-v73/
    │   └── ...
    └── sdk.yaml
```

```
qairt/
├── README.md
└── qairt/
    └── 2.50.0.260828/        ← 官方压缩包自带一层 qairt/ 目录，保持原样也可以
```

例如：

```bash
cd qairt
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip
unzip -q v2.50.0.260828.zip && rm v2.50.0.260828.zip    # 解压后是 qairt/qairt/2.50.0.260828/
```

判断方法：SDK 根目录下应该有 `include/QNN/QnnInterface.h` 和 `sdk.yaml` 这两个文件。

## 3. 运行流水线

回到仓库根目录，直接运行：

```bash
bash scripts/run_pipeline.sh
```

脚本会在本目录下自动查找 SDK，并打印检测到的路径和版本。以下情况需要手动指定路径：
- 本目录下放了多个版本的 SDK；
- SDK 放在其他位置。

手动指定的方法是设置环境变量 `QAIRT_SDK`：

```bash
QAIRT_SDK=$PWD/qairt/2.50.0.260828 bash scripts/run_pipeline.sh
```

## 流水线用到了 SDK 的哪些内容

只有最后一步 `export/make_deploy.py` 需要 SDK，它会从 SDK 里拷贝以下文件到板端部署包：

| SDK 中的路径 | 用途 |
|---|---|
| `lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so` 等 | 开发板上的 QNN 运行库（QCS8550 在 Linux 上只能用 gcc11.2 这套） |
| `lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so` | 运行在 NPU（CDSP）上的一侧 |
| `include/QNN/` | 在开发板上编译 `libqnn_reranker.so` 用的头文件 |
| `bin/aarch64-oe-linux-gcc11.2/qnn-platform-validator` 等 | 板端诊断工具 |

模型的编译在 Qualcomm AI Hub 云端完成，本地不需要 SDK 里的转换工具。

> 本目录下除本文件外的所有内容都已写入 `.gitignore`，不会被提交到仓库。
