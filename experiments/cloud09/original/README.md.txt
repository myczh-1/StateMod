# Cloud09：14B文字预检运行包

状态：本地准备，未进行模型前向。8题和评分合同来自Cloud08，TASKS.md保留原文件标题以体现字节不变；新运行协议见PROTOCOL.md。

模型固定Qwen3-14B `40c069824f4251a91eefaf281ebe4c544efd3e18`，BF16/eager、非thinking。16次完整代码、8次直接类型回答；另16条候选概率路径。两种文字能力分开报告。旧模块、阈值和Bank不动，不打包其权重。

## 1. 本地准备检查

```sh
.venv/bin/python -B v-conditional-cloud09/src/test_precheck.py
.venv/bin/python -B v-conditional-cloud09/tools/verify_prepared.py
```

这些命令不访问远端、不加载语言模型。Tokenizer已用固定14B文件校验；模型权重将在无卡主机下载。

## 2. 用户同步数据盘到无卡主机之后

将artifacts/runtime.tar.gz上传并解压到独立目录：

```text
[REMOTE_PATH]
```

先运行只读盘点和下载预检（下面命令都在新目录内）：

```sh
python3 -B tools/storage_inventory.py --output storage-proposal.json
python3 -B tools/prepare_nogpu.py
```

盘点只列出旧Cloud运行的runs和results.tar候选，不执行删除。先与本地回收证明逐文件核对，再按明确授权清单清理；旧模型、Bank、模块、源码及未核对内容均排除。不能将上次对Cloud05的删除授权自动扩大到其他目录。

空间足够后再下载：

```sh
python3 -B tools/prepare_nogpu.py --download
```

8个权重分片约29.54GB，加配置和Tokenizer合计约29.55GB；建议空闲至少约32.6GiB（含5GiB结果余量）。下载按原始文件直写，不创建另一份HF权重缓存。支持Range续传；SHA失败保留文件，不用其他版本补救。

独立目标目录：`[REMOTE_PATH]`。

迁移盘上的旧Python环境路径预期仍为：`[REMOTE_PATH]`。只核查固定依赖，不自动安装或升级；若数据盘同步后不可用，记录环境问题，GPU不开机，先另行处理。

成功标志：`cpu-preparation/<attempt>/READY_FOR_GPU_MIGRATION.json`。看到它后停止，向用户报告下载和验证已完成，等待用户迁回48GB主机。不在无卡机器运行实验。

## 3. 用户迁回GPU主机并授权后

从本地控制器执行，SSH alias必须由当次主机配置明确指定：

```sh
.venv/bin/python -B v-conditional-cloud09/tools/launch_collect_shutdown.py --execute-after-user-start --host <GPU_SSH_ALIAS>
```

若阶段1已经在目标实验目录解压，控制器识别并核对已准备目录，不能覆盖源文件或已有运行。所有环境、显存、磁盘、模型SHA再次检查。原始结果先取回逐文件校验，再自动关机，本地执行评分。控制器中断只允许--collect-only接管，不能重跑模型。

本轮迁移配置只允许更换SSH地址，实验协议、模型、精度、Prompt和门槛不随主机改变。新的GPU环境若不满足冻结要求，停止并保存原因。

## 4. 输出与解释

execution/evaluation分别输出选型、代码、数值、路径敏感性及Cloud08描述性对照。行为门槛沿用Cloud08；BF16数值门槛另立，未经本机14B实测。首选Token跨路径不一致时单独标注，不宣称数值鲁棒通过。

本轮结束后不编译Bank、不训练模块、不自动扩展任务。14B BF16对8B FP32包含精度变化，不能仅归因为模型规模。
