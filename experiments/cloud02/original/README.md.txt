# 固定FS教师 → 8B判断模块

遵循PROTOCOL.md；Cloud01的另一种顺序失败不作为本轮准入。使用已保存的24/24 RULE_FS教师，无规则代码状态作为学生输入。原模块、主模型和Bank保持不变。

本地目录：`[LOCAL_PATH]`。

云端只采集24条无规则自然轨迹的Block13 return决策状态；无Bank、无训练。保留原16条BASE的逐Token数值复现。采集包路径`artifacts/runtime.tar.gz`，上传至`[REMOTE_PATH]`。

在新云端目录中，以原环境、独立cache/tmp路径运行：

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/cache" TORCH_HOME="$PWD/cache/torch" XDG_CACHE_HOME="$PWD/cache" TMPDIR="$PWD/tmp"
mkdir -p cache tmp
[REMOTE_PATH] -B src/capture.py --model [REMOTE_PATH] --previous-run [REMOTE_PATH] --output capture
```

取回capture目录及其manifest并逐文件验证，放到本地`returned/capture`。云端不再需要计算时，按用户授权通过AutoDL现有`bash /usr/bin/shutdown`关闭实例，并记录请求与SSH后验；不删除实例或磁盘，不声称直接核实账单。

本地训练：

```bash
cd [LOCAL_PATH]
.venv/bin/python -B v-conditional-cloud02/src/train.py
```

训练只用8条教师软目标，固定一次256步；16条留出不参与拟合或选参。checkpoint先保存再评分。训练结果不自动触发V实验。现有24题为已见开发题，不是独立确认。

数值探针离线测试：`.venv/bin/python -B v-conditional-cloud02/src/test_probe.py -v`，仅使用CPU微型随机模型，不加载主模型。
