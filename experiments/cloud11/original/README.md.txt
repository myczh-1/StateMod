# Cloud11-J：准备好的14B判断模块实验

范围：只采集、拟合、验证新判断模块，不编译或加载Bank。旧模块、旧阈值和旧Bank不变。[冻结协议](PROTOCOL.md)。

模型：已缓存Qwen3-14B / BF16 / CUDA / eager。16条BASE、8条新老师选型、16条完整候选评分路径；8条原文字老师直接复用。Cached为主、Fresh差异独立描述。

先用下面的命令做本地校验，无SSH、无主模型加载：

```sh
cd [LOCAL_PATH]
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v-conditional-cloud11/tools/verify_prepared.py
```

**只有用户确认开机后**执行：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v-conditional-cloud11/tools/launch_collect_shutdown.py --execute-after-user-start --host [HOST_ALIAS]
```

该命令检查远端空间/环境，排他启动一次采集，取回归档并核验SHA，立即自动关机，然后在本地训练和验证。不修改任何历史目录，不下载新模型，不自动清理。

断线后只恢复收集：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v-conditional-cloud11/tools/launch_collect_shutdown.py --execute-after-user-start --host [HOST_ALIAS] --collect-only
```

本地拟合已完成时不会重跑。若已取回但尚未本地评价，可在新的、尚不存在的输出目录运行：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v-conditional-cloud11/src/evaluate.py --run v-conditional-cloud11/execution/runs --output v-conditional-cloud11/execution/evaluation
```

本地测试使用随机小型CPU模型和合成数组，仅检查实现，不使用真实14B隐藏状态训练。测试产物写临时目录后销毁。准备状态不代表实际采集、训练或行为验证成功。

预定主标准：原八题与老师一致≥7/8，其中N拒绝4/4、E启用≥3/4；原八题不参加拟合，但与训练题同族，故只称开发措辞留出。BASE功能失败不作为准入阻断。判断通过也不会自动开始V实验。
