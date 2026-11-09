# Cloud12：措辞与围栏交叉诊断

8个固定前缀条件。没有新训练、完整代码生成或 V 注入。旧模块及0.5阈值不变。

## 本地验证（不连接服务器）

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B v-conditional-cloud12/tools/verify_prepared.py
```

## 用户开机后执行（当前未执行）

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B v-conditional-cloud12/tools/launch_collect_shutdown.py --execute-after-user-start --host [HOST_ALIAS]
```

先检查磁盘/显存/软件与SHA；8格Cached重放后做Fresh诊断；下载结果并逐文件核验；自动关机；本地使用冻结模块评分。独立输出路径为 `[REMOTE_PATH]`。不修改已有远端目录，不清理文件。

若通信中断，禁止重复启动；先检查 `execution/LAUNCH_ATTEMPT.json`，已有运行只能用 `--collect-only` 恢复取回。

## 本地复算（仅在已取回数据后）

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B v-conditional-cloud12/src/evaluate.py --run v-conditional-cloud12/execution/runs --output v-conditional-cloud12/reanalysis
```

输出目录必须不存在。所有正式原始数组、报告和失败记录保留。完整冻结定义见 `PROTOCOL.md`。
