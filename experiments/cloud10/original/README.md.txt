# Cloud10：先测文字能力，另报数值差异

新协议承接Cloud09；旧STOP_NUMERICAL及全部文件保持只读。本包本地准备，不开机、不SSH、不进行模型前向。

固定14B模型、原8题、评分合同、规则和解码预算。先保存全部24条Cached自然输出，再执行16条候选概率路径与Fresh同历史重放。Fresh超出原数值界限只记录诊断，不能修改Cached回答或行为成绩；Cached自身工程故障仍硬停止。

## 本地验证

```sh
.venv/bin/python -B -m unittest discover -s v-conditional-cloud10/src -p 'test_*.py'
.venv/bin/python -B v-conditional-cloud10/tools/verify_prepared.py
```

## 用户开机授权后，一次执行

无需重新下载模型。使用原48GB主机：

```sh
.venv/bin/python -B -u v-conditional-cloud10/tools/launch_collect_shutdown.py --execute-after-user-start --host [HOST_ALIAS]
```

远端新目录：`[REMOTE_PATH]`。
模型目录沿用：`[REMOTE_PATH]`。

控制器自动检查环境与空间、只启动一次、回收归档并校验SHA，然后关机，本地评分。出现中断时只恢复回收：

```sh
.venv/bin/python -B -u v-conditional-cloud10/tools/launch_collect_shutdown.py --execute-after-user-start --host [HOST_ALIAS] --collect-only
```

## 输出

- `execution/evaluation/REPORT.md`：逐题文字成绩。
- `SUMMARY.json`：直接选型、完整代码、E联合成功、N新增退化。
- `DECISION.json`：Cached行为判断，单独标记旧Cloud09状态。
- `NUMERICAL_AUDIT.json`：工程探针与正式步数值差异，保留超界与缺失。
- `PATH_SENSITIVITY.json`：同历史排名翻转，不冒充独立Fresh生成。
- 原始Token/Logits、固定工程KV、模型/参数审计、成本、回收SHA与关机记录。

模块、0.5阈值、Bank不加载、不改变。此处仍是已见任务上的开发文字预检，不是独立确认。
