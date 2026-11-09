# Cloud03：8B静态V接回冻结判断模块

当前 **PREPARED_NOT_RUN**。本目录只做本地准备；尚未连接GPU、编译主模型Bank或执行行为实验。

- 冻结判断：Cloud02 checkpoint SHA `3a35f380d49fdbf9d76832d252554c8f9c55b715ed1fc7a3ad3d47ccda6bace7`，阈值0.5。
- 新V：原静态规则在8B上编译，固定14–27层、36侧槽、w=1、FIRST_DECISION_ONLY。
- 16条拟合留出开发题；BASE / MANUAL_V / MODULE_V新运行；TEXT_CODE用原同题冻结参照。
- 原1.7B Bank和所有旧实验只读；不是8B效果已成立。

协议见PROTOCOL.md。所有输入与源码见freeze.json。开发模块/Bank行为与独立确认分别报告。

## 用户开机后才执行

```bash
cd [LOCAL_PATH]
.venv/bin/python -B v-conditional-cloud03/tools/launch_collect_shutdown.py --execute-after-user-start
```

该命令依次：校验本地来源→连接原statemod-gpu→上传独立目录→检查环境→编译并封存8B载荷→工程审计及48条开发轨迹→回收校验→关机→本地执行评分。不会启动/租用实例；目录已存在即拒绝覆盖。没有失败后自动重试实验。

本地启动器终端必须保留；远端60分钟看门狗作为失联兜底。关机仅停止实例，不删除磁盘或释放资源。若下载失败，远端完整结果保留，后续应只恢复原归档。SSH拒绝连接不等于已读取平台账单。

## 本地无主模型检查

```bash
.venv/bin/python -B v-conditional-cloud03/src/test_backend.py -v
.venv/bin/python -B v-conditional-cloud03/src/test_pipeline.py -v
.venv/bin/python -B v-conditional-cloud03/tools/check_prepared.py
```

测试只用微型随机CPU Qwen及已保存的数值状态。没有读取主模型权重来进行前向，没有创建真实8B Bank。

## 文件

- src/compile_bank.py：规则正文→8B侧K/V；独立RoPE审计。
- src/backend.py：原V算法的显式8B适配、独立Attention和历史重算。
- src/readout.py：只读原checkpoint，无训练接口。
- src/run.py：真实同检查点分支、访问追踪、失败封存。
- src/evaluate.py：关机后函数执行评分和分项判定。
- tools/launch_collect_shutdown.py：显式未来启动、结果校验及正常关机。
- tools/remote_job.py、shutdown_watchdog.py：限时远端作业及异常关机。
- artifacts/runtime.tar.gz：冻结运行包；package.json：包SHA。

新实验固定的是绝对14–27层；8B共有36层，不称与1.7B最后14层深度等价。8B的GQA参数由模型读取，原硬编码repeat=2已去除；合成测试另使用4:1 GQA验证形状处理。方法沿用旧工程，不声称新8B载荷已经通过行为验证。
