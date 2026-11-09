# Cloud05：固定简单计算的措辞对照运行包

研究方案：[PROTOCOL.md](PROTOCOL.md)。执行与回收约束：[EXECUTION.md](EXECUTION.md)。本目录准备完成后才封存freeze.json；没有模型实验成绩。

4个旧开发族（乘二、加一、取负、小写），E/N两合同，两版措辞，BASE/TEXT_CODE/MANUAL_V/MODULE_V四组，计划64条生成。模块、阈值、8B Bank和FIRST_DECISION_ONLY冻结不动。

本地检查（不连接服务器）：

```sh
cd [LOCAL_PATH]
.venv/bin/python -B v-conditional-cloud05/tools/check_prepared.py
```

只有用户明确表示原实例已开机并要求执行之后，才运行：

```sh
.venv/bin/python -B v-conditional-cloud05/tools/launch_collect_shutdown.py --execute-after-user-start
```

该命令上传独立运行包、检查环境、执行一次、回收并核验SHA、自动关机，最后本地评分。只使用原远端实例与缓存；不创建/启动实例、不安装环境、不修改历史实验。若条件不满足，保留失败证据，不重试模型。

若作业已启动但控制端中断，只允许恢复观察/回收，禁止再次启动：

```sh
.venv/bin/python -B v-conditional-cloud05/tools/launch_collect_shutdown.py --execute-after-user-start --collect-only
```

合成CPU工程测试：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s v-conditional-cloud05/src -p 'test_*.py' -v
```

这些测试使用随机小型Qwen结构与合成载荷，测试缓存、单次访问、只读探针、异常清理、评分门槛和回收顺序；不是8B行为结果。主模型/远端GPU的实际检查需开机后执行。

包内有冻结Bank与模块文件，不含主模型权重。W_ANCHOR原始Logits引用本机旧证据，仅回收后的锚点审计读取，不向在线判断器传入参考答案。

准备材料封存在freeze.json、PREPARATION_AUDIT.json、manifest.json；真实运行记录以后仅写execution/，准备状态不冒充实验通过。
