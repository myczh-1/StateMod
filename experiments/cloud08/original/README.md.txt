# Cloud08：8题文字预检，准备完成但未运行

固定四个已见开发族：整数乘二、字符串小写、整数八进制文本、布尔取反。每族E/N两种返回后使用合同，共8题。模型、模块、阈值和Bank不变。

|记录|数量|评价|
|---|---:|---|
|BASE完整代码|8|原成员计算和可变合同|
|TEXT_CODE完整代码|8|同一条件文字规则下的功能、真实类型和退化|
|TEXT_SELECT自然短回答|8|原RULE_FS包装；选型与格式是否正确|
|TEXT_SELECT完整候选评分|16条路径|分别记录名称、名称+EOS总logP，不按Token数归一化|

候选实际Tokenizer：`frozenset`=[69,299,18549,295]，`set`=[746]；各追加EOS=151645。不使用生成代码中的带前导空格构造器Token来代替直接答案计分。

两种文字基线独立报告。自然选型失败不能由强制候选高分修正；选择正确不能代替完整代码执行通过。完整门槛见PROTOCOL.md。

任务与可执行用例：TASKS.md / TASKS.json。精确三条件输入及Token：data/inputs.json。原老师包装：data/select_wrapper.json。保护清单：data/source_sha.json、data/protected_artifacts.json。

本地检查：

```sh
.venv/bin/python -B v-conditional-cloud08/src/test_precheck.py
.venv/bin/python -B v-conditional-cloud08/tools/verify_prepared.py
```

以上不会连接服务器、加载模型或执行实验。CPU只执行gold/负例与模拟Logits检查。

用户开机并授权后，才执行：

```sh
.venv/bin/python -B v-conditional-cloud08/tools/launch_collect_shutdown.py --execute-after-user-start
```

运行前检查缓存模型、固定版本、GPU独占和至少5 GiB结果盘空闲；本地至少6 GiB。空间不足停止，不自动清理旧数据。取回归档并逐文件SHA校验后自动关机，随后在本地评分。本轮无Bank、无判断模块调用，不自动进入新的确认。

只读重评已回收输出须使用新目录：

```sh
.venv/bin/python -B v-conditional-cloud08/src/evaluate.py --run v-conditional-cloud08/execution/runs --output v-conditional-cloud08/reproduction-new
```

资源规划保留逐步Cached/Fresh完整词表数组，不额外重复保存整段Logits堆叠。5 GiB准入按96/8 Token预算的原始数组、归档及缓冲估算，并非已测得的运行成本。本次没有核查远端实时空间。
