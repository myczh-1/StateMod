# Cloud06：同等简单计算的新族与新措辞独立确认

先读PROTOCOL.md、SEMANTIC_REVIEW.md与PREPARATION_AUDIT.json。4个新计算族、两版未见合同句、E/N与四条件，共16输入、64条生成。模块、阈值、Bank和原注入接口不变；评分器原样使用Cloud05-Rescore01。

准备期间不连接服务器。只读本地完整性检查：

```sh
.venv/bin/python -B v-conditional-cloud06/tools/check_prepared.py
```

只有用户明确开机并要求执行后运行：

```sh
.venv/bin/python -B v-conditional-cloud06/tools/launch_collect_shutdown.py --execute-after-user-start
```

会先检查环境与空间；证据取回校验后关机，再本地评分。远端空间尚未实测，可能受上轮留档影响；本包不会自行删除旧结果。没有模型、阈值或Bank搜索。

若已启动，只恢复取回，不重启模型：

```sh
.venv/bin/python -B v-conditional-cloud06/tools/launch_collect_shutdown.py --execute-after-user-start --collect-only
```

本地合成工程和执行合同测试：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s v-conditional-cloud06/src -p 'test_*.py' -v
```

合成小模型不是主8B模型，不能作为确认行为证据。第一次正式前向前以freeze.json和runtime.tar.gz封存；准备完成后不得改任务、措辞、测试、门槛或源码。

最终输出逐任务原始生成/概率/Bank访问、能力与匹配/人工V/自动V分离评分、全部失败、成本、数值审计和manifest。所有旧状态保持不变。通过也只支持4个具体计算族上的小样本确认，不宣称通用适用性识别。
