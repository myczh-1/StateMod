# Cloud05-Rescore01

这是Cloud05保存生成结果的本地修订评分，不运行模型、不连接服务器。先看REPORT.md，精确修改见SCORER_DIFF.patch。

已完成：14项合同测试、64条原评分复现、64条新评分、逐条差异与旧目录SHA保护。原始Cloud05状态不覆盖。

## 复算

在StateMod项目根目录，用已有本地Python运行：

```sh
.venv/bin/python -B v-conditional-cloud05-rescore01/src/test_scoring.py
.venv/bin/python -B v-conditional-cloud05-rescore01/src/rescore.py --output /tmp/statemod-cloud05-rescore01-reproduction
```

输出目录必须不存在，脚本拒绝覆盖。正式冻结文件已存在，不要再次执行--freeze。脚本先验证冻结源码，再用原评分器复现全部结果，随后新评分，最后验证原目录SHA；不读模型权重、不触发远端执行。

results/DECISION.json为修订开发分析状态；原v-conditional-cloud05/execution/evaluation/DECISION.json仍是原状态。原工程和数值结果从旧审计继承，本轮没有进行新数值前向。

主要文件：PROTOCOL.md、freeze.json、TEST_RESULTS.txt、OLD_FILES_BEFORE.json、results/SCORED_RESULTS.json、results/SCORE_DIFF.json、results/ORIGINAL_SCORER_REPLAY.json、results/SUMMARY.json、results/MATCHING.json、results/FAILURES.json、results/audit.json、manifest.json。
