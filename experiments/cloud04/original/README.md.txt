# Cloud04：新任务与新合同措辞独立确认

准备完成后以freeze.json与PREPARATION_AUDIT.json为准。本地准备不连接服务器，不加载8B主模型作前向。

冻结对象：Cloud02判断模块、阈值0.5、Cloud03的8B Bank；Block14–27、36侧槽、权重1、FIRST_DECISION_ONLY不变。

8族×E/N×W3/W4=32项；BASE / TEXT_CODE / MANUAL_V / MODULE_V，共128条新生成。每套措辞分别验收，详见PROTOCOL.md。任务语义近邻见SEMANTIC_REVIEW.md；金标准测试只验证评分合同，不能当作模型能力结果。

## 本地核验（不连接服务器）

在StateMod根目录：

```sh
.venv/bin/python -B v-conditional-cloud04/tools/check_prepared.py
```

合成CPU工程测试重现（不运行主模型）：

```sh
.venv/bin/python -B -m unittest discover -s v-conditional-cloud04/src -p 'test_*.py'
```

## 用户开机并授权后才执行

```sh
.venv/bin/python -B v-conditional-cloud04/tools/launch_collect_shutdown.py --execute-after-user-start
```

脚本校验本地冻结包和旧保护文件后上传；远端从已有缓存加载模型和Bank，不编译、不训练。使用原statemod-gpu别名；不会开机。正式路径存在时拒绝覆盖。

GPU空闲/资源检查通过后启动限时任务。结果取回、SHA核验后自动关机，再本地评分；90分钟看门狗只防失联，正常不等待它。远端结果保留在持久盘。传输或SSH回执未知时不得重启模型任务；先读取LAUNCH_PID、JOB_ARMED.json、READY_TO_COLLECT.json和不可变归档恢复回收。

独立确认只执行一次。未通过保留原始失败，不修任务、扩窗口或改阈值。执行评分可从原始数据离线复算，但不会再次运行模型：

```sh
.venv/bin/python -B v-conditional-cloud04/src/evaluate.py --run v-conditional-cloud04/execution/runs --output v-conditional-cloud04/execution/evaluation-recheck
```

结果：execution/evaluation/REPORT.md、DECISION.json、SCORED_RESULTS.json、MATCHING.json、FAMILY_BOOTSTRAP.json、PAIRED.json、audit.json/manifest.json。原始Token/Logits/KV/访问轨迹在execution/runs；传输和关机证据在execution。

主模块和V执行器读取数值状态；执行标签、gold和人工调度只给独立评分/MANUAL对照。TEXT_CODE为可见规则参考，其他三组Prompt没有规则正文。格式模板仍与开发相同，不把新合同措辞称为全新聊天模板或总体泛化证明。
