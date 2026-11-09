# Cloud07 三类失败的有界诊断

本轮是已见开发任务机制定位，非独立确认。模块、阈值、Bank、评分器保持冻结。

- A：四族 N 的原接口文字/明确调用者文字。
- B：两族 E/N 的原USAGE/明确调用者，原判断器与原文字规则基线；另8个固定前缀只读。
- C：字符计数的自然OFF/ON，及干预历史×固定构造器四格诊断。

规模：24条自然生成、4条强制构造器诊断续写、8次辅助读出。

协议见 PROTOCOL.md，执行细节见 EXECUTION.md。运行后结果在 execution；历史 Cloud06及plan保持只读。

离线评分复现（输出必须是新目录）：

```sh
.venv/bin/python -B v-conditional-cloud07/src/evaluate.py --run v-conditional-cloud07/execution/runs --output v-conditional-cloud07/reproduction-new
```
