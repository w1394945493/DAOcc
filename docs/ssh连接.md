<!-- 检查端口是否释放： -->
ss -lntpe '( sport = :17891 )'
<!-- 找到对应的 SSH 进程 -->
loginctl session-status 7268 --no-pager
<!-- 打印内容： -->
7268 - wangyushen (1059)
  Since: Fri 2026-11-06 09:56:36 CST; 1h 20min ago
  State: active
 Leader: 3458675 (sshd)
 Remote: 10.62.212.205
Service: sshd
   Type: tty
  Class: user
   Idle: no
   Unit: session-7268.scope
         ├─3458675 "sshd: wangyushen [priv]"
         └─3458744 "sshd: wangyushen"

<!-- 3458675  sshd: wangyushen [priv]  # root 权限进程
3458744  sshd: wangyushen         # 你的用户进程 -->

<!-- 之前残留、不再需要的 SSH 连接，直接执行： -->
kill 3458744