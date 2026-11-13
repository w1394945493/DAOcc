<!-- 检查端口是否释放： -->
ss -lntpe '( sport = :17891 )'
State             Recv-Q            Send-Q                       Local Address:Port                          Peer Address:Port            Process            
LISTEN            0                 128                              127.0.0.1:17891                              0.0.0.0:*                uid:1059 ino:97866462 sk:300f cgroup:/user.slice/user-1059.slice/session-4986.scope <->
<!-- 找到对应的 SSH 进程 -->
loginctl session-status 4986 --no-pager
<!-- 打印内容： -->
4986 - wangyushen (1059)
  Since: Fri 2026-11-13 17:35:12 CST; 2h 20min ago
  State: active
 Leader: 1363101 (sshd)
 Remote: 10.62.203.194
Service: sshd
   Type: tty
  Class: user
   Idle: no
   Unit: session-4986.scope
         ├─1363101 "sshd: wangyushen [priv]"
         └─1363151 "sshd: wangyushen"

<!-- 3458675  sshd: wangyushen [priv]  # root 权限进程
3458744  sshd: wangyushen         # 你的用户进程 -->

<!-- 之前残留、不再需要的 SSH 连接，直接执行： -->
kill 1363151

<!-- git -->
git config --local http.proxy http://127.0.0.1:17891
git push