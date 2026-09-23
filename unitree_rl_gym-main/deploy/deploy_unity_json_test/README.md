# Unity TCP JSON 收发测试

这是与 G1 策略服务完全独立的网络测试。它不会加载 `.pt` 模型，也不会控制机器人。

目的只有一个：确认 Unity 客户端能够通过 TCP 从 Python 服务器读取 JSON 响应。

```text
Unity 客户端                         Python 服务端
UnityJsonTcpTestClient.cs          json_test_server.py
        |                                      |
        |--------- TCP <PYTHON_PC_IP>:9003 ---->|
        |<-------- welcome JSON --------------|
        |--------- ping JSON ---------------->|
        |<-------- pong JSON -----------------|
```

## 1. Python 启动测试服务器

在 Python 电脑执行：

```bash
cd /path/to/unitree/unitree_rl_gym-main
python deploy/deploy_unity_json_test/json_test_server.py \
  --host 0.0.0.0 \
  --port 9003
```

不要使用 `127.0.0.1`，因为 Unity 在另一台电脑。Python 终端应显示：

```text
Unity JSON test server listening on 0.0.0.0:9003
```

如果 Ubuntu 防火墙启用：

```bash
sudo ufw allow 9003/tcp
```

## 2. 不使用 Unity 的本机验证

先在 Python 电脑另开终端：

```bash
cd /path/to/unitree/unitree_rl_gym-main
python deploy/deploy_unity_json_test/python_test_client.py
```

预期输出：

```text
RX welcome: {"type":"welcome",...}
TX ping:   {"type": "ping", "seq": 1}
RX pong:   {"type":"pong","seq":1,...}
```

## 3. Unity 客户端测试

1. 将 `UnityJsonTcpTestClient.cs` 拷贝到 Unity 项目的 `Assets/Scripts/`。
2. 在 Unity 场景创建空物体，例如 `PythonJsonTestClient`。
3. 给该物体挂载 `UnityJsonTcpTestClient` 组件。
4. 在 Inspector 设置：

```text
Server Ip:   <PYTHON_PC_IP>
Server Port: 9003
```

5. 打开 Unity Console：`Window -> General -> Console`。
6. 取消勾选 `Collapse`，然后进入 Play Mode。

预期 Console 输出：

```text
[Unity] Connected to <PYTHON_PC_IP>:9003
[Python TX welcome] {"type":"welcome",...}
[Unity TX] {"type":"ping","seq":0}
[Python TX response] {"type":"pong","seq":0,...}
```

在 Play Mode 中按 `P`，会再发送一个 ping，并在 Unity Console 看到新的 pong。

## 4. 这个测试证明什么

成功后可证明：

- Unity 可以连接 Python 电脑的 IP 和端口。
- Unity 可以读取 Python 主动发送的 `welcome` JSON。
- Unity 可以发送 JSON。
- Unity 可以通过同一条 TCP 连接读取 Python 的 `pong` JSON。
- `StreamReader.ReadLine()`、`StreamWriter.WriteLine()` 的换行分帧正确。

它不证明 G1 物理、坐标系、关节顺序或策略推理已经正确。网络测试成功后，再接回 `deploy/deploy_unity/policy_server.py`。

## 5. 常见问题

### Unity 无法连接

确认 Unity 填写的是 Python 电脑的局域网地址和端口 `9003`，不是自己的 IP；只有两端运行在同一台电脑时才使用 `127.0.0.1`。

在 Unity 的 Windows 电脑 PowerShell 测试：

```powershell
Test-NetConnection <PYTHON_PC_IP> -Port 9003
```

### Python 有客户端连接但 Unity 没有 welcome

确认 `UnityJsonTcpTestClient.cs` 已挂到场景物体，且 Unity Console 的 `Log` 筛选开启。脚本会在后台线程读取 TCP 数据，在主线程 `Update()` 中调用 `Debug.Log`。

### Unity 能看到 welcome，但看不到 pong

确认 Python 终端同时出现：

```text
RX ... {"type":"ping","seq":0}
TX ... {"type":"pong","seq":0,...}
```

如果没有 RX，Unity 端没有把 ping 写入连接；如果有 TX 而 Unity 没看到 pong，检查 Unity 是否只有一个线程调用 `ReadLine()`。
