// Attach this component to an empty GameObject in a Unity scene.
// It is intentionally separate from the G1 policy client.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UnityJsonTcpTestClient : MonoBehaviour
{
    [Header("Python test-server endpoint")]
    public string serverIp = "127.0.0.1";
    public int serverPort = 9003;

    private readonly ConcurrentQueue<string> incomingLogs = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<string> outgoingJson = new ConcurrentQueue<string>();
    private Thread networkThread;
    private volatile bool isRunning;
    private int nextSequence;

    [Serializable]
    private class PingMessage
    {
        public string type = "ping";
        public int seq;
    }

    void Start()
    {
        isRunning = true;
        networkThread = new Thread(NetworkLoop) { IsBackground = true };
        networkThread.Start();
    }

    void Update()
    {
        while (incomingLogs.TryDequeue(out string message))
        {
            // Debug.Log must run on Unity's main thread.
            Debug.Log(message);
        }

        // Press P in Play mode to send another ping.
        if (Input.GetKeyDown(KeyCode.P))
        {
            SendPing();
        }
    }

    public void SendPing()
    {
        PingMessage ping = new PingMessage { seq = nextSequence++ };
        string json = JsonUtility.ToJson(ping);
        outgoingJson.Enqueue(json);
        incomingLogs.Enqueue("[Unity TX] " + json);
    }

    private void NetworkLoop()
    {
        try
        {
            using (TcpClient client = new TcpClient())
            {
                client.NoDelay = true;
                client.Connect(serverIp, serverPort);
                using (NetworkStream stream = client.GetStream())
                using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
                using (StreamWriter writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true })
                {
                    incomingLogs.Enqueue("[Unity] Connected to " + serverIp + ":" + serverPort);

                    // The test server sends this JSON immediately after connecting.
                    string welcome = reader.ReadLine();
                    incomingLogs.Enqueue("[Python TX welcome] " + welcome);

                    SendPing();
                    while (isRunning)
                    {
                        if (!outgoingJson.TryDequeue(out string requestJson))
                        {
                            Thread.Sleep(10);
                            continue;
                        }

                        writer.WriteLine(requestJson);
                        string responseJson = reader.ReadLine();
                        if (responseJson == null)
                        {
                            throw new IOException("Python server closed the TCP connection");
                        }
                        incomingLogs.Enqueue("[Python TX response] " + responseJson);
                    }
                }
            }
        }
        catch (Exception exception)
        {
            incomingLogs.Enqueue("[Unity TCP error] " + exception.Message);
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        if (networkThread != null && networkThread.IsAlive)
        {
            networkThread.Join(200);
        }
    }
}
