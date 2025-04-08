<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bottle Detection Dashboard</title>
  <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
  <style>
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #f2f2f2;
      margin: 0;
      padding: 0;
    }
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
    h1 {
      color: #333;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 20px;
      margin-top: 20px;
    }
    .card {
      background: #fff;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      min-width: 200px;
      text-align: center;
    }
    .card h2 {
      margin: 0;
      font-size: 32px;
      color: #007bff;
    }
    .card span {
      display: block;
      margin-top: 5px;
      font-size: 16px;
      color: #666;
    }
    img {
      margin-top: 20px;
      max-height: 300px;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Bottle Detection Dashboard</h1>
    <img src="/static/reference.jpg" alt="Reference Object">

    <div class="stats">
      <div class="card">
        <h2 id="count">0</h2>
        <span>Total Bottles Detected</span>
      </div>
      <div class="card">
        <h2 id="accuracy">0%</h2>
        <span>Average Confidence</span>
      </div>
      <div class="card">
        <h2 id="speed">0 ms</h2>
        <span>Avg Processing Time</span>
      </div>
      <div class="card">
        <h2 id="fps">0</h2>
        <span>Frames Per Second</span>
      </div>
      <div class="card">
        <h2 id="power">0 mW</h2>
        <span>Power Usage</span>
      </div>
      <div class="card">
        <h2 id="energy">0 mWh</h2>
        <span>Total Energy Consumed</span>
      </div>
    </div>
  </div>

  <script>
    const socket = io();
    socket.on("bottle_stats", data => {
      document.getElementById("count").textContent = data.count;
      document.getElementById("accuracy").textContent = data.accuracy + "%";
      document.getElementById("speed").textContent = data.speed + " ms";
      document.getElementById("fps").textContent = data.fps;
      document.getElementById("power").textContent = data.power + " mW";
      document.getElementById("energy").textContent = data.energy + " mWh";
    });
  </script>
</body>
</html>