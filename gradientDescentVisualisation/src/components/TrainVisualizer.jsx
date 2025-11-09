import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

const TrainVisualizer = ({ config }) => {
  const canvasRef = useRef(null);
  const lossCanvasRef = useRef(null);
  const [training, setTraining] = useState(false);
  const [finalLoss, setFinalLoss] = useState(null);
  const [finalacc, setFinalacc] = useState(null);

  useEffect(() => {
    if (!config) return;

    const ctx = canvasRef.current.getContext("2d");
    const lossCtx = lossCanvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, 500, 500);
    lossCtx.clearRect(0, 0, 500, 150);

    const {
      dataset,
      lossFunction,
      epochs,
      lr,
      activationFunction,
      outputActivation,
      optimizer,
      layers,
      npl,
    } = config;
    const { X, y, numClasses, inputShape } = generateDataset(dataset);

    // ✅ Compute and store normalization stats once
    const mean = X[0].map((_, j) => X.reduce((s, v) => s + v[j], 0) / X.length);
    const std = X[0].map((_, j) =>
      Math.sqrt(X.reduce((s, v) => s + (v[j] - mean[j]) ** 2, 0) / X.length)
    );
    const X_scaled = X.map((v) =>
      v.map((val, j) => (val - mean[j]) / (std[j] || 1e-6))
    );
    const xs = tf.tensor2d(X_scaled, [X_scaled.length, inputShape]);

    let ys;
    if (numClasses === 1) ys = tf.tensor2d(y.map((v) => [v]));
    else if (numClasses === 2) ys = tf.tensor2d(y.map((v) => [v]));
    else ys = tf.oneHot(tf.tensor1d(y, "int32"), numClasses);

    const isRegression = dataset === "linear";

    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        units: npl[0] || 8,
        activation: activationFunction || "relu",
        inputShape: [inputShape],
      })
    );
    for (let i = 1; i < layers; i++) {
      model.add(
        tf.layers.dense({
          units: npl[i] || 8,
          activation: activationFunction || "relu",
        })
      );
    }
    model.add(
      tf.layers.dense({
        units: numClasses === 2 ? 1 : numClasses,
        activation:
          numClasses === 1
            ? "linear"
            : numClasses === 2
            ? "sigmoid"
            : "softmax",
      })
    );

    model.compile({
      optimizer: tf.train[optimizer || "adam"](lr || 0.01),
      loss: isRegression
        ? "meanSquaredError"
        : lossFunction || "categoricalCrossentropy",
      metrics: isRegression ? [] : ["accuracy"],
    });

    const losses = [];
    async function train() {
      setTraining(true);
      const history = await model.fit(xs, ys, {
        epochs,
        batchSize: 32,
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            losses.push(logs.loss);
            drawLoss(lossCtx, losses);
            drawAxes(ctx);

            if (dataset === "linear")
              drawRegressionLine(model, ctx, X, y, mean, std);
            else drawDecisionBoundary(model, ctx, X, y, mean, std);

            await tf.nextFrame();
          },
          onTrainEnd: () => setTraining(false),
        },
      });
      console.log(history.history);
      setFinalLoss(history.history.loss[history.history.loss.length - 1]);
      
        setFinalacc(history.history.acc[history.history.acc.length - 1]);
    }

    train();

    return () => {
      xs.dispose();
      ys.dispose();
      model.dispose();
    };
  }, [config]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-evenly",
      }}
    >
      <canvas
        ref={canvasRef}
        width="500"
        height="500"
        style={{ background: "black", borderRadius: 10 }}
      />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <label htmlFor="loss">Loss Graph</label>
        <canvas
          ref={lossCanvasRef}
          width="500"
          height="150"
          id="loss"
          style={{ background: "white", marginTop: 10, borderRadius: 10 }}
        />
        {training ? (
          <p style={{ color: "lime", marginTop: 5 }}>Training in progress...</p>
        ) : (
          <p style={{ color: "lime", marginTop: 5 }}>
            Final Loss: {finalLoss}
            <br />
            Final Acc: {finalacc}
          </p>
        )}
      </div>
    </div>
  );
};

// 🧭 Axes
function drawAxes(ctx) {
  ctx.strokeStyle = "gray";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, 250);
  ctx.lineTo(500, 250);
  ctx.moveTo(250, 0);
  ctx.lineTo(250, 500);
  ctx.stroke();
}

// ✅ Fixed Decision Boundary (uses same mean/std)
function drawDecisionBoundary(model, ctx, X, y, mean, std) {
  const width = 500,
    height = 500,
    gridSize = 150;
  const xVals = X.map((p) => p[0]);
  const yVals = X.map((p) => p[1]);
  const xMin = Math.min(...xVals) - 0.5,
    xMax = Math.max(...xVals) + 0.5;
  const yMin = Math.min(...yVals) - 0.5,
    yMax = Math.max(...yVals) + 0.5;
  const xStep = (xMax - xMin) / gridSize,
    yStep = (yMax - yMin) / gridSize;

  const grid = [];
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const gx = (xMin + j * xStep - mean[0]) / std[0];
      const gy = (yMin + i * yStep - mean[1]) / std[1];
      grid.push([gx, gy]);
    }
  }

  const preds = tf.tidy(() => {
    const input = tf.tensor2d(grid);
    const output = model.predict(input);
    if (output.shape[1] === 1) {
      const probs = output.dataSync();
      return Array.from(probs).map((p) => (p > 0.5 ? 1 : 0));
    } else {
      return output.argMax(1).dataSync();
    }
  });

  const palette = [
    [0, 255, 255],
    [255, 165, 0],
    [50, 205, 50],
    [255, 0, 255],
    [255, 255, 0],
  ];

  const imageData = ctx.createImageData(gridSize, gridSize);
  for (let i = 0; i < preds.length; i++) {
    const [r, g, b] = palette[preds[i] % palette.length];
    imageData.data[i * 4 + 0] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = 100;
  }

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = gridSize;
  tmpCanvas.height = gridSize;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.putImageData(imageData, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(tmpCanvas, 0, 0, width, height);
  drawAxes(ctx);

  for (let i = 0; i < X.length; i++) {
    const [x, yVal] = X[i];
    ctx.beginPath();
    ctx.arc(
      ((x - xMin) / (xMax - xMin)) * width,
      height - ((yVal - yMin) / (yMax - yMin)) * height,
      4,
      0,
      2 * Math.PI
    );
    const [r, g, b] = palette[y[i] % palette.length];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fill();
  }
}

// ✅ Regression visualization
function drawRegressionLine(model, ctx, X, y, mean, std) {
  ctx.clearRect(0, 0, 500, 500);
  drawAxes(ctx);
  const yMin = Math.min(...y),
    yMax = Math.max(...y);

  // Plot data points
  for (let i = 0; i < X.length; i++) {
    const xPixel = (X[i][0] + 1) * 250;
    const yPixel = (1 - (y[i] - yMin) / (yMax - yMin)) * 500;
    ctx.beginPath();
    ctx.arc(xPixel, yPixel, 3, 0, 2 * Math.PI);
    ctx.fillStyle = "cyan";
    ctx.fill();
  }

  const xVals = tf.linspace(-1, 1, 100);
  const preds = tf.tidy(() => {
    const scaled = xVals.sub(mean[0]).div(std[0]).reshape([100, 1]);
    return model.predict(scaled).dataSync();
  });

  const xArr = xVals.arraySync();
  ctx.beginPath();
  for (let i = 0; i < preds.length; i++) {
    const xPixel = (xArr[i] + 1) * 250;
    const yPixel = (1 - (preds[i] - yMin) / (yMax - yMin)) * 500;
    if (i === 0) ctx.moveTo(xPixel, yPixel);
    else ctx.lineTo(xPixel, yPixel);
  }
  ctx.strokeStyle = "orange";
  ctx.lineWidth = 2;
  ctx.stroke();
  xVals.dispose();
}

// ✅ Loss plot
function drawLoss(ctx, losses) {
  ctx.clearRect(0, 0, 500, 150);
  ctx.strokeStyle = "gray";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(30, 0);
  ctx.lineTo(30, 150);
  ctx.moveTo(30, 140);
  ctx.lineTo(500, 140);
  ctx.stroke();
  ctx.beginPath();
  for (let i = 0; i < losses.length; i++) {
    const x = 30 + (i / losses.length) * 450;
    const y = 140 - losses[i] * 100;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ✅ Dataset Generators
function generateDataset(type) {
  switch (type) {
    case "moons":
      return { ...makeMoons(500, 0.1), numClasses: 2, inputShape: 2 };
    case "circles":
      return { ...makeCircles(500, 0.1), numClasses: 2, inputShape: 2 };
    case "linear":
      return { ...makeLinear(200), numClasses: 1, inputShape: 1 };
    case "iris":
      return { ...makeIris(), numClasses: 3, inputShape: 2 };
    default:
      return { ...makeMoons(500, 0.1), numClasses: 2, inputShape: 2 };
  }
}

function makeMoons(n, noise = 0.1) {
  const X = [],
    y = [];
  for (let i = 0; i < n; i++) {
    const angle = Math.PI * Math.random();
    const label = Math.random() > 0.5 ? 1 : 0;
    const r = 1 + (Math.random() - 0.5) * noise;
    if (label === 0) X.push([Math.cos(angle) * r, Math.sin(angle) * r]);
    else X.push([1 - Math.cos(angle) * r, 1 - Math.sin(angle) * r - 0.5]);
    y.push(label);
  }
  return { X, y };
}

function makeCircles(n, noise = 0.1) {
  const X = [],
    y = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? 0 : 1;
    const radius = label === 0 ? 0.5 : 1;
    const angle = Math.random() * 2 * Math.PI;
    const r = radius + (Math.random() - 0.5) * noise;
    X.push([r * Math.cos(angle), r * Math.sin(angle)]);
    y.push(label);
  }
  return { X, y };
}

function makeLinear(n) {
  const X = [],
    y = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const noise = (Math.random() - 0.5) * 0.1;
    const target = 0.8 * x + 0.3 + noise;
    X.push([x]);
    y.push(target);
  }
  return { X, y };
}

function makeIris() {
  const X = [],
    y = [];
  for (let i = 0; i < 150; i++) {
    const cls = Math.floor(i / 50);
    const x1 = Math.random() * 2 + cls * 0.5;
    const x2 = Math.random() * 2 + (2 - cls) * 0.5;
    X.push([x1, x2]);
    y.push(cls);
  }
  return { X, y };
}

export default TrainVisualizer;
