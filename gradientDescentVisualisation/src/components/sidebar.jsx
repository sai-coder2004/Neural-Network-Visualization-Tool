import React, { useState } from "react";
import "./sidebar.css";

const Sidebar = ({ onStart }) => {
  const [dataset, setDataSet] = useState("");
  const [lossFunction, setLossFunction] = useState("");
  const [epochs, setEpochs] = useState(1);
  const [lr, setLr] = useState(0.01);
  const [activationFunction, setActivationFunction] = useState("");
  const [layers, setLayers] = useState(0);
  const [outputActivation, setOutputActivation] = useState("");
  const [npl, setNpl] = useState([]);
  const [optimizer, setOptimizer] = useState("");

  const handleNodeChange = (index, value) => {
    const updated = [...npl];
    updated[index] = Math.max(0, parseInt(value, 10) || 0);
    setNpl(updated);
  };

  const handleStart = () => {
    const config = {
      dataset,
      lossFunction,
      epochs,
      lr,
      activationFunction,
      outputActivation,
      optimizer,
      layers,
      npl,
    };
    if(config.dataset===""||config.lossFunction===""||config.epochs<=0||config.lr<=0||config.activationFunction===""||config.outputActivation===""
        ||config.optimizer===""||config.layers<=0||config.npl.find(v => v === 0)
    ){
        alert("select valid patrameters");
    }
    else onStart(config);
  };

  const handleLayerChange = (e) => {
    let n = parseInt(e.target.value, 10) || 0;
    if (n < 0) n = 0;
    if (n > 10) n = 10;
    setLayers(n);
    setNpl(Array(n).fill(0));
  };

  const handleEpochChange = (e) => {
    const val = parseInt(e.target.value, 10);
    setEpochs(val >= 1 ? val : "");
  };

  const handleLrChange = (e) => {
    const val = parseFloat(e.target.value);
    setLr(val >= 0 ? val : 0);
  };

  return (
    <div className="side-bar">
      {/* Dataset Dropdown */}
      <div>
        <label htmlFor="algo-dropdown">Select Dataset:</label>
        <select
          id="algo-dropdown"
          value={dataset}
          onChange={(e) => setDataSet(e.target.value)}
        >
          <option value="">--Select--</option>
          <option value="iris">Iris Dataset (Classification)</option>
          <option value="moons">Moons Dataset (Binary Classification)</option>
          <option value="circles">Circles Dataset (Binary Classification)</option>
          <option value="linear">Linear Regression Dataset (Regression Task)</option>
        </select>
      </div>

      {/* Loss Function */}
      <div>
        <label htmlFor="loss-dropdown">Loss Function:</label>
        <select
          id="loss-dropdown"
          value={lossFunction}
          onChange={(e) => setLossFunction(e.target.value)}
        >
          <option value="">--Select--</option>
          <option value="meanSquaredError">Mean Squared Error (MSE)</option>
          <option value="meanAbsoluteError">Mean Absolute Error (MAE)</option>
          <option value="huberLoss">Huber Loss</option>
          <option value="binaryCrossentropy">Binary Cross-Entropy (Log Loss)</option>
          <option value="categoricalCrossentropy">Categorical Cross-Entropy</option>
          <option value="hinge">Hinge Loss</option>
        </select>
      </div>

      {/* Epochs */}
      <div>
        <label htmlFor="epochs">No. of Epochs:</label>
        <input
          type="number"
          min="1"
          id="epochs"
          value={epochs}
          onChange={handleEpochChange}
        />
      </div>

      {/* Learning Rate */}
      <div>
        <label htmlFor="lr">Learning Rate:</label>
        <input
          type="number"
          min="0"
          step="0.0001"
          id="lr"
          value={lr}
          onChange={handleLrChange}
        />
      </div>

      {/* Hidden Layer Activation Function */}
      <div>
        <label htmlFor="activation-dropdown">Hidden Layer Activation Function:</label>
        <select
          id="activation-dropdown"
          value={activationFunction}
          onChange={(e) => setActivationFunction(e.target.value)}
        >
          <option value="">--Select--</option>
          <option value="relu">ReLU</option>
          <option value="sigmoid">Sigmoid</option>
          <option value="tanh">Tanh</option>
          <option value="leakyRelu">Leaky ReLU</option>
          <option value="linear">Linear</option>
        </select>
      </div>

      {/* Output Layer Activation */}
      <div>
        <label htmlFor="output-activation-dropdown">Output Layer Activation:</label>
        <select
          name="output-activation"
          id="output-activation-dropdown"
          value={outputActivation}
          onChange={(e) => setOutputActivation(e.target.value)}
        >
          <option value="">--Select--</option>
          <option value="sigmoid">Sigmoid (Binary)</option>
          <option value="softmax">Softmax (Multi-class)</option>
          <option value="linear">Linear (Regression)</option>
        </select>
      </div>

      {/* Optimizer */}
      <div>
        <label htmlFor="optimizer-dropdown">Optimizer:</label>
        <select
          name="optimizer"
          id="optimizer-dropdown"
          value={optimizer}
          onChange={(e) => setOptimizer(e.target.value)}
        >
          <option value="">--Select--</option>
          <option value="sgd">SGD (Stochastic Gradient Descent)</option>
          <option value="adam">Adam</option>
          <option value="rmsprop">RMSProp</option>
          <option value="adagrad">AdaGrad</option>
          <option value="adadelta">AdaDelta</option>
        </select>
      </div>

      {/* Layers */}
      <div>
        <label htmlFor="layers">No. of Layers (1–10):</label>
        <input
          type="number"
          min="1"
          max="10"
          id="layers"
          value={layers}
          onChange={handleLayerChange}
        />
      </div>

      {/* Nodes per Layer */}
      <div>
        {Array.from({ length: layers }).map((_, i) => (
          <div key={i}>
            <label>Nodes in Layer {i + 1}:</label>
            <input
              type="number"
              min="0"
              value={npl[i] || ""}
              onChange={(e) => handleNodeChange(i, e.target.value)}
            />
          </div>
        ))}
      </div>

      <button onClick={handleStart}>Start Training</button>
    </div>
  );
};

export default Sidebar;
