import { useState } from "react";
import "./App.css";
import Heading from "./components/heading.jsx";
import Sidebar from "./components/sidebar.jsx";
import TrainVisualizer from "./components/TrainVisualizer.jsx";

function App() {
  const [modelConfig, setModelConfig] = useState(null);

  const handleStart = (config) => {
    console.log("🧠 Model Config:", config);
    setModelConfig(config); // Send config to TrainVisualizer
  };

  return (
    <div className="main-container">
      <Heading />
      <div className="inner-class">
        <Sidebar onStart={handleStart} />
        {modelConfig ? (
          <TrainVisualizer config={modelConfig} />
        ) : (
          <div style={{ margin: "auto", color: "gray" }}>
            <h3>Waiting for configuration...</h3>
            <p>Set parameters in sidebar and click “Start Training”</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
