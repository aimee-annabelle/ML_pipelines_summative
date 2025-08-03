import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import Retrain from "./pages/Retrain";
import Visualizations from "./pages/Visualizations";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/retrain" element={<Retrain />} />
            <Route path="/visualizations" element={<Visualizations />} />
          </Routes>
        </Layout>
      </div>
    </Router>
  );
}

export default App;
