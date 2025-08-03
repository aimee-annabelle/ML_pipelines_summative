import React, { useState, useEffect } from "react";
import {
  Activity,
  AlertCircle,
  Clock,
  Brain,
  TrendingUp,
  Database,
  Cpu,
  RefreshCw,
} from "lucide-react";
import { StatusResponse } from "../types/api";
import apiService from "../services/api";

const Dashboard: React.FC = () => {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const data = await apiService.getStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch system status");
      console.error("Status fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const getStatusColor = (isHealthy: boolean) => {
    return isHealthy ? "text-green-500" : "text-red-500";
  };

  const getStatusBg = (isHealthy: boolean) => {
    return isHealthy
      ? "bg-green-50 border-green-200"
      : "bg-red-50 border-red-200";
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          X-Ray Classification Dashboard
        </h1>
        <p className="text-gray-600">
          Monitor your pneumonia detection system status and performance
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Status Cards */}
      {status && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* API Status */}
          <div
            className={`rounded-lg shadow-sm p-6 border ${getStatusBg(
              status.api_status === "healthy"
            )}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">API Status</p>
                <p className="text-2xl font-bold text-gray-900">
                  {status.api_status}
                </p>
              </div>
              <Activity
                className={`h-8 w-8 ${getStatusColor(
                  status.api_status === "healthy"
                )}`}
              />
            </div>
          </div>

          {/* Model Status */}
          <div
            className={`rounded-lg shadow-sm p-6 border ${getStatusBg(
              status.model_loaded
            )}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Model Status
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {status.model_loaded ? "Loaded" : "Not Loaded"}
                </p>
              </div>
              <Brain
                className={`h-8 w-8 ${getStatusColor(status.model_loaded)}`}
              />
            </div>
          </div>

          {/* Total Predictions */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Total Predictions
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {status.total_predictions}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-blue-500" />
            </div>
          </div>

          {/* System Uptime */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Uptime</p>
                <p className="text-2xl font-bold text-gray-900">
                  {formatUptime(status.uptime_seconds)}
                </p>
              </div>
              <Clock className="h-8 w-8 text-purple-500" />
            </div>
          </div>
        </div>
      )}

      {/* Detailed Information */}
      {status && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Information */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Model Information
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Version:</span>
                <span className="font-medium">{status.model_version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Accuracy:</span>
                <span className="font-medium">
                  {status.model_accuracy
                    ? `${(status.model_accuracy * 100).toFixed(2)}%`
                    : "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Training Status:</span>
                <span
                  className={`font-medium px-2 py-1 rounded-full text-xs ${
                    status.training_status === "IDLE"
                      ? "bg-gray-100 text-gray-800"
                      : status.training_status === "TRAINING"
                      ? "bg-yellow-100 text-yellow-800"
                      : status.training_status === "COMPLETED"
                      ? "bg-green-100 text-green-800"
                      : "bg-red-100 text-red-800"
                  }`}
                >
                  {status.training_status}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Last Prediction:</span>
                <span className="font-medium">
                  {status.last_prediction_time
                    ? new Date(status.last_prediction_time).toLocaleString()
                    : "None"}
                </span>
              </div>
            </div>
          </div>

          {/* System Resources */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              System Resources
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Memory Usage:</span>
                <div className="flex items-center">
                  <Database className="h-4 w-4 text-blue-500 mr-1" />
                  <span className="font-medium">
                    {status.memory_usage_mb.toFixed(2)} MB
                  </span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">GPU Available:</span>
                <div className="flex items-center">
                  <Cpu
                    className={`h-4 w-4 mr-1 ${
                      status.gpu_available ? "text-green-500" : "text-gray-400"
                    }`}
                  />
                  <span className="font-medium">
                    {status.gpu_available ? "Yes" : "No"}
                  </span>
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Model Path:</span>
                <span
                  className="font-medium text-sm truncate max-w-xs"
                  title={status.model_path}
                >
                  {status.model_path}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => (window.location.href = "/predict")}
            className="flex items-center justify-center p-4 bg-primary-50 border border-primary-200 rounded-lg hover:bg-primary-100 transition-colors"
          >
            <div className="text-center">
              <div className="bg-primary-500 rounded-full p-2 mx-auto mb-2">
                <Activity className="h-5 w-5 text-white" />
              </div>
              <h3 className="font-medium text-gray-900">Make Prediction</h3>
              <p className="text-sm text-gray-600">Upload X-ray for analysis</p>
            </div>
          </button>

          <button
            onClick={() => (window.location.href = "/retrain")}
            className="flex items-center justify-center p-4 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors"
          >
            <div className="text-center">
              <div className="bg-green-500 rounded-full p-2 mx-auto mb-2">
                <RefreshCw className="h-5 w-5 text-white" />
              </div>
              <h3 className="font-medium text-gray-900">Retrain Model</h3>
              <p className="text-sm text-gray-600">Improve model accuracy</p>
            </div>
          </button>

          <button
            onClick={() => (window.location.href = "/visualizations")}
            className="flex items-center justify-center p-4 bg-purple-50 border border-purple-200 rounded-lg hover:bg-purple-100 transition-colors"
          >
            <div className="text-center">
              <div className="bg-purple-500 rounded-full p-2 mx-auto mb-2">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              <h3 className="font-medium text-gray-900">View Analytics</h3>
              <p className="text-sm text-gray-600">Performance insights</p>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
