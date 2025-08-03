import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from "recharts";
import {
  TrendingUp,
  Users,
  Clock,
  RefreshCw,
  BarChart3,
  PieChart as PieChartIcon,
  Activity,
} from "lucide-react";
import { StatusResponse } from "../types/api";
import apiService from "../services/api";

// Mock data for demonstration - in a real app, this would come from your API
const mockAccuracyData = [
  { epoch: 1, accuracy: 0.65, val_accuracy: 0.62, loss: 0.45, val_loss: 0.48 },
  { epoch: 2, accuracy: 0.72, val_accuracy: 0.68, loss: 0.38, val_loss: 0.42 },
  { epoch: 3, accuracy: 0.78, val_accuracy: 0.75, loss: 0.32, val_loss: 0.35 },
  { epoch: 4, accuracy: 0.83, val_accuracy: 0.8, loss: 0.28, val_loss: 0.31 },
  { epoch: 5, accuracy: 0.87, val_accuracy: 0.84, loss: 0.24, val_loss: 0.28 },
];

const mockClassDistribution = [
  { label: "NORMAL", count: 245, percentage: 52.1 },
  { label: "PNEUMONIA", count: 225, percentage: 47.9 },
];

const mockPredictionHistory = [
  { date: "2024-01-01", normal: 12, pneumonia: 8, total: 20 },
  { date: "2024-01-02", normal: 15, pneumonia: 5, total: 20 },
  { date: "2024-01-03", normal: 18, pneumonia: 7, total: 25 },
  { date: "2024-01-04", normal: 22, pneumonia: 13, total: 35 },
  { date: "2024-01-05", normal: 19, pneumonia: 11, total: 30 },
  { date: "2024-01-06", normal: 25, pneumonia: 15, total: 40 },
  { date: "2024-01-07", normal: 20, pneumonia: 10, total: 30 },
];

const mockTrainingLogs = [
  {
    id: "training_001",
    date: "2024-01-01",
    epochs: 5,
    final_accuracy: 0.87,
    duration: "12 min",
    status: "COMPLETED",
  },
  {
    id: "training_002",
    date: "2024-01-03",
    epochs: 8,
    final_accuracy: 0.91,
    duration: "18 min",
    status: "COMPLETED",
  },
  {
    id: "training_003",
    date: "2024-01-05",
    epochs: 10,
    final_accuracy: 0.89,
    duration: "25 min",
    status: "COMPLETED",
  },
];

const COLORS = ["#10B981", "#EF4444", "#3B82F6", "#F59E0B"];

const Visualizations: React.FC = () => {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const data = await apiService.getStatus();
      setStatus(data);
    } catch (err) {
      console.error("Failed to fetch status:", err);
    } finally {
      setLoading(false);
    }
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
          Analytics & Visualizations
        </h1>
        <p className="text-gray-600">
          Monitor model performance, usage patterns, and training progress
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">
                Total Predictions
              </p>
              <p className="text-2xl font-bold text-gray-900">
                {status?.total_predictions || 0}
              </p>
            </div>
            <Activity className="h-8 w-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">
                Model Accuracy
              </p>
              <p className="text-2xl font-bold text-gray-900">
                {status?.model_accuracy
                  ? `${(status.model_accuracy * 100).toFixed(1)}%`
                  : "N/A"}
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">
                Training Sessions
              </p>
              <p className="text-2xl font-bold text-gray-900">
                {mockTrainingLogs.length}
              </p>
            </div>
            <RefreshCw className="h-8 w-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Training Data</p>
              <p className="text-2xl font-bold text-gray-900">
                {mockClassDistribution.reduce(
                  (sum, item) => sum + item.count,
                  0
                )}
              </p>
            </div>
            <Users className="h-8 w-8 text-orange-500" />
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Accuracy Over Time */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center mb-4">
            <BarChart3 className="h-5 w-5 text-gray-500 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">
              Model Training Progress
            </h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockAccuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis domain={[0, 1]} />
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${(value * 100).toFixed(1)}%`,
                  name === "accuracy"
                    ? "Training Accuracy"
                    : name === "val_accuracy"
                    ? "Validation Accuracy"
                    : name === "loss"
                    ? "Training Loss"
                    : "Validation Loss",
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#10B981"
                strokeWidth={2}
                name="Training Accuracy"
              />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                stroke="#3B82F6"
                strokeWidth={2}
                name="Validation Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Class Distribution */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center mb-4">
            <PieChartIcon className="h-5 w-5 text-gray-500 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">
              Training Data Distribution
            </h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={mockClassDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ label, percentage }) =>
                  `${label}: ${percentage.toFixed(1)}%`
                }
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {mockClassDistribution.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Daily Predictions */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center mb-4">
            <Activity className="h-5 w-5 text-gray-500 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">
              Daily Prediction Volume
            </h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mockPredictionHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="normal" stackId="a" fill="#10B981" name="Normal" />
              <Bar
                dataKey="pneumonia"
                stackId="a"
                fill="#EF4444"
                name="Pneumonia"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Training Loss */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center mb-4">
            <TrendingUp className="h-5 w-5 text-gray-500 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">
              Training Loss Over Time
            </h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockAccuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip
                formatter={(value: number, name: string) => [
                  value.toFixed(3),
                  name === "loss" ? "Training Loss" : "Validation Loss",
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#EF4444"
                strokeWidth={2}
                name="Training Loss"
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="#F59E0B"
                strokeWidth={2}
                name="Validation Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Training History Table */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center mb-4">
          <Clock className="h-5 w-5 text-gray-500 mr-2" />
          <h2 className="text-xl font-semibold text-gray-900">
            Training History
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Training ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Epochs
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Final Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Duration
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {mockTrainingLogs.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {log.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {log.date}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {log.epochs}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {(log.final_accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {log.duration}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        log.status === "COMPLETED"
                          ? "bg-green-100 text-green-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {log.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Current Model Performance
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <h3 className="text-lg font-medium text-green-900">Accuracy</h3>
            <p className="text-3xl font-bold text-green-600">
              {status?.model_accuracy
                ? `${(status.model_accuracy * 100).toFixed(1)}%`
                : "N/A"}
            </p>
            <p className="text-sm text-green-700">
              Overall prediction accuracy
            </p>
          </div>

          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <h3 className="text-lg font-medium text-blue-900">
              Predictions Made
            </h3>
            <p className="text-3xl font-bold text-blue-600">
              {status?.total_predictions || 0}
            </p>
            <p className="text-sm text-blue-700">Total images analyzed</p>
          </div>

          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <h3 className="text-lg font-medium text-purple-900">
              Model Version
            </h3>
            <p className="text-xl font-bold text-purple-600">
              {status?.model_version || "N/A"}
            </p>
            <p className="text-sm text-purple-700">Current model iteration</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Visualizations;
