import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  Image as ImageIcon,
  AlertCircle,
  CheckCircle,
  Loader2,
  Clock,
  Brain,
} from "lucide-react";
import { PredictionResponse } from "../types/api";
import apiService from "../services/api";

const Predict: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);

      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);

      // Reset previous results
      setPrediction(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const result = await apiService.predictPneumonia(selectedFile);
      setPrediction(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to analyze X-ray image");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-50 border-green-200";
    if (confidence >= 0.6) return "bg-yellow-50 border-yellow-200";
    return "bg-red-50 border-red-200";
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          X-Ray Pneumonia Detection
        </h1>
        <p className="text-gray-600">
          Upload a chest X-ray image to get an AI-powered pneumonia diagnosis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Upload X-Ray Image
          </h2>

          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? "border-primary-400 bg-primary-50"
                : "border-gray-300 hover:border-primary-400 hover:bg-gray-50"
            }`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center">
              <Upload className="h-12 w-12 text-gray-400 mb-4" />
              {isDragActive ? (
                <p className="text-primary-600 font-medium">
                  Drop the X-ray image here
                </p>
              ) : (
                <div>
                  <p className="text-gray-600 font-medium mb-2">
                    Drop an X-ray image here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports JPEG, PNG up to 10MB
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Selected File Info */}
          {selectedFile && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <ImageIcon className="h-5 w-5 text-gray-500 mr-2" />
                  <div>
                    <p className="font-medium text-gray-900">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleReset}
                  className="text-red-600 hover:text-red-800 text-sm font-medium"
                >
                  Remove
                </button>
              </div>
            </div>
          )}

          {/* Image Preview */}
          {previewUrl && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-gray-900 mb-2">
                Preview
              </h3>
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <img
                  src={previewUrl}
                  alt="X-ray preview"
                  className="w-full h-64 object-cover"
                />
              </div>
            </div>
          )}

          {/* Predict Button */}
          <div className="mt-6">
            <button
              onClick={handlePredict}
              disabled={!selectedFile || loading}
              className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                !selectedFile || loading
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-primary-600 text-white hover:bg-primary-700"
              }`}
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" />
                  Analyzing...
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <Brain className="h-5 w-5 mr-2" />
                  Analyze X-Ray
                </div>
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Analysis Results
          </h2>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-red-700">{error}</span>
              </div>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <Loader2 className="h-12 w-12 animate-spin text-primary-500 mx-auto mb-4" />
                <p className="text-gray-600">Analyzing X-ray image...</p>
                <p className="text-sm text-gray-500 mt-2">
                  This may take a few seconds
                </p>
              </div>
            </div>
          )}

          {/* Prediction Results */}
          {prediction && (
            <div className="space-y-4">
              {/* Main Result */}
              <div
                className={`p-6 rounded-lg border-2 ${getConfidenceBg(
                  prediction.confidence
                )}`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    {prediction.prediction === "NORMAL" ? (
                      <CheckCircle className="h-8 w-8 text-green-500 mr-3" />
                    ) : (
                      <AlertCircle className="h-8 w-8 text-red-500 mr-3" />
                    )}
                    <div>
                      <h3 className="text-2xl font-bold text-gray-900">
                        {prediction.prediction}
                      </h3>
                      <p className="text-gray-600">
                        {prediction.prediction === "NORMAL"
                          ? "No pneumonia detected"
                          : "Pneumonia detected"}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">Confidence</p>
                    <p
                      className={`text-2xl font-bold ${getConfidenceColor(
                        prediction.confidence
                      )}`}
                    >
                      {(prediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Medical Advice */}
                <div className="bg-white bg-opacity-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700">
                    <strong>Medical Advice:</strong> {prediction.medical_advice}
                  </p>
                </div>
              </div>

              {/* Detailed Probabilities */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3">
                  Detailed Probabilities
                </h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Normal:</span>
                    <div className="flex items-center">
                      <div className="w-32 bg-gray-200 rounded-full h-2 mr-3">
                        <div
                          className="bg-green-500 h-2 rounded-full"
                          style={{
                            width: `${prediction.probability_normal * 100}%`,
                          }}
                        ></div>
                      </div>
                      <span className="font-medium text-gray-900 w-12">
                        {(prediction.probability_normal * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Pneumonia:</span>
                    <div className="flex items-center">
                      <div className="w-32 bg-gray-200 rounded-full h-2 mr-3">
                        <div
                          className="bg-red-500 h-2 rounded-full"
                          style={{
                            width: `${prediction.probability_pneumonia * 100}%`,
                          }}
                        ></div>
                      </div>
                      <span className="font-medium text-gray-900 w-12">
                        {(prediction.probability_pneumonia * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="flex items-center text-gray-600 mb-1">
                    <Clock className="h-4 w-4 mr-1" />
                    Processing Time
                  </div>
                  <p className="font-medium text-gray-900">
                    {prediction.processing_time_seconds.toFixed(2)}s
                  </p>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="flex items-center text-gray-600 mb-1">
                    <Brain className="h-4 w-4 mr-1" />
                    Model Version
                  </div>
                  <p className="font-medium text-gray-900">
                    {prediction.model_version}
                  </p>
                </div>
              </div>

              {/* Timestamp */}
              <div className="text-xs text-gray-500 text-center">
                Analysis completed on{" "}
                {new Date(prediction.timestamp).toLocaleString()}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!prediction && !loading && !error && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <ImageIcon className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">
                  Upload an X-ray image to see analysis results
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Predict;
