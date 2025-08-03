import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  Settings,
  AlertCircle,
  CheckCircle,
  Loader2,
  Tags,
  Play,
  X,
} from "lucide-react";
import {
  TrainingParameters,
  UploadResponse,
  RetrainResponse,
} from "../types/api";
import apiService from "../services/api";

interface FileWithLabel {
  file: File;
  label: "NORMAL" | "PNEUMONIA";
  preview: string;
}

const Retrain: React.FC = () => {
  const [filesWithLabels, setFilesWithLabels] = useState<FileWithLabel[]>([]);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [trainingResult, setTrainingResult] = useState<RetrainResponse | null>(
    null
  );
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const [trainingParams, setTrainingParams] = useState<TrainingParameters>({
    epochs: 5,
    learning_rate: 0.001,
    use_uploaded_data: true,
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file) => ({
      file,
      label: "NORMAL" as const,
      preview: URL.createObjectURL(file),
    }));

    setFilesWithLabels((prev) => [...prev, ...newFiles]);
    setUploadError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const updateFileLabel = (index: number, label: "NORMAL" | "PNEUMONIA") => {
    setFilesWithLabels((prev) =>
      prev.map((item, i) => (i === index ? { ...item, label } : item))
    );
  };

  const removeFile = (index: number) => {
    setFilesWithLabels((prev) => {
      const item = prev[index];
      URL.revokeObjectURL(item.preview);
      return prev.filter((_, i) => i !== index);
    });
  };

  const handleUploadImages = async () => {
    if (filesWithLabels.length === 0) return;

    setUploading(true);
    setUploadError(null);

    try {
      const files = filesWithLabels.map((item) => item.file);
      const labels = filesWithLabels.map((item) => item.label);

      const result = await apiService.uploadTrainingImages(files, labels);
      setUploadResult(result);

      // Clear uploaded files
      filesWithLabels.forEach((item) => URL.revokeObjectURL(item.preview));
      setFilesWithLabels([]);
    } catch (err: any) {
      setUploadError(
        err.response?.data?.detail || "Failed to upload training images"
      );
    } finally {
      setUploading(false);
    }
  };

  const handleStartTraining = async () => {
    setTraining(true);
    setTrainingError(null);

    try {
      const result = await apiService.retrainModel(trainingParams);
      setTrainingResult(result);
    } catch (err: any) {
      setTrainingError(
        err.response?.data?.detail || "Failed to start model training"
      );
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Model Retraining
        </h1>
        <p className="text-gray-600">
          Upload labeled training data and retrain the pneumonia detection model
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="xl:col-span-2 space-y-6">
          {/* File Upload */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Upload Training Images
            </h2>

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
                    Drop the images here
                  </p>
                ) : (
                  <div>
                    <p className="text-gray-600 font-medium mb-2">
                      Drop training images here, or click to select
                    </p>
                    <p className="text-sm text-gray-500">
                      Multiple files supported • JPEG, PNG up to 10MB each
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* File List */}
            {filesWithLabels.length > 0 && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Uploaded Files ({filesWithLabels.length})
                  </h3>
                  <button
                    onClick={handleUploadImages}
                    disabled={uploading}
                    className="bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {uploading ? (
                      <div className="flex items-center">
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        Uploading...
                      </div>
                    ) : (
                      <div className="flex items-center">
                        <Upload className="h-4 w-4 mr-2" />
                        Upload {filesWithLabels.length} Files
                      </div>
                    )}
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                  {filesWithLabels.map((item, index) => (
                    <div
                      key={index}
                      className="border border-gray-200 rounded-lg p-4"
                    >
                      <div className="flex items-start space-x-3">
                        <img
                          src={item.preview}
                          alt={`Preview ${index}`}
                          className="w-16 h-16 object-cover rounded-lg"
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {item.file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {(item.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                          <div className="mt-2 flex items-center space-x-2">
                            <Tags className="h-4 w-4 text-gray-400" />
                            <select
                              value={item.label}
                              onChange={(e) =>
                                updateFileLabel(
                                  index,
                                  e.target.value as "NORMAL" | "PNEUMONIA"
                                )
                              }
                              className="text-xs border border-gray-300 rounded px-2 py-1"
                            >
                              <option value="NORMAL">NORMAL</option>
                              <option value="PNEUMONIA">PNEUMONIA</option>
                            </select>
                          </div>
                        </div>
                        <button
                          onClick={() => removeFile(index)}
                          className="text-red-500 hover:text-red-700"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Result */}
            {uploadResult && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                  <div>
                    <p className="text-green-700 font-medium">
                      {uploadResult.message}
                    </p>
                    <p className="text-green-600 text-sm">
                      Ready for training:{" "}
                      {uploadResult.ready_for_training ? "Yes" : "No"}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Upload Error */}
            {uploadError && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                  <span className="text-red-700">{uploadError}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Training Configuration */}
        <div className="space-y-6">
          {/* Training Parameters */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center mb-4">
              <Settings className="h-5 w-5 text-gray-500 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">
                Training Configuration
              </h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Epochs
                </label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={trainingParams.epochs}
                  onChange={(e) =>
                    setTrainingParams((prev) => ({
                      ...prev,
                      epochs: parseInt(e.target.value) || 5,
                    }))
                  }
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  More epochs = better accuracy but longer training time
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.0001"
                  min="0.0001"
                  max="0.1"
                  value={trainingParams.learning_rate}
                  onChange={(e) =>
                    setTrainingParams((prev) => ({
                      ...prev,
                      learning_rate: parseFloat(e.target.value) || 0.001,
                    }))
                  }
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Lower values = more stable training
                </p>
              </div>

              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={trainingParams.use_uploaded_data}
                    onChange={(e) =>
                      setTrainingParams((prev) => ({
                        ...prev,
                        use_uploaded_data: e.target.checked,
                      }))
                    }
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">
                    Use uploaded training data
                  </span>
                </label>
                <p className="text-xs text-gray-500 mt-1">
                  Uncheck to use original dataset (if available)
                </p>
              </div>
            </div>
          </div>

          {/* Start Training */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Start Training
            </h3>

            <div className="mb-4 text-sm text-gray-600">
              <p className="mb-2">Estimated training time:</p>
              <p className="font-medium">{trainingParams.epochs * 2} minutes</p>
            </div>

            <button
              onClick={handleStartTraining}
              disabled={training}
              className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {training ? (
                <div className="flex items-center justify-center">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" />
                  Starting Training...
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <Play className="h-5 w-5 mr-2" />
                  Start Model Training
                </div>
              )}
            </button>

            {/* Training Result */}
            {trainingResult && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center mb-2">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
                  <span className="text-green-700 font-medium">
                    Training Started
                  </span>
                </div>
                <div className="text-sm text-green-600">
                  <p>Training ID: {trainingResult.training_id}</p>
                  <p>Status: {trainingResult.status}</p>
                  <p>
                    Estimated completion:{" "}
                    {trainingResult.estimated_completion_minutes} minutes
                  </p>
                </div>
              </div>
            )}

            {/* Training Error */}
            {trainingError && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                  <span className="text-red-700">{trainingError}</span>
                </div>
              </div>
            )}
          </div>

          {/* Training Tips */}
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 mb-2">Training Tips</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Upload diverse X-ray images for better accuracy</li>
              <li>• Ensure equal distribution of NORMAL and PNEUMONIA cases</li>
              <li>• Higher epochs improve accuracy but take longer</li>
              <li>• Monitor the dashboard for training progress</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Retrain;
