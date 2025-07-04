import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { 
  Upload, 
  Brain, 
  Zap, 
  CheckCircle, 
  AlertCircle, 
  Info, 
  Activity,
  Target,
  Clock,
  TrendingUp,
  BarChart3,
  Eye,
  Sparkles,
  Shield,
  Gauge
} from 'lucide-react';
import { apiClient } from '@/lib/api';
import { processPredictionData, ProcessedPredictionData } from '@/lib/data-processor';
import { EnsemblePredictionResult } from '@/types/api';
import { MODEL_CONFIG } from '@/constants';
import { motion, AnimatePresence } from 'framer-motion';

interface PredictionStep {
  id: string;
  label: string;
  icon: React.ElementType;
  status: 'pending' | 'processing' | 'completed' | 'error';
  result?: string;
  confidence?: number;
  processingTime?: number;
}

export default function AIPredictionPipeline() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedData, setProcessedData] = useState<ProcessedPredictionData | null>(null);
  const [rawResult, setRawResult] = useState<EnsemblePredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Helper function for model names
  const getStandardizedModelName = (modelName: string): string => {
    if (modelName.toLowerCase().includes('unet') || modelName.includes('model1')) {
      return MODEL_CONFIG.MODEL_NAMES.UNET; // "U-Net"
    }
    if (modelName.toLowerCase().includes('deeplab') || modelName.includes('model2')) {
      return MODEL_CONFIG.MODEL_NAMES.DEEPLAB; // "DeepLabV3+"
    }
    return modelName;
  };

  // Pipeline steps
  const [steps, setSteps] = useState<PredictionStep[]>([
    {
      id: 'upload',
      label: 'Image Upload',
      icon: Upload,
      status: 'pending'
    },
    {
      id: 'preprocessing',
      label: 'Image Preprocessing',
      icon: Eye,
      status: 'pending'
    },
    {
      id: 'analysis',
      label: 'AI Model Analysis',
      icon: Brain,
      status: 'pending'
    },
    {
      id: 'ensemble',
      label: 'Ensemble Processing',
      icon: Target,
      status: 'pending'
    }
  ]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setError(null);
      setProcessedData(null);
      setRawResult(null);
      
      // Reset steps
      setSteps(prev => prev.map(step => 
        step.id === 'upload' 
          ? { ...step, status: 'completed' }
          : { ...step, status: 'pending' }
      ));
    }
  };

  const updateStep = (stepId: string, status: PredictionStep['status'], result?: string, confidence?: number, processingTime?: number) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { ...step, status, result, confidence, processingTime }
        : step
    ));
  };

  const runPrediction = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    try {
      // Step 1: Preprocessing
      updateStep('preprocessing', 'processing');
      await new Promise(resolve => setTimeout(resolve, 800)); // Simulate preprocessing
      updateStep('preprocessing', 'completed', 'Image preprocessed', 1.0, 0.8);

      // Step 2: AI Analysis
      updateStep('analysis', 'processing');
      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate AI processing
      updateStep('analysis', 'completed', 'Models analyzed', 0.9, 2.3);

      // Step 3: Ensemble Processing
      updateStep('ensemble', 'processing');
      const ensembleResult = await apiClient.detailedEnsemblePredict(selectedFile);
      
      if (!ensembleResult.success) {
        throw new Error(ensembleResult.error || 'Prediction failed');
      }
      
      updateStep('ensemble', 'completed', ensembleResult.ensemble_prediction, ensembleResult.ensemble_confidence, ensembleResult.total_processing_time);

      // Process data through master processor
      setRawResult(ensembleResult);
      const processed = processPredictionData(ensembleResult);
      setProcessedData(processed);

    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred during prediction');
      updateStep('ensemble', 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const getStepIcon = (step: PredictionStep) => {
    const IconComponent = step.icon;
    return <IconComponent className="w-5 h-5" />;
  };

  const getStepStatusColor = (status: PredictionStep['status']) => {
    switch (status) {
      case 'completed': return 'bg-gradient-to-r from-green-500 to-green-600';
      case 'processing': return 'bg-gradient-to-r from-blue-500 to-blue-600 animate-pulse';
      case 'error': return 'bg-gradient-to-r from-red-500 to-red-600';
      default: return 'bg-gradient-to-r from-gray-400 to-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <div className="flex items-center justify-center gap-3">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full text-white">
              <Brain className="w-8 h-8" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Prediction Pipeline
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced oil spill detection using ensemble deep learning models with real-time analysis
          </p>
        </motion.div>

        {/* Upload Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="border-2 border-dashed border-blue-300 bg-white/70 backdrop-blur-sm">
            <CardContent className="p-8">
              <div className="space-y-6">
                {/* File Upload Area */}
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                  />
                  <label 
                    htmlFor="file-upload" 
                    className="cursor-pointer block p-8 text-center rounded-xl border-2 border-dashed border-blue-300 hover:border-blue-500 transition-all duration-300 hover:bg-blue-50"
                  >
                    <Upload className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-700 mb-2">Upload Satellite Image</h3>
                    <p className="text-gray-500">PNG, JPG, GIF up to 10MB • Drag & drop or click to browse</p>
                  </label>
                </div>

                {/* Image Preview */}
                {previewUrl && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex justify-center"
                  >
                    <div className="relative">
                      <img 
                        src={previewUrl} 
                        alt="Preview" 
                        className="max-w-md max-h-64 object-contain rounded-xl shadow-lg border-2 border-white"
                      />
                      <div className="absolute -top-2 -right-2 bg-green-500 text-white rounded-full p-2">
                        <CheckCircle className="w-4 h-4" />
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Action Button */}
                <div className="flex justify-center">
                  <Button 
                    onClick={runPrediction}
                    disabled={!selectedFile || isProcessing}
                    size="lg"
                    className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold rounded-xl shadow-lg transition-all duration-300 transform hover:scale-105"
                  >
                    {isProcessing ? (
                      <>
                        <Activity className="w-5 h-5 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5 mr-2" />
                        Run AI Analysis
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Pipeline Steps */}
        <AnimatePresence>
          {(isProcessing || processedData) && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-white/80 backdrop-blur-sm border-0 shadow-xl">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Activity className="w-6 h-6 text-blue-600" />
                    Analysis Pipeline
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {steps.map((step, index) => (
                      <motion.div
                        key={step.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-center gap-4"
                      >
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-300 ${getStepStatusColor(step.status)}`}>
                          {step.status === 'completed' ? (
                            <CheckCircle className="w-6 h-6" />
                          ) : step.status === 'error' ? (
                            <AlertCircle className="w-6 h-6" />
                          ) : (
                            getStepIcon(step)
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-semibold text-lg">{step.label}</span>
                            {step.status === 'completed' && step.result && (
                              <Badge 
                                variant={step.result.includes('Oil Spill Detected') ? 'destructive' : 'default'}
                                className="text-sm px-3 py-1"
                              >
                                {step.result}
                              </Badge>
                            )}
                          </div>
                          {step.status === 'completed' && step.confidence && (
                            <div className="space-y-2">
                              <div className="flex items-center gap-4">
                                <Progress 
                                  value={step.confidence * 100} 
                                  className="flex-1 h-3"
                                />
                                <span className="text-sm font-medium text-gray-700 min-w-fit">
                                  {(step.confidence * 100).toFixed(1)}%
                                </span>
                                {step.processingTime && (
                                  <span className="text-xs text-gray-500 flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {step.processingTime.toFixed(2)}s
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <AnimatePresence>
          {processedData && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="bg-white/90 backdrop-blur-sm border-0 shadow-2xl">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-3 text-2xl">
                    <Target className="w-7 h-7 text-purple-600" />
                    Analysis Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="summary" className="w-full">
                    <TabsList className="grid w-full grid-cols-4 mb-6 bg-gray-100 p-1 rounded-xl">
                      <TabsTrigger value="summary" className="rounded-lg">Summary</TabsTrigger>
                      <TabsTrigger value="models" className="rounded-lg">Models</TabsTrigger>
                      <TabsTrigger value="analysis" className="rounded-lg">Analysis</TabsTrigger>
                      <TabsTrigger value="details" className="rounded-lg">Details</TabsTrigger>
                    </TabsList>

                    {/* Summary Tab */}
                    <TabsContent value="summary" className="space-y-6">
                      <div className="text-center space-y-6">
                        <motion.div
                          initial={{ scale: 0.9 }}
                          animate={{ scale: 1 }}
                          className={`inline-flex items-center px-8 py-4 rounded-2xl border-2 text-xl font-bold shadow-lg ${processedData.riskColor}`}
                        >
                          {processedData.finalPrediction === "Oil Spill Detected" ? (
                            <AlertCircle className="w-6 h-6 mr-3" />
                          ) : (
                            <CheckCircle className="w-6 h-6 mr-3" />
                          )}
                          {processedData.finalPrediction}
                        </motion.div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                            <CardContent className="p-6 text-center">
                              <div className="text-3xl font-bold text-blue-600 mb-2">
                                {processedData.confidencePercentage}%
                              </div>
                              <div className="text-sm text-blue-700 font-medium">
                                Confidence Score
                              </div>
                              <Progress 
                                value={processedData.confidencePercentage} 
                                className="mt-3 h-2"
                              />
                            </CardContent>
                          </Card>

                          <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                            <CardContent className="p-6 text-center">
                              <div className="text-3xl font-bold text-purple-600 mb-2">
                                {processedData.riskLevel}
                              </div>
                              <div className="text-sm text-purple-700 font-medium">
                                Risk Level
                              </div>
                              <div className={`mt-3 px-3 py-1 rounded-full text-xs font-medium ${processedData.riskColor}`}>
                                Assessment Complete
                              </div>
                            </CardContent>
                          </Card>

                          <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
                            <CardContent className="p-6 text-center">
                              <div className="text-3xl font-bold text-green-600 mb-2">
                                {processedData.totalProcessingTime.toFixed(1)}s
                              </div>
                              <div className="text-sm text-green-700 font-medium">
                                Processing Time
                              </div>
                              <div className="mt-3 text-xs text-green-600">
                                {processedData.modelCount} models analyzed
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                      </div>
                    </TabsContent>

                    {/* Models Tab */}
                    <TabsContent value="models" className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {processedData.individualResults.map((result, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <Card className="h-full bg-gradient-to-br from-gray-50 to-gray-100 border-gray-200 hover:shadow-lg transition-all duration-300">
                              <CardHeader className="pb-3">
                                <CardTitle className="flex items-center gap-2 text-lg">
                                  {result.modelName === "U-Net" ? (
                                    <Brain className="w-5 h-5 text-blue-600" />
                                  ) : (
                                    <Zap className="w-5 h-5 text-purple-600" />
                                  )}
                                  {result.modelName}
                                </CardTitle>
                              </CardHeader>
                              <CardContent className="space-y-4">
                                <Badge 
                                  variant={result.prediction.includes('Oil Spill Detected') ? 'destructive' : 'default'}
                                  className="text-sm px-3 py-1"
                                >
                                  {result.prediction}
                                </Badge>
                                
                                <div className="space-y-3">
                                  <div className="flex justify-between items-center">
                                    <span className="text-sm font-medium">Confidence:</span>
                                    <span className="text-sm font-bold">{result.confidencePercentage}%</span>
                                  </div>
                                  <Progress 
                                    value={result.confidencePercentage} 
                                    className="h-2"
                                  />
                                  
                                  <div className="flex justify-between items-center text-xs text-gray-600">
                                    <span>Processing Time:</span>
                                    <span className="font-medium">{result.processingTime.toFixed(2)}s</span>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          </motion.div>
                        ))}
                      </div>

                      {/* Model Agreement */}
                      <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200">
                        <CardContent className="p-6">
                          <div className="flex items-center gap-3 mb-4">
                            <TrendingUp className="w-5 h-5 text-indigo-600" />
                            <h3 className="text-lg font-semibold text-indigo-800">Model Agreement</h3>
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <div className="text-2xl font-bold text-indigo-600">
                                {processedData.modelAgreement.agreementPercentage}%
                              </div>
                              <div className="text-sm text-indigo-700">Agreement Score</div>
                            </div>
                            <div>
                              <Badge 
                                variant={processedData.modelAgreement.consensus ? 'default' : 'destructive'}
                                className="text-sm"
                              >
                                {processedData.modelAgreement.consensus ? 'Consensus Reached' : 'Models Disagree'}
                              </Badge>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </TabsContent>

                    {/* Analysis Tab */}
                    <TabsContent value="analysis" className="space-y-6">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Oil Spill Analysis */}
                        <Card className="bg-gradient-to-br from-red-50 to-orange-50 border-red-200">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-red-800">
                              <Eye className="w-5 h-5" />
                              Oil Spill Analysis
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-4">
                            <div className="space-y-3">
                              <div className="flex justify-between">
                                <span className="text-sm">Pixel Coverage:</span>
                                <span className="font-bold">{processedData.oilSpillAnalysis.pixelPercentage.toFixed(2)}%</span>
                              </div>
                              
                              <div className="flex justify-between">
                                <span className="text-sm">Detection Status:</span>
                                <Badge variant={processedData.oilSpillAnalysis.isDetected ? 'destructive' : 'default'}>
                                  {processedData.oilSpillAnalysis.isDetected ? 'Detected' : 'Not Detected'}
                                </Badge>
                              </div>
                              
                              <div className="flex justify-between">
                                <span className="text-sm">Severity Level:</span>
                                <span className="font-medium">{processedData.oilSpillAnalysis.severity}</span>
                              </div>
                            </div>
                            
                            <Separator />
                            
                            <div className="text-xs text-red-700 bg-red-100 p-3 rounded-lg">
                              <strong>Detection Threshold:</strong> 1.0% pixel coverage (based on notebook specifications)
                            </div>
                          </CardContent>
                        </Card>

                        {/* Class Breakdown */}
                        <Card className="bg-gradient-to-br from-green-50 to-blue-50 border-green-200">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-green-800">
                              <BarChart3 className="w-5 h-5" />
                              Class Distribution
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-3">
                              {processedData.classBreakdown.map((cls, index) => (
                                <div key={index} className="space-y-1">
                                  <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                      <div 
                                        className="w-3 h-3 rounded-full border"
                                        style={{ backgroundColor: `rgb(${cls.color.join(',')})` }}
                                      ></div>
                                      <span className="text-sm font-medium">{cls.className}</span>
                                    </div>
                                    <span className="text-sm font-bold">{cls.percentage}%</span>
                                  </div>
                                  <Progress 
                                    value={cls.percentage} 
                                    className="h-1"
                                  />
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </TabsContent>

                    {/* Details Tab */}
                    <TabsContent value="details" className="space-y-6">
                      <Alert className="border-blue-200 bg-blue-50">
                        <Info className="h-4 w-4 text-blue-600" />
                        <AlertDescription className="text-blue-800">
                          Detailed pixel-level classification across all 5 classes from notebook: Background, Oil Spill, Ships, Looklike, Wakes.
                          Using 1.0% detection threshold and exact color mappings from training data.
                        </AlertDescription>
                      </Alert>
                      
                      {rawResult?.individual_predictions && (
                        <div className="space-y-4">
                          <h4 className="text-lg font-semibold">Raw Model Outputs</h4>
                          <div className="grid grid-cols-1 gap-4">
                            {rawResult.individual_predictions.map((prediction, index) => (
                              <Card key={index} className="bg-gray-50">
                                <CardContent className="p-4">
                                  <div className="space-y-2">
                                    <div className="font-medium text-lg">{getStandardizedModelName(prediction.model_name)}</div>
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                      <div>
                                        <span className="text-gray-600">Result:</span>
                                        <span className="font-mono ml-2">{prediction.prediction}</span>
                                      </div>
                                      <div>
                                        <span className="text-gray-600">Confidence:</span>
                                        <span className="font-mono ml-2">{((prediction.confidence || 0) * 100).toFixed(2)}%</span>
                                      </div>
                                      <div>
                                        <span className="text-gray-600">Processing:</span>
                                        <span className="font-mono ml-2">{(prediction.processing_time || 0).toFixed(3)}s</span>
                                      </div>
                                      <div>
                                        <span className="text-gray-600">Model:</span>
                                        <span className="font-mono ml-2">{prediction.model_name}</span>
                                      </div>
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                        </div>
                      )}
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <Alert variant="destructive" className="bg-red-50 border-red-300">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-red-800">{error}</AlertDescription>
              </Alert>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
