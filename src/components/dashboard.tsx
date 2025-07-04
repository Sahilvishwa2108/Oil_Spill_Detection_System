"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FileUpload } from "@/components/ui/file-upload"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PredictionResults } from "@/components/prediction-results"
import { apiClient } from "@/lib/api"
import { HealthStatus, ModelInfo, EnsemblePredictionResult } from "@/types/api"
import { 
  Upload, 
  Zap, 
  Activity, 
  Info, 
  Brain,
  Image as ImageIcon,
  BarChart3,
  AlertCircle
} from "lucide-react"

export default function Dashboard() {  // State management
  const [files, setFiles] = useState<File[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<EnsemblePredictionResult | null>(null)
  const [originalImageUrl, setOriginalImageUrl] = useState<string>("")
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)
  const [modelsInfo, setModelsInfo] = useState<Record<string, ModelInfo>>({})
  const [error, setError] = useState<string>("")
  const [retryCount, setRetryCount] = useState<number>(0)
  const [isRetrying, setIsRetrying] = useState<boolean>(false)
  // Load initial data
  useEffect(() => {
    loadHealthStatus()
    loadModelsInfo()
    
    // Set up periodic health checks every 30 seconds
    const healthCheckInterval = setInterval(() => {
      if (!isLoading && !isRetrying) {
        loadHealthStatus()
      }
    }, 30000)

    return () => clearInterval(healthCheckInterval)
  }, [isLoading, isRetrying])
  
  const loadHealthStatus = async () => {
    try {
      const status = await apiClient.healthCheck()
      setHealthStatus(status || { status: 'unknown', models_loaded: { model1: false, model2: false }, timestamp: new Date().toISOString() })
      setError("") // Clear any previous connection errors
    } catch (err) {
      console.error("Failed to load health status:", err)
      setError("Unable to connect to the backend service. Please check if the service is running.")
      setHealthStatus({ status: 'offline', models_loaded: { model1: false, model2: false }, timestamp: new Date().toISOString() })
    }
  }
  const loadModelsInfo = async () => {
    try {
      const response = await apiClient.getModelsInfo()
      setModelsInfo(response.models || {})
    } catch (err) {
      console.error("Failed to load models info:", err)
      setError("Failed to load model information. Some features may not work correctly.")
    }
  }
  const handlePredict = async () => {
    if (files.length === 0) {
      setError("Please select an image first")
      return
    }

    setIsLoading(true)
    setError("")
    setPrediction(null)

    try {
      // Create preview URL for original image
      setOriginalImageUrl(URL.createObjectURL(files[0]))
      
      // Use detailed prediction for better analytics
      const result = await apiClient.detailedEnsemblePredict(files[0])
      setPrediction(result)
      setError("") // Clear any previous errors on success
    } catch (err) {
      console.error("Prediction error:", err)
      
      // Fallback to regular ensemble prediction if detailed fails
      try {
        console.log("Falling back to regular ensemble prediction...")
        const result = await apiClient.ensemblePredict(files[0])
        setPrediction(result)
        setError("") // Clear any previous errors on success
      } catch (fallbackErr) {
        console.error("Fallback prediction also failed:", fallbackErr)
        
        // Provide more specific error messages based on error type
        if (err instanceof Error) {
          if (err.message.includes('Unable to connect')) {
            setError("Cannot connect to the prediction service. Please check your internet connection and try again.")
          } else if (err.message.includes('413') || err.message.includes('too large')) {
            setError("Image file is too large. Please try with a smaller image.")
          } else if (err.message.includes('415') || err.message.includes('unsupported')) {
            setError("Unsupported image format. Please use PNG, JPG, JPEG, GIF, BMP, or TIFF files.")
          } else if (err.message.includes('500')) {
            setError("Server error occurred during prediction. Please try again later.")
          } else {
            setError(`Prediction failed: ${err.message}`)
          }
        } else {
          setError("An unexpected error occurred. Please try again.")
        }
      }
    } finally {
      setIsLoading(false)
    }
  }
  const resetPrediction = () => {
    setPrediction(null)
    setFiles([])
    setOriginalImageUrl("")
    setError("")
    setRetryCount(0)
  }

  const retryPrediction = async () => {
    if (retryCount >= 3) {
      setError("Maximum retry attempts reached. Please try again later.")
      return
    }

    setIsRetrying(true)
    setRetryCount(prev => prev + 1)
    
    // Wait a bit before retrying
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    try {
      await handlePredict()
    } finally {
      setIsRetrying(false)
    }
  }

  const handleFileUploadError = (error: string) => {
    setError(error)
  }

  const isBackendHealthy = healthStatus?.status === "healthy"
  const modelsLoaded = healthStatus?.models_loaded || { model1: false, model2: false }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-4">
            Oil Spill Detection Dashboard
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Advanced machine learning models for detecting oil spills in satellite and aerial imagery
          </p>
        </div>

        {/* System Status & Model Performance */}
        <div className="mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    System Status
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      loadHealthStatus()
                      loadModelsInfo()
                    }}
                    disabled={isLoading || isRetrying}
                  >
                    Refresh
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Backend API</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${isBackendHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
                      <Badge variant={isBackendHealthy ? 'default' : 'destructive'}>
                        {isBackendHealthy ? 'Online' : 'Offline'}
                      </Badge>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">UNet Model</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${modelsLoaded.model1 ? 'bg-green-500' : 'bg-gray-500'}`} />
                      <Badge variant={modelsLoaded.model1 ? 'default' : 'secondary'}>
                        {modelsLoaded.model1 ? 'Ready' : 'Loading...'}
                      </Badge>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">DeepLabV3+ Model</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${modelsLoaded.model2 ? 'bg-green-500' : 'bg-gray-500'}`} />
                      <Badge variant={modelsLoaded.model2 ? 'default' : 'secondary'}>
                        {modelsLoaded.model2 ? 'Ready' : 'Loading...'}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Model Performance Comparison */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Model Performance
                </CardTitle>
                <CardDescription>
                  Comparison of our trained models on oil spill detection
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Dynamic Model Performance */}
                  {Object.entries(modelsInfo).map(([modelKey, modelData]) => (
                    <div key={modelKey} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{modelData.name}</span>
                        <span className="text-sm text-muted-foreground">
                          {modelData.f1_score ? (modelData.f1_score * 100).toFixed(2) : '93.56'}% F1
                        </span>
                      </div>
                      <Progress value={modelData.f1_score ? modelData.f1_score * 100 : 93.56} className="h-2" />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>
                          {modelData.name === 'UNet' ? 'Fast' : 'Accurate'} • {modelData.size_mb || 'N/A'} MB
                        </span>
                        <span>{modelData.description}</span>
                      </div>
                    </div>
                  ))}

                  {/* Fallback display if no model info loaded */}
                  {Object.keys(modelsInfo).length === 0 && (
                    <>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">UNet</span>
                          <span className="text-sm text-muted-foreground">93.56% F1</span>
                        </div>
                        <Progress value={93.56} className="h-2" />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Fast • 22.4 MB</span>
                          <span>Optimized for speed</span>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">DeepLabV3+</span>
                          <span className="text-sm text-muted-foreground">96.68% F1</span>
                        </div>
                        <Progress value={96.68} className="h-2" />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Accurate • 204.6 MB</span>
                          <span>Highest accuracy</span>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Ensemble Benefits */}
                  <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-4 h-4 text-blue-600" />
                      <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                        Ensemble Advantage
                      </span>
                    </div>
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      Our ensemble combines both models for improved accuracy and reliability, 
                      achieving better performance than individual models.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        <Tabs defaultValue="prediction" className="space-y-8">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="prediction" className="flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Prediction
            </TabsTrigger>
            <TabsTrigger value="models" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Models
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Analytics
            </TabsTrigger>
          </TabsList>

          {/* Prediction Tab */}
          <TabsContent value="prediction" className="space-y-6">            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription className="flex items-center justify-between">
                  <span>{error}</span>
                  <div className="flex gap-2 ml-4">
                    {prediction === null && files.length > 0 && retryCount < 3 && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={retryPrediction}
                        disabled={isRetrying}
                      >
                        {isRetrying ? "Retrying..." : `Retry (${3 - retryCount} left)`}
                      </Button>
                    )}
                    <button
                      onClick={() => setError("")}
                      className="text-sm underline hover:no-underline"
                    >
                      Dismiss
                    </button>
                  </div>
                </AlertDescription>
              </Alert>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Input Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Upload Image
                  </CardTitle>
                  <CardDescription>
                    Upload satellite or aerial imagery for oil spill detection
                  </CardDescription>
                </CardHeader>                <CardContent className="space-y-6">                  <FileUpload
                    onFilesChange={setFiles}
                    disabled={isLoading || isRetrying}
                    onError={handleFileUploadError}
                    maxSize={5} // 5MB limit
                  />

                  <div className="text-sm text-muted-foreground bg-blue-50 dark:bg-blue-950 p-3 rounded-lg">
                    <p className="font-medium text-blue-900 dark:text-blue-100 mb-1">Ensemble Prediction</p>
                    <p className="text-blue-700 dark:text-blue-300">
                      Both U-Net and DeepLab V3+ models will analyze your image simultaneously for the most accurate results.
                    </p>
                  </div>

                  <div className="flex gap-3">                    <Button
                      onClick={handlePredict}
                      disabled={isLoading || isRetrying || files.length === 0 || !isBackendHealthy}
                      className="flex-1"
                    >
                      {isLoading || isRetrying ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                          {isRetrying ? `Retrying... (${retryCount}/3)` : "Analyzing..."}
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4 mr-2" />
                          Detect Oil Spill
                        </>
                      )}
                    </Button>
                    
                    {prediction && (
                      <Button variant="outline" onClick={resetPrediction}>
                        Reset
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Quick Stats */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Info className="w-5 h-5" />
                    Quick Stats
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">2</div>
                        <div className="text-sm text-muted-foreground">AI Models</div>
                      </div>
                      <div className="text-center p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold text-green-600">95%</div>
                        <div className="text-sm text-muted-foreground">Accuracy</div>
                      </div>
                    </div>
                    
                    <div className="text-center p-4 bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950 dark:to-cyan-950 rounded-lg">
                      <ImageIcon className="w-8 h-8 mx-auto mb-2 text-blue-600" />
                      <div className="text-sm">
                        Supports: PNG, JPG, JPEG, GIF, BMP, TIFF
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Results Section */}
            {prediction && (
              <PredictionResults 
                result={prediction} 
                originalImage={originalImageUrl}
              />
            )}
          </TabsContent>

          {/* Models Tab */}
          <TabsContent value="models" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(modelsInfo).map(([modelKey, model]) => (                <Card key={modelKey}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span className="flex items-center gap-2">
                        <Brain className="w-5 h-5" />
                        {model.name}
                      </span>
                      <Badge variant={modelsLoaded[modelKey as keyof typeof modelsLoaded] ? "default" : "destructive"}>
                        {modelsLoaded[modelKey as keyof typeof modelsLoaded] ? "Loaded" : "Not Loaded"}
                      </Badge>
                    </CardTitle>
                    <CardDescription>{model.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="text-sm">
                      <span className="font-medium">Status:</span>
                      <div className="text-muted-foreground">
                        {model.loaded ? "Ready for predictions" : "Will load on first use"}
                      </div>
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">Type:</span>
                      <div className="text-muted-foreground">
                        Semantic Segmentation Model
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Detection Accuracy</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-600 mb-2">95.2%</div>
                  <Progress value={95.2} className="mb-2" />
                  <p className="text-sm text-muted-foreground">
                    Average accuracy across validation dataset
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Processing Speed</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-600 mb-2">1.2s</div>
                  <p className="text-sm text-muted-foreground">
                    Average processing time per image
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Precision</span>
                      <span className="text-sm font-medium">94.8%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Recall</span>
                      <span className="text-sm font-medium">95.6%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">F1-Score</span>
                      <span className="text-sm font-medium">95.2%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
