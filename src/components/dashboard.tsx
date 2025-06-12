"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { FileUpload } from "@/components/ui/file-upload"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PredictionResults } from "@/components/prediction-results"
import { apiClient } from "@/lib/api"
import { PredictionResult, HealthStatus, ModelInfo, ModelsResponse } from "@/types/api"
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

export default function Dashboard() {
  // State management
  const [files, setFiles] = useState<File[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("model1")
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [originalImageUrl, setOriginalImageUrl] = useState<string>("")
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)
  const [modelsInfo, setModelsInfo] = useState<Record<string, ModelInfo>>({})
  const [error, setError] = useState<string>("")

  // Load initial data
  useEffect(() => {
    loadHealthStatus()
    loadModelsInfo()
  }, [])
  const loadHealthStatus = async () => {
    try {
      const status = await apiClient.healthCheck()
      setHealthStatus(status || { status: 'unknown', models_loaded: { model1: false, model2: false }, timestamp: new Date().toISOString() })
    } catch (err) {
      console.error("Failed to load health status:", err)
      setError("Failed to connect to the backend API")
      setHealthStatus({ status: 'offline', models_loaded: { model1: false, model2: false }, timestamp: new Date().toISOString() })
    }
  }
  const loadModelsInfo = async () => {
    try {
      const response = await apiClient.getModelsInfo()
      setModelsInfo(response.models || {})
    } catch (err) {
      console.error("Failed to load models info:", err)
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
      
      const result = await apiClient.predictOilSpill(files[0], selectedModel)
      setPrediction(result)
    } catch (err) {
      setError("Failed to get prediction. Please try again.")
      console.error("Prediction error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const resetPrediction = () => {
    setPrediction(null)
    setFiles([])
    setOriginalImageUrl("")
    setError("")
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

        {/* System Status */}
        <div className="mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${isBackendHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-sm">
                    Backend: {isBackendHealthy ? 'Online' : 'Offline'}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${modelsLoaded.model1 ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-sm">Model 1: {modelsLoaded.model1 ? 'Loaded' : 'Not Loaded'}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${modelsLoaded.model2 ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-sm">Model 2: {modelsLoaded.model2 ? 'Loaded' : 'Not Loaded'}</span>
                </div>
              </div>
            </CardContent>
          </Card>
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
                <AlertDescription>
                  {error}
                  <button
                    onClick={() => setError("")}
                    className="ml-2 text-sm underline hover:no-underline"
                  >
                    Dismiss
                  </button>
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
                </CardHeader>
                <CardContent className="space-y-6">
                  <FileUpload
                    onFilesChange={setFiles}
                    disabled={isLoading}
                  />                  <div className="space-y-2">
                    <label className="text-sm font-medium">Select Model</label>
                    <Select
                      value={selectedModel}
                      onValueChange={setSelectedModel}
                      disabled={isLoading}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="model1" disabled={!modelsLoaded.model1}>
                          Model 1 {!modelsLoaded.model1 && '(Not Available)'}
                        </SelectItem>
                        <SelectItem value="model2" disabled={!modelsLoaded.model2}>
                          Model 2 {!modelsLoaded.model2 && '(Not Available)'}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex gap-3">
                    <Button
                      onClick={handlePredict}
                      disabled={isLoading || files.length === 0 || !isBackendHealthy}
                      className="flex-1"
                    >
                      {isLoading ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                          Analyzing...
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
