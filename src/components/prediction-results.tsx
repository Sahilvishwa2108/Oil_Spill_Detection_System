"use client"

import * as React from "react"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { motion } from "framer-motion"
import { Clock, Zap, Target, Activity, Users, Brain, BarChart3, TrendingUp } from "lucide-react"
import { BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer, CartesianGrid, XAxis, YAxis, Tooltip } from "recharts"
import { EnsemblePredictionResult } from "@/types/api"

interface PredictionResultsProps {
  result: EnsemblePredictionResult
  originalImage?: string
}

// Color scheme for charts
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

export function PredictionResults({ result, originalImage }: PredictionResultsProps) {
  if (!result.success) {
    return (
      <Card className="border-destructive">
        <CardHeader>
          <CardTitle className="text-destructive">Prediction Failed</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{result.error}</p>
        </CardContent>
      </Card>
    )
  }

  const confidencePercentage = Math.round((result.ensemble_confidence || 0) * 100)
  const hasOilSpill = confidencePercentage > 50

  // Prepare data for charts
  const modelComparisonData = result.individual_predictions?.map((pred) => ({
    name: pred.model_name,
    confidence: Math.round(pred.confidence * 100),
    time: pred.processing_time,
    prediction: pred.prediction
  })) || []

  const confidenceDistributionData = [
    { name: 'Oil Spill', value: confidencePercentage, color: COLORS[3] },
    { name: 'Clean Water', value: 100 - confidencePercentage, color: COLORS[1] }
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
        ease: "easeOut"
      }
    }
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Main Results Header */}
      <motion.div variants={itemVariants}>
        <Card className={`border-2 ${hasOilSpill ? 'border-red-200 bg-red-50/50' : 'border-green-200 bg-green-50/50'}`}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Target className="w-6 h-6" />
              </motion.div>
              Ensemble Detection Results
            </CardTitle>
            <CardDescription>
              Combined analysis from {result.individual_predictions?.length || 2} AI models
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-white/50 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Activity className="w-4 h-4" />
                  <span className="text-sm font-medium">Final Detection</span>
                </div>
                <Badge variant={hasOilSpill ? "destructive" : "secondary"} className="text-lg px-4 py-2">
                  {result.ensemble_prediction || "Analysis Complete"}
                </Badge>
              </div>
              
              <div className="text-center p-4 bg-white/50 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Zap className="w-4 h-4" />
                  <span className="text-sm font-medium">Ensemble Confidence</span>
                </div>
                <div className="space-y-2">
                  <motion.div 
                    className="text-2xl font-bold"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    {confidencePercentage}%
                  </motion.div>
                  <Progress value={confidencePercentage} className="h-2" />
                </div>
              </div>
              
              <div className="text-center p-4 bg-white/50 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Clock className="w-4 h-4" />
                  <span className="text-sm font-medium">Total Processing Time</span>
                </div>
                <motion.div 
                  className="text-2xl font-bold"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  {result.total_processing_time?.toFixed(2)}s
                </motion.div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Visual Analytics */}
      <motion.div variants={itemVariants}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Visual Analytics
            </CardTitle>
            <CardDescription>
              Advanced visualization of prediction confidence and model performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Confidence Distribution Chart */}
              <div className="space-y-4">
                <h4 className="font-medium flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Confidence Distribution
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={confidenceDistributionData}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}%`}
                      >
                        {confidenceDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Model Performance Comparison */}
              <div className="space-y-4">
                <h4 className="font-medium flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Model Performance
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={modelComparisonData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="confidence" fill={COLORS[0]} name="Confidence %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Individual Model Results */}
      {result.individual_predictions && result.individual_predictions.length > 0 && (
        <motion.div variants={itemVariants}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="w-5 h-5" />
                Individual Model Results
              </CardTitle>
              <CardDescription>
                Detailed analysis from each AI model in the ensemble
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.individual_predictions.map((modelResult, index) => {
                  const modelConfidence = Math.round((modelResult.confidence || 0) * 100)
                  const modelHasOilSpill = modelConfidence > 50
                  
                  return (
                    <motion.div 
                      key={index} 
                      className="p-4 border rounded-lg space-y-3"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.1 * index }}
                    >
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium flex items-center gap-2">
                          <Brain className="w-4 h-4" />
                          {modelResult.model_name}
                        </h4>
                        <Badge variant={modelHasOilSpill ? "destructive" : "secondary"}>
                          {modelResult.prediction}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Confidence:</span>
                          <span className="font-medium">{modelConfidence}%</span>
                        </div>
                        <Progress value={modelConfidence} className="h-1" />
                      </div>
                      
                      <div className="text-sm text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Processing time: {modelResult.processing_time?.toFixed(2)}s
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Magical AI Prediction Flow */}
      <motion.div variants={itemVariants}>
        <Card className="bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50 dark:from-slate-900 dark:via-blue-950 dark:to-cyan-950 border-2 border-blue-200 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <motion.div
                animate={{ 
                  rotate: [0, 360],
                  scale: [1, 1.2, 1]
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                <Brain className="w-6 h-6 text-purple-600" />
              </motion.div>
              AI Prediction Pipeline
            </CardTitle>
            <CardDescription>
              Step-by-step neural network analysis with magical transitions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-8">
              {/* Step-by-step prediction flow */}
              <div className="relative">
                {/* Progress Line */}
                <div className="absolute top-24 left-1/2 transform -translate-x-1/2 w-0.5 h-full bg-gradient-to-b from-cyan-400 via-blue-500 to-purple-600 opacity-30" />
                
                {/* Original Image */}
                {originalImage && (
                  <motion.div
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.8 }}
                    className="relative mb-8"
                  >
                    <div className="flex items-center gap-4">                        <div className="w-40 h-40 relative">
                          <motion.div
                            className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-xl blur-lg"
                            animate={{
                              scale: [1, 1.1, 1],
                              opacity: [0.3, 0.6, 0.3]
                            }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                          <div className="relative bg-white dark:bg-gray-900 rounded-xl overflow-hidden border-2 border-cyan-200 dark:border-cyan-800">
                            <Image
                              src={originalImage}
                              alt="Original satellite/aerial imagery"
                              fill
                              className="object-cover"
                              sizes="(max-width: 768px) 100vw, 50vw"
                              priority
                            />
                            {/* Image Classification Label */}
                            <div className="absolute top-2 left-2 right-2">
                              <Badge className="bg-cyan-500 text-white text-xs font-medium">
                                📡 INPUT IMAGE
                              </Badge>
                            </div>
                            <div className="absolute bottom-2 left-2 right-2">
                              <div className="bg-black/70 backdrop-blur-sm rounded px-2 py-1">
                                <div className="text-white text-xs font-medium">Original Data</div>
                                <div className="text-white/80 text-xs">Satellite/Aerial Imagery</div>
                              </div>
                            </div>
                          </div>
                        </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <motion.div
                            className="w-3 h-3 bg-gradient-to-r from-cyan-400 to-blue-500 rounded-full"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 1.5, repeat: Infinity }}
                          />
                          <h4 className="font-semibold text-lg">Original Image</h4>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Input satellite/aerial imagery ready for neural network analysis
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
                
                {/* Model Predictions */}
                {result.individual_predictions?.map((modelResult, index) => (
                  modelResult.prediction_mask && (
                    <motion.div
                      key={`step-${index}`}
                      initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.8, delay: 0.3 * (index + 1) }}
                      className="relative mb-8"
                    >
                      <div className={`flex items-center gap-4 ${index % 2 === 1 ? 'flex-row-reverse' : ''}`}>
                        <div className="w-40 h-40 relative">
                          <motion.div
                            className={`absolute inset-0 rounded-xl blur-lg ${
                              modelResult.model_name === 'UNet' 
                                ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20' 
                                : 'bg-gradient-to-r from-purple-500/20 to-pink-500/20'
                            }`}
                            animate={{
                              scale: [1, 1.1, 1],
                              opacity: [0.3, 0.6, 0.3]
                            }}
                            transition={{ duration: 2, repeat: Infinity, delay: 0.5 * index }}
                          />
                          <div className={`relative bg-white dark:bg-gray-900 rounded-xl overflow-hidden border-2 ${
                            modelResult.model_name === 'UNet' 
                              ? 'border-green-200 dark:border-green-800' 
                              : 'border-purple-200 dark:border-purple-800'
                          }`}>
                            <Image
                              src={`data:image/png;base64,${modelResult.prediction_mask}`}
                              alt={`${modelResult.model_name} prediction`}
                              fill
                              className="object-cover"
                              sizes="(max-width: 768px) 100vw, 50vw"
                            />
                          </div>
                          
                          {/* Confidence Badge */}
                          <motion.div
                            className="absolute -top-2 -right-2"
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ duration: 0.5, delay: 0.5 * (index + 1) }}
                          >
                            <Badge className={`${
                              modelResult.model_name === 'UNet' 
                                ? 'bg-green-500 hover:bg-green-600' 
                                : 'bg-purple-500 hover:bg-purple-600'
                            } text-white`}>
                              {Math.round(modelResult.confidence * 100)}%
                            </Badge>
                          </motion.div>
                        </div>
                        
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <motion.div
                              className={`w-3 h-3 rounded-full ${
                                modelResult.model_name === 'UNet' 
                                  ? 'bg-gradient-to-r from-green-400 to-emerald-500' 
                                  : 'bg-gradient-to-r from-purple-400 to-pink-500'
                              }`}
                              animate={{ scale: [1, 1.2, 1] }}
                              transition={{ duration: 1.5, repeat: Infinity, delay: 0.3 * index }}
                            />
                            <h4 className="font-semibold text-lg">{modelResult.model_name}</h4>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2">
                            {modelResult.model_name === 'UNet' 
                              ? 'Fast semantic segmentation with U-Net architecture' 
                              : 'High-accuracy detection with DeepLab V3+ model'}
                          </p>
                          <div className="flex items-center gap-4 text-sm">
                            <div className="flex items-center gap-1">
                              <Target className="w-4 h-4 text-blue-500" />
                              <span>Confidence: {Math.round(modelResult.confidence * 100)}%</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4 text-gray-500" />
                              <span>{modelResult.processing_time?.toFixed(2)}s</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )
                ))}
                
                {/* Ensemble Result */}
                {result.ensemble_mask && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1, delay: 0.8 }}
                    className="relative"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-40 h-40 relative">
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-yellow-500/20 via-orange-500/20 to-red-500/20 rounded-xl blur-lg"
                          animate={{
                            scale: [1, 1.2, 1],
                            rotate: [0, 360],
                            opacity: [0.4, 0.8, 0.4]
                          }}
                          transition={{ duration: 3, repeat: Infinity }}
                        />
                        <div className="relative bg-white dark:bg-gray-900 rounded-xl overflow-hidden border-2 border-yellow-200 dark:border-yellow-800">
                          <Image
                            src={`data:image/png;base64,${result.ensemble_mask}`}
                            alt="Ensemble prediction"
                            fill
                            className="object-cover"
                            sizes="(max-width: 768px) 100vw, 50vw"
                          />
                        </div>
                        
                        {/* Final Result Badge */}
                        <motion.div
                          className="absolute -top-2 -right-2"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ duration: 0.5, delay: 1 }}
                        >
                          <Badge className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white">
                            Final
                          </Badge>
                        </motion.div>
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <motion.div
                            className="w-3 h-3 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full"
                            animate={{ 
                              scale: [1, 1.3, 1],
                              boxShadow: [
                                "0 0 0 0 rgba(251, 191, 36, 0.7)",
                                "0 0 0 10px rgba(251, 191, 36, 0)",
                                "0 0 0 0 rgba(251, 191, 36, 0)"
                              ]
                            }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                          <h4 className="font-bold text-lg">Ensemble Detection</h4>
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                          >
                            <Zap className="w-5 h-5 text-yellow-500" />
                          </motion.div>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">
                          Combined intelligence from all neural networks for maximum accuracy
                        </p>
                        <div className="flex items-center gap-4 text-sm">
                          <div className="flex items-center gap-1">
                            <Activity className="w-4 h-4 text-green-500" />
                            <span>Ensemble: {confidencePercentage}%</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="w-4 h-4 text-gray-500" />
                            <span>Total: {result.total_processing_time?.toFixed(2)}s</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}
