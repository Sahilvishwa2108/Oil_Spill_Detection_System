"use client"

import * as React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { motion } from "framer-motion"
import { Clock, Zap, Target, Activity, Users, BarChart3, TrendingUp, AlertTriangle, Brain } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from "recharts"
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
  const modelComparisonData = result.individual_predictions?.map((pred, index) => ({
    name: pred.model_name,
    confidence: Math.round(pred.confidence * 100),
    time: pred.processing_time,
    prediction: pred.prediction
  })) || []

  const confidenceDistributionData = [
    { name: 'Oil Spill', value: confidencePercentage, color: COLORS[3] },
    { name: 'Clean Water', value: 100 - confidencePercentage, color: COLORS[1] }
  ]

  const processingTimeData = result.individual_predictions?.map((pred, index) => ({
    name: pred.model_name,
    time: pred.processing_time * 1000, // Convert to milliseconds for better visualization
    efficiency: Math.max(0, 100 - (pred.processing_time * 50))
  })) || []

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

      {/* Image Comparison */}
      <motion.div variants={itemVariants}>
        <Card>
          <CardHeader>
            <CardTitle>Image Analysis</CardTitle>
            <CardDescription>
              Original image and AI-generated detection masks
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {originalImage && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="space-y-3">
                    <h4 className="font-medium text-center">Original Image</h4>
                    <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={originalImage}
                        alt="Original"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                </motion.div>
              )}
              
              {/* Individual Model Prediction Masks */}
              {result.individual_predictions?.map((modelResult, index) => (
                modelResult.prediction_mask && (
                  <motion.div
                    key={`mask-${index}`}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.1 * (index + 1) }}
                  >
                    <div className="space-y-3">
                      <h4 className="font-medium text-center">{modelResult.model_name} Detection</h4>
                      <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={`data:image/png;base64,${modelResult.prediction_mask}`}
                          alt={`${modelResult.model_name} prediction mask`}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <p className="text-xs text-muted-foreground text-center">
                        Highlighted areas show detected oil spills
                      </p>
                    </div>
                  </motion.div>
                )
              ))}

              {/* Ensemble Prediction Mask */}
              {result.ensemble_mask && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <div className="space-y-3">
                    <h4 className="font-medium text-center flex items-center justify-center gap-2">
                      <Zap className="w-4 h-4" />
                      Ensemble Detection
                    </h4>
                    <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`data:image/png;base64,${result.ensemble_mask}`}
                        alt="Ensemble prediction mask"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground text-center">
                      Combined detection from all models
                    </p>
                  </div>
                </motion.div>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}
