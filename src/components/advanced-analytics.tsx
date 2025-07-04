"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { 
  Brain, 
  Activity, 
  Target, 
  Zap, 
  BarChart3, 
  TrendingUp,
  Gauge,
  Radar,
  Layers,
  Cpu
} from "lucide-react"
import { EnsemblePredictionResult } from "@/types/api"

interface AdvancedAnalyticsProps {
  result: EnsemblePredictionResult
}

export function AdvancedAnalytics({ result }: AdvancedAnalyticsProps) {
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

  const confidencePercentage = Math.round((result.ensemble_confidence || 0) * 100)
  const riskLevel = confidencePercentage > 80 ? 'High' : confidencePercentage > 50 ? 'Medium' : 'Low'
  const riskColor = confidencePercentage > 80 ? 'text-red-600' : confidencePercentage > 50 ? 'text-orange-600' : 'text-green-600'
  const riskBgColor = confidencePercentage > 80 ? 'bg-red-100 border-red-200' : confidencePercentage > 50 ? 'bg-orange-100 border-orange-200' : 'bg-green-100 border-green-200'
  
  const hasOilSpill = confidencePercentage > 50

  // Calculate advanced metrics
  const modelAgreement = result.individual_predictions?.length > 1 ? 
    (result.individual_predictions.filter(p => p.prediction === result.ensemble_prediction).length / result.individual_predictions.length) * 100 : 100

  const avgModelConfidence = result.individual_predictions?.length > 0 ?
    result.individual_predictions.reduce((sum, p) => sum + p.confidence, 0) / result.individual_predictions.length : 0

  const processingEfficiency = result.total_processing_time ? 
    Math.max(0, 100 - (result.total_processing_time * 10)) : 85

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* AI Confidence Meter */}
      <motion.div variants={itemVariants}>
        <Card className="border-2 border-blue-200 dark:border-blue-800 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-950 dark:to-cyan-950">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <Brain className="w-6 h-6 text-blue-600" />
              </motion.div>
              AI Confidence Analysis
            </CardTitle>
            <CardDescription>
              Neural network prediction confidence and reliability metrics
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Main Confidence Gauge */}
            <div className="relative">
              <div className="text-center mb-4">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className="text-4xl font-bold"
                >
                  {confidencePercentage}%
                </motion.div>
                <div className="text-sm text-muted-foreground">Primary Confidence</div>
              </div>
              
              <div className="relative h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidencePercentage}%` }}
                  transition={{ duration: 1.5, ease: "easeOut" }}
                  className={`h-full rounded-full ${
                    confidencePercentage > 80 
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                      : confidencePercentage > 60 
                      ? 'bg-gradient-to-r from-yellow-500 to-orange-500'
                      : 'bg-gradient-to-r from-red-500 to-pink-500'
                  }`}
                />
                <motion.div
                  animate={{ x: [0, 10, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="absolute top-0 left-0 h-full w-2 bg-white/30 rounded-full"
                  style={{ left: `${Math.max(0, confidencePercentage - 2)}%` }}
                />
              </div>
            </div>

            {/* Advanced Metrics Grid */}
            <div className="grid grid-cols-3 gap-4">
              <motion.div 
                variants={itemVariants}
                className="text-center p-3 bg-white/50 dark:bg-black/20 rounded-lg"
              >
                <Gauge className="w-5 h-5 mx-auto mb-2 text-purple-600" />
                <div className="text-2xl font-bold">{modelAgreement.toFixed(0)}%</div>
                <div className="text-xs text-muted-foreground">Model Agreement</div>
              </motion.div>
              
              <motion.div 
                variants={itemVariants}
                className="text-center p-3 bg-white/50 dark:bg-black/20 rounded-lg"
              >
                <Activity className="w-5 h-5 mx-auto mb-2 text-cyan-600" />
                <div className="text-2xl font-bold">{(avgModelConfidence * 100).toFixed(0)}%</div>
                <div className="text-xs text-muted-foreground">Avg Confidence</div>
              </motion.div>
              
              <motion.div 
                variants={itemVariants}
                className="text-center p-3 bg-white/50 dark:bg-black/20 rounded-lg"
              >
                <Zap className="w-5 h-5 mx-auto mb-2 text-yellow-600" />
                <div className="text-2xl font-bold">{processingEfficiency.toFixed(0)}%</div>
                <div className="text-xs text-muted-foreground">Efficiency</div>
              </motion.div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Model Performance Breakdown */}
      <motion.div variants={itemVariants}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-green-600" />
              Neural Network Analysis
            </CardTitle>
            <CardDescription>
              Individual model performance and ensemble intelligence
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {result.individual_predictions?.map((prediction, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                className="border rounded-lg p-4 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${
                      prediction.model_name === 'UNet' ? 'bg-blue-500' : 'bg-purple-500'
                    }`} />
                    <span className="font-medium">{prediction.model_name}</span>
                    <Badge variant={prediction.prediction.includes('Oil Spill') ? 'destructive' : 'secondary'}>
                      {prediction.prediction}
                    </Badge>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{(prediction.confidence * 100).toFixed(1)}%</div>
                    <div className="text-xs text-muted-foreground">
                      {prediction.processing_time.toFixed(2)}s
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Confidence Level</span>
                    <span className={`font-medium ${
                      prediction.confidence > 0.8 ? 'text-green-600' :
                      prediction.confidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {prediction.confidence > 0.8 ? 'High' :
                       prediction.confidence > 0.6 ? 'Medium' : 'Low'}
                    </span>
                  </div>
                  <Progress 
                    value={prediction.confidence * 100} 
                    className="h-2"
                  />
                </div>
              </motion.div>
            ))}
          </CardContent>
        </Card>
      </motion.div>

      {/* Risk Assessment */}
      <motion.div variants={itemVariants}>
        <Card className={`border-2 ${
          hasOilSpill 
            ? 'border-red-200 dark:border-red-800 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-950 dark:to-pink-950'
            : 'border-green-200 dark:border-green-800 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950'
        }`}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className={`w-5 h-5 ${hasOilSpill ? 'text-red-600' : 'text-green-600'}`} />
              Environmental Risk Assessment
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6">
              {/* Risk Level */}
              <div className="space-y-3">
                <div className="text-sm font-medium text-muted-foreground">RISK LEVEL</div>
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                  className="text-center"
                >
                  <div className={`text-3xl font-bold ${
                    hasOilSpill ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {hasOilSpill ? 
                      (confidencePercentage > 80 ? 'HIGH' : 'MODERATE') : 
                      'LOW'
                    }
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Based on AI Analysis
                  </div>
                </motion.div>
              </div>

              {/* Response Urgency */}
              <div className="space-y-3">
                <div className="text-sm font-medium text-muted-foreground">RESPONSE</div>
                <div className="space-y-2">
                  <Badge className={`w-full justify-center py-2 ${
                    hasOilSpill ? 'bg-red-100 text-red-800 hover:bg-red-200' : 'bg-green-100 text-green-800 hover:bg-green-200'
                  }`}>
                    {hasOilSpill ? 
                      (confidencePercentage > 80 ? '🚨 IMMEDIATE ACTION' : '⚠️ MONITOR CLOSELY') :
                      '✅ CONTINUE SURVEILLANCE'
                    }
                  </Badge>
                  <div className="text-xs text-center text-muted-foreground">
                    Recommended Protocol
                  </div>
                </div>
              </div>
            </div>

            <Separator className="my-4" />

            {/* Processing Stats */}
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-lg font-bold text-blue-600">
                  {result.total_processing_time?.toFixed(2)}s
                </div>
                <div className="text-xs text-muted-foreground">Total Time</div>
              </div>
              <div>
                <div className="text-lg font-bold text-purple-600">
                  {result.individual_predictions?.length || 0}
                </div>
                <div className="text-xs text-muted-foreground">Models Used</div>
              </div>
              <div>
                <div className="text-lg font-bold text-green-600">
                  {((result.ensemble_confidence || 0) * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-muted-foreground">Reliability</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}
