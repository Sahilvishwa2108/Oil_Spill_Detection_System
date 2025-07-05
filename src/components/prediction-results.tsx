"use client"

import * as React from "react"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { motion, AnimatePresence, cubicBezier } from "framer-motion"
import { 
  Clock, 
  Zap, 
  Target, 
  Activity, 
  Users, 
  Brain, 
  BarChart3, 
  TrendingUp,
  Shield,
  AlertTriangle,
  CheckCircle,
  Sparkles,
  Cpu,
  Eye,
  Layers,
  Binary,
  Radar,
  Satellite,
  Gauge
} from "lucide-react"
import { 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  CartesianGrid, 
  XAxis, 
  YAxis, 
  Tooltip,
  LineChart,
  Line,
  RadialBarChart,
  RadialBar,
  Legend,
  Area,
  AreaChart
} from "recharts"
import { processPredictionData, ProcessedPredictionData } from "@/lib/data-processor"
import { EnsemblePredictionResult } from "@/types/api"

interface PredictionResultsProps {
  result: EnsemblePredictionResult
  originalImage?: string
}

// Advanced color schemes for different visualization types
const CHART_COLORS = {
  primary: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
  gradient: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
  ocean: ['#006994', '#1b9aaa', '#06ffa5', '#ffaa00', '#ff006e'],
  neural: ['#ff006e', '#8338ec', '#3a86ff', '#06ffa5', '#ffbe0b']
}

// Class information matching the notebook exactly
const CLASS_INFO = [
  { name: 'Background', color: '#000000', icon: '🌊', description: 'Clean water surface' },
  { name: 'Oil Spill', color: '#00ffff', icon: '🛢️', description: 'Oil contamination detected' },
  { name: 'Ships', color: '#ff0000', icon: '🚢', description: 'Vessel structures' },
  { name: 'Look-alike', color: '#994c00', icon: '⚠️', description: 'False positive areas' },
  { name: 'Wakes', color: '#009900', icon: '💨', description: 'Ship wake patterns' }
]

export function PredictionResults({ result, originalImage }: PredictionResultsProps) {
  if (!result.success) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="border-red-200 bg-red-50/50">
          <CardHeader>
            <CardTitle className="text-red-700 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Prediction Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-red-600">{result.error}</p>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  // Process data through master processor for CONSISTENCY
  const processedData: ProcessedPredictionData = processPredictionData(result);

  // Advanced animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { y: 30, opacity: 0, scale: 0.95 },
    visible: {
      y: 0,
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.6,
        ease: cubicBezier(0.25, 0.1, 0.25, 1)
      }
    }
  }

  const glowVariants = {
    initial: { boxShadow: "0 0 0 rgba(59, 130, 246, 0)" },
    animate: {
      boxShadow: [
        "0 0 0 rgba(59, 130, 246, 0)",
        "0 0 20px rgba(59, 130, 246, 0.3)",
        "0 0 0 rgba(59, 130, 246, 0)"
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      {/* ===== FUTURISTIC MAIN RESULTS HEADER ===== */}
      <motion.div variants={itemVariants}>
        <Card className={`relative overflow-hidden border-2 ${processedData.finalPrediction === "Oil Spill Detected" 
          ? 'border-red-200 bg-gradient-to-br from-red-50/50 via-orange-50/30 to-red-50/50' 
          : 'border-emerald-200 bg-gradient-to-br from-emerald-50/50 via-green-50/30 to-emerald-50/50'
        }`}>
          {/* Animated background pattern */}
          <div className="absolute inset-0 opacity-5">
            <motion.div
              className="w-full h-full"
              style={{
                backgroundImage: 'radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.3) 0%, transparent 50%), radial-gradient(circle at 80% 50%, rgba(16, 185, 129, 0.3) 0%, transparent 50%)',
              }}
              animate={{
                backgroundPosition: ['0% 0%', '100% 100%', '0% 0%'],
              }}
              transition={{
                duration: 20,
                repeat: Infinity,
                ease: "linear"
              }}
            />
          </div>

          <CardHeader className="relative z-10">
            <CardTitle className="flex items-center gap-3 text-2xl">
              <motion.div
                animate={{ 
                  rotate: [0, 360],
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="relative"
              >
                <Target className="w-8 h-8 text-blue-600" />
                <motion.div
                  className="absolute inset-0 bg-blue-400 rounded-full blur-md opacity-20"
                  animate={{
                    scale: [1, 1.5, 1],
                    opacity: [0.2, 0.4, 0.2]
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              </motion.div>
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI Ensemble Detection Results
              </span>
              <motion.div
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </motion.div>
            </CardTitle>
            <CardDescription className="text-lg">
              Advanced neural network analysis combining {processedData.modelCount} state-of-the-art AI models
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-8 relative z-10">
            {/* Main status display */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Detection Status */}
              <motion.div 
                className="relative overflow-hidden rounded-xl p-6 bg-white/70 backdrop-blur-sm border border-white/50"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex items-center justify-center mb-4">
                  <motion.div
                    className="relative"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    {processedData.finalPrediction === "Oil Spill Detected" ? (
                      <AlertTriangle className="w-12 h-12 text-red-500" />
                    ) : (
                      <CheckCircle className="w-12 h-12 text-green-500" />
                    )}
                    <motion.div
                      className={`absolute inset-0 ${processedData.finalPrediction === "Oil Spill Detected" ? 'bg-red-400' : 'bg-green-400'} rounded-full blur-xl opacity-20`}
                      animate={{
                        scale: [1, 1.3, 1],
                        opacity: [0.2, 0.4, 0.2]
                      }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                  </motion.div>
                </div>
                <div className="text-center">
                  <Badge 
                    variant={processedData.finalPrediction === "Oil Spill Detected" ? "destructive" : "secondary"} 
                    className="text-lg px-4 py-2 font-semibold"
                  >
                    {processedData.finalPrediction}
                  </Badge>
                  <p className="text-sm text-gray-600 mt-2">Final Detection Status</p>
                </div>
              </motion.div>

              {/* Confidence Gauge */}
              <motion.div 
                className="relative overflow-hidden rounded-xl p-6 bg-white/70 backdrop-blur-sm border border-white/50"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2 }}
              >
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 mb-4">
                    <Gauge className="w-5 h-5 text-blue-600" />
                    <span className="font-medium text-gray-700">Ensemble Confidence</span>
                  </div>
                  
                  <motion.div 
                    className="relative w-32 h-32 mx-auto mb-4"
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ duration: 1, delay: 0.3 }}
                  >
                    <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
                      <circle
                        cx="60"
                        cy="60"
                        r="54"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="transparent"
                        className="text-gray-200"
                      />
                      <motion.circle
                        cx="60"
                        cy="60"
                        r="54"
                        stroke="url(#gradient)"
                        strokeWidth="8"
                        fill="transparent"
                        strokeLinecap="round"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: processedData.confidencePercentage / 100 }}
                        transition={{ duration: 1.5, delay: 0.5 }}
                        style={{
                          strokeDasharray: `${54 * 2 * Math.PI}`,
                        }}
                      />
                      <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#3b82f6" />
                          <stop offset="50%" stopColor="#8b5cf6" />
                          <stop offset="100%" stopColor="#ec4899" />
                        </linearGradient>
                      </defs>
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <motion.span 
                        className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5, delay: 1 }}
                      >
                        {processedData.confidencePercentage}%
                      </motion.span>
                    </div>
                  </motion.div>
                  
                  <p className="text-sm text-gray-600">Neural Network Certainty</p>
                </div>
              </motion.div>

              {/* Processing Stats */}
              <motion.div 
                className="relative overflow-hidden rounded-xl p-6 bg-white/70 backdrop-blur-sm border border-white/50"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2 }}
              >
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <Cpu className="w-5 h-5 text-green-600" />
                      <span className="font-medium text-gray-700">Processing Time</span>
                    </div>
                    <motion.div 
                      className="text-2xl font-bold text-green-600"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.4 }}
                    >
                      {processedData.totalProcessingTime.toFixed(2)}s
                    </motion.div>
                  </div>
                  
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <Brain className="w-5 h-5 text-purple-600" />
                      <span className="font-medium text-gray-700">Models Active</span>
                    </div>
                    <motion.div 
                      className="text-2xl font-bold text-purple-600"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.5 }}
                    >
                      {processedData.modelCount}
                    </motion.div>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Risk Assessment */}
            <motion.div
              className="relative overflow-hidden rounded-xl p-4 bg-gradient-to-r from-slate-100 to-slate-50 border border-slate-200"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Shield className="w-6 h-6 text-slate-600" />
                  <span className="font-semibold text-slate-700">Risk Assessment</span>
                </div>
                <Badge className={`${processedData.riskColor} text-sm font-semibold px-3 py-1`}>
                  {processedData.riskLevel} RISK
                </Badge>
              </div>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ===== PREDICTION IMAGES GALLERY ===== */}
      <motion.div variants={itemVariants}>
        <Card className="relative overflow-hidden border-2 border-indigo-200 bg-gradient-to-br from-indigo-50/50 via-purple-50/30 to-pink-50/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-xl">
              <motion.div
                animate={{ 
                  rotateY: [0, 360],
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                <Eye className="w-7 h-7 text-indigo-600" />
              </motion.div>
              <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Neural Network Prediction Gallery
              </span>
              <Layers className="w-6 h-6 text-purple-500" />
            </CardTitle>
            <CardDescription className="text-base">
              Visual comparison of original image and AI-generated segmentation masks
            </CardDescription>
          </CardHeader>

          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
              {/* Original Image */}
              {originalImage && (
                <motion.div
                  className="space-y-4"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.1 }}
                >
                  <div className="relative group">
                    <motion.div
                      className="absolute -inset-1 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000"
                      animate={{
                        scale: [1, 1.02, 1],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                    <div className="relative bg-white rounded-lg p-2 border-2 border-white/50 backdrop-blur-sm">
                      <div className="relative aspect-square rounded-md overflow-hidden">
                        <Image
                          src={originalImage}
                          alt="Original satellite image"
                          fill
                          className="object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent" />
                        <div className="absolute bottom-2 left-2 right-2">
                          <Badge className="bg-blue-600 text-white">
                            <Satellite className="w-3 h-3 mr-1" />
                            Original
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <h4 className="font-semibold text-gray-700">Original Image</h4>
                    <p className="text-sm text-gray-500">Satellite radar data</p>
                  </div>
                </motion.div>
              )}

              {/* UNet Prediction */}
              {result.prediction_images?.unet_predicted && (
                <motion.div
                  className="space-y-4"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                >
                  <div className="relative group">
                    <motion.div
                      className="absolute -inset-1 bg-gradient-to-r from-green-400 to-emerald-400 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000"
                      animate={{
                        scale: [1, 1.02, 1],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                    <div className="relative bg-white rounded-lg p-2 border-2 border-white/50 backdrop-blur-sm">
                      <div className="relative aspect-square rounded-md overflow-hidden">
                        <Image
                          src={`data:image/png;base64,${result.prediction_images.unet_predicted}`}
                          alt="UNet segmentation prediction"
                          fill
                          className="object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent" />
                        <div className="absolute bottom-2 left-2 right-2">
                          <Badge className="bg-green-600 text-white">
                            <Brain className="w-3 h-3 mr-1" />
                            U-Net
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <h4 className="font-semibold text-gray-700">U-Net Segmentation</h4>
                    <p className="text-sm text-gray-500">Encoder-decoder architecture</p>
                  </div>
                </motion.div>
              )}

              {/* DeepLab Prediction */}
              {result.prediction_images?.deeplab_predicted && (
                <motion.div
                  className="space-y-4"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.3 }}
                >
                  <div className="relative group">
                    <motion.div
                      className="absolute -inset-1 bg-gradient-to-r from-purple-400 to-pink-400 rounded-lg blur opacity-25 group-hover:opacity-50 transition duration-1000"
                      animate={{
                        scale: [1, 1.02, 1],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                    <div className="relative bg-white rounded-lg p-2 border-2 border-white/50 backdrop-blur-sm">
                      <div className="relative aspect-square rounded-md overflow-hidden">
                        <Image
                          src={`data:image/png;base64,${result.prediction_images.deeplab_predicted}`}
                          alt="DeepLabV3+ segmentation prediction"
                          fill
                          className="object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent" />
                        <div className="absolute bottom-2 left-2 right-2">
                          <Badge className="bg-purple-600 text-white">
                            <Radar className="w-3 h-3 mr-1" />
                            DeepLab
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <h4 className="font-semibold text-gray-700">DeepLabV3+ Segmentation</h4>
                    <p className="text-sm text-gray-500">Atrous spatial pyramid</p>
                  </div>
                </motion.div>
              )}

              {/* Ensemble Prediction */}
              {result.prediction_images?.ensemble_predicted && (
                <motion.div
                  className="space-y-4"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.6, delay: 0.4 }}
                >
                  <div className="relative group">
                    <motion.div
                      className="absolute -inset-1 bg-gradient-to-r from-yellow-400 via-orange-400 to-red-400 rounded-lg blur opacity-30 group-hover:opacity-60 transition duration-1000"
                      animate={{
                        scale: [1, 1.05, 1],
                        rotate: [0, 1, -1, 0],
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                    <div className="relative bg-white rounded-lg p-2 border-2 border-yellow-200/70 backdrop-blur-sm">
                      <div className="relative aspect-square rounded-md overflow-hidden">
                        <Image
                          src={`data:image/png;base64,${result.prediction_images.ensemble_predicted}`}
                          alt="Ensemble combined prediction"
                          fill
                          className="object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent" />
                        <div className="absolute bottom-2 left-2 right-2">
                          <Badge className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white border-0">
                            <Sparkles className="w-3 h-3 mr-1" />
                            Ensemble
                          </Badge>
                        </div>
                        <motion.div
                          className="absolute top-2 right-2"
                          animate={{
                            scale: [1, 1.2, 1],
                            opacity: [0.7, 1, 0.7]
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        >
                          <div className="w-6 h-6 bg-yellow-400 rounded-full flex items-center justify-center">
                            <span className="text-xs">✨</span>
                          </div>
                        </motion.div>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <h4 className="font-semibold text-gray-700">Ensemble Prediction</h4>
                    <p className="text-sm text-gray-500">Combined AI consensus</p>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Class Legend */}
            <motion.div
              className="mt-8 p-6 bg-white/60 backdrop-blur-sm rounded-xl border border-white/50"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              <h4 className="font-semibold text-lg mb-4 flex items-center gap-2">
                <Binary className="w-5 h-5 text-indigo-600" />
                Segmentation Class Legend
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {CLASS_INFO.map((classInfo, index) => (
                  <motion.div 
                    key={index}
                    className="flex items-center gap-3 p-3 bg-white/70 rounded-lg border border-white/30"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 * index }}
                    whileHover={{ scale: 1.05 }}
                  >
                    <div 
                      className="w-4 h-4 rounded-full border border-gray-300"
                      style={{ backgroundColor: classInfo.color }}
                    />
                    <div>
                      <div className="font-medium text-sm text-gray-700">{classInfo.name}</div>
                      <div className="text-xs text-gray-500">{classInfo.icon}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ===== ADVANCED ANALYTICS & CHARTS ===== */}
      <motion.div variants={itemVariants}>
        <Card className="relative overflow-hidden border-2 border-emerald-200 bg-gradient-to-br from-emerald-50/50 via-green-50/30 to-cyan-50/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-xl">
              <motion.div
                animate={{ 
                  scale: [1, 1.2, 1],
                  rotate: [0, 180, 360]
                }}
                transition={{ 
                  duration: 4, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                <BarChart3 className="w-7 h-7 text-emerald-600" />
              </motion.div>
              <span className="bg-gradient-to-r from-emerald-600 to-cyan-600 bg-clip-text text-transparent">
                Advanced Analytics Dashboard
              </span>
              <TrendingUp className="w-6 h-6 text-cyan-500" />
            </CardTitle>
            <CardDescription className="text-base">
              Comprehensive AI model performance analysis and data visualization
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Model Performance Comparison Chart */}
              <motion.div
                className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/50 shadow-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <h4 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-600" />
                  Model Performance Comparison
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={processedData.chartsData.modelComparison}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis 
                        dataKey="name" 
                        tick={{ fontSize: 12 }}
                        axisLine={{ stroke: '#64748b' }}
                      />
                      <YAxis 
                        tick={{ fontSize: 12 }}
                        axisLine={{ stroke: '#64748b' }}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                        }}
                      />
                      <Bar 
                        dataKey="confidence" 
                        fill="url(#barGradient)" 
                        name="Confidence %" 
                        radius={[4, 4, 0, 0]}
                      />
                      <defs>
                        <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#3b82f6" />
                          <stop offset="100%" stopColor="#8b5cf6" />
                        </linearGradient>
                      </defs>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Class Distribution Pie Chart */}
              <motion.div
                className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/50 shadow-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                <h4 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-emerald-600" />
                  Detection Class Distribution
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={processedData.chartsData.classDistribution}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        innerRadius={40}
                        paddingAngle={2}
                        dataKey="value"
                      >
                        {processedData.chartsData.classDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} stroke="#fff" strokeWidth={2} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value: any) => [`${value}%`, 'Percentage']}
                        contentStyle={{
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #e2e8f0',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                        }}
                      />
                      <Legend 
                        verticalAlign="bottom" 
                        height={36}
                        formatter={(value: string) => (
                          <span style={{ color: '#374151', fontSize: '12px' }}>{value}</span>
                        )}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            </div>

            {/* Individual Model Results Cards */}
            {processedData.individualResults && processedData.individualResults.length > 0 && (
              <motion.div
                className="space-y-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
              >
                <h4 className="font-semibold text-lg flex items-center gap-2">
                  <Users className="w-5 h-5 text-blue-600" />
                  Individual AI Model Analysis
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {processedData.individualResults.map((modelResult, index) => (
                    <motion.div 
                      key={index} 
                      className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/50 shadow-lg hover:shadow-xl transition-all duration-300"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5, delay: 0.1 * index }}
                      whileHover={{ y: -5 }}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                            modelResult.modelName === 'U-Net' 
                              ? 'bg-gradient-to-r from-green-400 to-emerald-500' 
                              : 'bg-gradient-to-r from-purple-400 to-pink-500'
                          }`}>
                            <Brain className="w-6 h-6 text-white" />
                          </div>
                          <div>
                            <h5 className="font-semibold text-gray-800">{modelResult.modelName}</h5>
                            <p className="text-sm text-gray-600">Neural Network</p>
                          </div>
                        </div>
                        <Badge 
                          variant={modelResult.prediction === "Oil Spill Detected" ? "destructive" : "secondary"}
                          className="font-medium"
                        >
                          {modelResult.prediction}
                        </Badge>
                      </div>
                      
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-600">Confidence Level:</span>
                            <span className="font-semibold text-gray-800">{modelResult.confidencePercentage}%</span>
                          </div>
                          <div className="relative">
                            <Progress value={modelResult.confidencePercentage} className="h-3" />
                            <motion.div
                              className="absolute top-0 left-0 h-3 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${modelResult.confidencePercentage}%` }}
                              transition={{ duration: 1, delay: 0.5 + index * 0.2 }}
                            />
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div className="flex items-center gap-2 text-gray-600">
                            <Clock className="w-4 h-4" />
                            <span>Time: {modelResult.processingTime?.toFixed(2)}s</span>
                          </div>
                          <div className="flex items-center gap-2 text-gray-600">
                            <Activity className="w-4 h-4" />
                            <span>Status: Active</span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Oil Spill Analysis Summary */}
            <motion.div
              className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-6 border-2 border-blue-200"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <h4 className="font-semibold text-lg mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-blue-600" />
                Oil Spill Analysis Summary
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-r from-blue-400 to-cyan-500 rounded-full flex items-center justify-center">
                    <span className="text-2xl">🛢️</span>
                  </div>
                  <div className="text-sm text-gray-600 mb-1">Oil Detection</div>
                  <div className="font-bold text-lg text-gray-800">
                    {processedData.oilSpillAnalysis.isDetected ? 'Detected' : 'None'}
                  </div>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full flex items-center justify-center">
                    <span className="text-2xl">📊</span>
                  </div>
                  <div className="text-sm text-gray-600 mb-1">Coverage</div>
                  <div className="font-bold text-lg text-gray-800">
                    {processedData.oilSpillAnalysis.pixelPercentage.toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-3 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center">
                    <span className="text-2xl">⚠️</span>
                  </div>
                  <div className="text-sm text-gray-600 mb-1">Severity</div>
                  <div className="font-bold text-lg text-gray-800">
                    {processedData.oilSpillAnalysis.severity}
                  </div>
                </div>
              </div>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}
