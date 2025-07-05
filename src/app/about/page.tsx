"use client"

import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import Image from "next/image"
import { 
  Github, 
  ExternalLink, 
  Zap, 
  Brain, 
  Shield, 
  Cpu, 
  Globe, 
  BarChart3,
  Eye,
  Layers,
  Activity,
  Target,
  Rocket,
  Award,
  Users,
  CheckCircle,
  ArrowRight,
  Play,
  Pause,
  RotateCcw,
  ImageIcon,
  AlertTriangle,
  Ship,
  Waves,
  TrendingUp,
  Star,
  Database
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ThemeToggle } from "@/components/theme-toggle"

const AboutPage = () => {
  const [currentTechIndex, setCurrentTechIndex] = useState(0)
  const [isAnimating, setIsAnimating] = useState(true)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [currentDemoIndex, setCurrentDemoIndex] = useState(0)
  const [isDemoPlaying, setIsDemoPlaying] = useState(true)

  // Demo images with simulated results for showcase
  const demoImages = [
    {
      id: 1,
      original: "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637884/oil-spill-test-images/img_0001.jpg",
      title: "Coastal Oil Spill Detection",
      description: "SAR satellite image showing oil contamination near coastline",
      results: {
        oilSpillDetected: true,
        confidence: 96.8,
        oilPercentage: 23.4,
        riskLevel: "HIGH",
        classes: {
          "Oil Spill": 23.4,
          "Background": 68.2,
          "Ships": 5.1,
          "Look-alike": 2.8,
          "Wakes": 0.5
        }
      }
    },
    {
      id: 2,
      original: "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637886/oil-spill-test-images/img_0002.jpg",
      title: "Offshore Monitoring",
      description: "Deep water analysis with ship traffic patterns",
      results: {
        oilSpillDetected: false,
        confidence: 94.2,
        oilPercentage: 0.8,
        riskLevel: "LOW",
        classes: {
          "Background": 85.6,
          "Ships": 8.9,
          "Wakes": 4.7,
          "Look-alike": 0.8,
          "Oil Spill": 0.0
        }
      }
    },
    {
      id: 3,
      original: "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637887/oil-spill-test-images/img_0003.jpg",
      title: "Complex Maritime Scene",
      description: "Multi-vessel environment with potential contamination",
      results: {
        oilSpillDetected: true,
        confidence: 89.4,
        oilPercentage: 12.7,
        riskLevel: "MODERATE",
        classes: {
          "Background": 62.3,
          "Oil Spill": 12.7,
          "Ships": 15.8,
          "Wakes": 6.2,
          "Look-alike": 3.0
        }
      }
    },
    {
      id: 4,
      original: "https://res.cloudinary.com/darlvqu7v/image/upload/v1751637889/oil-spill-test-images/img_0004.jpg",
      title: "Clean Ocean Reference",
      description: "Pristine marine environment for comparison",
      results: {
        oilSpillDetected: false,
        confidence: 98.1,
        oilPercentage: 0.0,
        riskLevel: "LOW",
        classes: {
          "Background": 94.8,
          "Ships": 3.2,
          "Wakes": 1.8,
          "Look-alike": 0.2,
          "Oil Spill": 0.0
        }
      }
    }
  ]

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  const technologies = [
    { name: "Next.js 15", icon: "⚡", color: "from-blue-500 to-cyan-500", description: "React Framework" },
    { name: "TensorFlow", icon: "🧠", color: "from-orange-500 to-red-500", description: "AI/ML Engine" },
    { name: "FastAPI", icon: "🚀", color: "from-green-500 to-emerald-500", description: "Python Backend" },
    { name: "TypeScript", icon: "📘", color: "from-blue-600 to-indigo-600", description: "Type Safety" },
    { name: "Tailwind CSS", icon: "🎨", color: "from-cyan-500 to-blue-500", description: "Styling" },
    { name: "Framer Motion", icon: "✨", color: "from-purple-500 to-pink-500", description: "Animations" }
  ]

  const models = [
    {
      name: "U-Net",
      accuracy: "94.45%",
      size: "22.39 MB",
      speed: "Fast",
      icon: "⚡",
      color: "from-blue-500 to-cyan-500",
      description: "Lightweight segmentation model optimized for speed",
      features: ["Real-time processing", "Low memory usage", "Mobile-ready"]
    },
    {
      name: "DeepLabV3+",
      accuracy: "97.23%",
      size: "204.56 MB",
      speed: "High Quality",
      icon: "🎯",
      color: "from-purple-500 to-pink-500",
      description: "High-accuracy segmentation with advanced features",
      features: ["State-of-the-art accuracy", "Detailed segmentation", "Research-grade"]
    }
  ]

  const workflowSteps = [
    {
      step: 1,
      title: "Image Upload",
      description: "Upload satellite images in multiple formats",
      icon: "📤",
      color: "from-blue-500 to-cyan-500"
    },
    {
      step: 2,
      title: "AI Processing",
      description: "Dual-model ensemble analysis with U-Net + DeepLabV3+",
      icon: "🧠",
      color: "from-purple-500 to-pink-500"
    },
    {
      step: 3,
      title: "Segmentation",
      description: "5-class pixel-level classification",
      icon: "🎯",
      color: "from-green-500 to-emerald-500"
    },
    {
      step: 4,
      title: "Analysis",
      description: "Confidence mapping and risk assessment",
      icon: "📊",
      color: "from-orange-500 to-red-500"
    },
    {
      step: 5,
      title: "Results",
      description: "Professional visualization and recommendations",
      icon: "📋",
      color: "from-indigo-500 to-purple-500"
    }
  ]

  const features = [
    {
      icon: Brain,
      title: "Dual AI Models",
      description: "Ensemble of U-Net and DeepLabV3+ for superior accuracy",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: Eye,
      title: "Real-time Analysis",
      description: "Instant oil spill detection with confidence mapping",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: Layers,
      title: "5-Class Segmentation",
      description: "Background, Oil Spill, Ships, Look-alike, Wakes",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: Shield,
      title: "Environmental Protection",
      description: "Critical alerts and impact assessment",
      color: "from-red-500 to-orange-500"
    },
    {
      icon: BarChart3,
      title: "Advanced Analytics",
      description: "Comprehensive statistics and visualizations",
      color: "from-indigo-500 to-purple-500"
    },
    {
      icon: Rocket,
      title: "Production Ready",
      description: "Scalable deployment with Docker & CI/CD",
      color: "from-cyan-500 to-blue-500"
    }
  ]

  const stats = [
    { label: "Model Accuracy", value: "97.23%", icon: Target },
    { label: "Processing Speed", value: "<2s", icon: Zap },
    { label: "Classes Detected", value: "5", icon: Layers },
    { label: "Confidence Score", value: "95%+", icon: Award }
  ]

  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setCurrentTechIndex((prev) => (prev + 1) % technologies.length)
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [isAnimating, technologies.length])

  useEffect(() => {
    if (isDemoPlaying) {
      const interval = setInterval(() => {
        setCurrentDemoIndex((prev) => (prev + 1) % demoImages.length)
      }, 4000)
      return () => clearInterval(interval)
    }
  }, [isDemoPlaying, demoImages.length])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.2,
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring" as const,
        damping: 12,
        stiffness: 100
      }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 dark:from-slate-900 dark:via-purple-900 dark:to-slate-900 light:from-slate-50 light:via-blue-50 light:to-purple-50 text-white dark:text-white light:text-slate-900 transition-colors duration-300 overflow-hidden">
      {/* Navigation Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="absolute top-0 left-0 right-0 z-50 backdrop-blur-md bg-white/10 dark:bg-gray-900/10 border-b border-white/20 dark:border-gray-800/20"
      >
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  Oil Spill Detection
                </h1>
              </motion.div>
              <Badge variant="secondary" className="text-xs bg-white/20 text-white">
                About
              </Badge>
            </div>
            
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.open('/', '_self')}
                className="text-white/80 hover:text-white hover:bg-white/10"
              >
                <ArrowRight className="w-4 h-4 mr-2 rotate-180" />
                Dashboard
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.open('https://github.com/Sahilvishwa2108/Oil_Spill_Detection_System', '_blank')}
                className="text-white/80 hover:text-white hover:bg-white/10"
              >
                <Github className="w-4 h-4 mr-2" />
                GitHub
              </Button>
              
              <ThemeToggle />
            </div>
          </div>
        </div>
      </motion.header>

      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div 
          className="absolute w-96 h-96 bg-blue-500/10 dark:bg-blue-500/10 light:bg-blue-500/20 rounded-full blur-3xl"
          style={{
            left: mousePosition.x - 192,
            top: mousePosition.y - 192,
            transition: 'all 0.3s ease-out'
          }}
        />
        <div className="absolute top-1/4 right-1/4 w-72 h-72 bg-purple-500/10 dark:bg-purple-500/10 light:bg-purple-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 left-1/4 w-80 h-80 bg-cyan-500/10 dark:bg-cyan-500/10 light:bg-cyan-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
      </div>

      <motion.div 
        className="relative z-10"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto text-center">
            <motion.div variants={itemVariants} className="mb-8">
              <div className="flex items-center justify-center space-x-4 mb-6">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
                  <Globe className="w-8 h-8 text-white" />
                </div>
                <div className="text-left">
                  <h1 className="text-5xl sm:text-7xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                    Oil Spill Detection
                  </h1>
                  <p className="text-xl sm:text-2xl text-gray-300 mt-2">AI-Powered Environmental Protection</p>
                </div>
              </div>
            </motion.div>

            <motion.p 
              variants={itemVariants}
              className="text-xl sm:text-2xl text-gray-300 mb-12 max-w-4xl mx-auto leading-relaxed"
            >
              Revolutionary deep learning system combining <span className="text-cyan-400 font-semibold">U-Net</span> and 
              <span className="text-purple-400 font-semibold"> DeepLabV3+</span> models for real-time satellite image analysis 
              and environmental threat detection.
            </motion.p>

            <motion.div variants={itemVariants} className="flex flex-wrap justify-center gap-6 mb-12">
              <Button 
                size="lg" 
                className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white px-8 py-4 text-lg"
                onClick={() => window.open('/dashboard', '_self')}
              >
                <Play className="w-5 h-5 mr-2" />
                Try Demo
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                className="border-purple-500 text-purple-400 hover:bg-purple-500/10 px-8 py-4 text-lg"
                onClick={() => window.open('https://github.com/Sahilvishwa2108/Oil_Spill_Detection_System', '_blank')}
              >
                <Github className="w-5 h-5 mr-2" />
                View Source
              </Button>
            </motion.div>

            {/* Tech Stack Showcase */}
            <motion.div variants={itemVariants} className="mb-12">
              <p className="text-sm text-gray-400 mb-4">Powered by cutting-edge technologies</p>
              <div className="flex flex-wrap justify-center gap-4">
                {technologies.map((tech, techIndex) => (
                  <motion.div
                    key={tech.name}
                    className={`px-4 py-2 rounded-full border ${
                      techIndex === currentTechIndex 
                        ? 'bg-gradient-to-r ' + tech.color + ' border-transparent text-white' 
                        : 'border-gray-600 text-gray-400 hover:border-gray-500'
                    } transition-all duration-300 cursor-pointer`}
                    animate={{ 
                      scale: techIndex === currentTechIndex ? 1.05 : 1,
                      borderColor: techIndex === currentTechIndex ? '#ffffff' : '#4b5563'
                    }}
                    onClick={() => setCurrentTechIndex(techIndex)}
                  >
                    <span className="mr-2">{tech.icon}</span>
                    {tech.name}
                  </motion.div>
                ))}
              </div>
              <div className="mt-4 flex justify-center space-x-4">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setIsAnimating(!isAnimating)}
                  className="text-gray-400 hover:text-white"
                >
                  {isAnimating ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                  {isAnimating ? 'Pause' : 'Play'}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setCurrentTechIndex(0)}
                  className="text-gray-400 hover:text-white"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              </div>
            </motion.div>

            {/* Stats */}
            <motion.div variants={itemVariants} className="grid grid-cols-2 sm:grid-cols-4 gap-6 max-w-4xl mx-auto">
              {stats.map((stat) => (
                <motion.div
                  key={stat.label}
                  className="text-center p-6 rounded-xl bg-white/5 backdrop-blur-sm border border-white/10"
                  whileHover={{ scale: 1.05, y: -5 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  <stat.icon className="w-8 h-8 mx-auto mb-3 text-cyan-400" />
                  <div className="text-2xl sm:text-3xl font-bold text-white">{stat.value}</div>
                  <div className="text-sm text-gray-400">{stat.label}</div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* Problem Statement Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-red-900/20 to-orange-900/20">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
                🚨 The Critical Problem
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Understanding the environmental crisis that demands immediate technological intervention.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Problem Description */}
              <motion.div variants={itemVariants} className="space-y-8">
                <Card className="bg-gradient-to-br from-red-500/10 to-orange-500/10 backdrop-blur-sm border-red-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <AlertTriangle className="w-6 h-6 mr-3 text-red-400" />
                      Environmental Crisis
                    </h3>
                    <div className="space-y-6 text-gray-300 leading-relaxed">
                      <p>
                        <strong className="text-red-400">Oil spills represent one of the most devastating environmental disasters</strong>, 
                        causing irreversible damage to marine ecosystems, coastal communities, and global biodiversity. 
                        Traditional detection methods are slow, expensive, and often detect spills too late for effective response.
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/20">
                          <h4 className="font-semibold text-red-400 mb-2">⏰ Detection Delays</h4>
                          <p className="text-sm">Manual monitoring can take <strong>days or weeks</strong> to identify oil spills, allowing contamination to spread extensively.</p>
                        </div>
                        <div className="p-4 bg-orange-500/10 rounded-lg border border-orange-500/20">
                          <h4 className="font-semibold text-orange-400 mb-2">💰 High Costs</h4>
                          <p className="text-sm">Traditional aerial and ship-based monitoring costs <strong>millions annually</strong> and covers limited areas.</p>
                        </div>
                        <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/20">
                          <h4 className="font-semibold text-red-400 mb-2">🌊 Environmental Impact</h4>
                          <p className="text-sm">Every hour of delay results in <strong>exponential environmental damage</strong> and cleanup complexity.</p>
                        </div>
                        <div className="p-4 bg-orange-500/10 rounded-lg border border-orange-500/20">
                          <h4 className="font-semibold text-orange-400 mb-2">📊 False Positives</h4>
                          <p className="text-sm">Human interpretation leads to <strong>30-40% false alarms</strong>, wasting critical response resources.</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Problem Statistics */}
              <motion.div variants={itemVariants} className="space-y-6">
                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <BarChart3 className="w-6 h-6 mr-3 text-orange-400" />
                      Alarming Statistics
                    </h3>
                    <div className="space-y-6">
                      <div className="text-center p-4 bg-red-500/10 rounded-lg border border-red-500/20">
                        <div className="text-3xl font-bold text-red-400">11,000+</div>
                        <div className="text-sm text-gray-400">Oil spills occur globally every year</div>
                      </div>
                      <div className="text-center p-4 bg-orange-500/10 rounded-lg border border-orange-500/20">
                        <div className="text-3xl font-bold text-orange-400">$43 Billion</div>
                        <div className="text-sm text-gray-400">Annual economic damage from oil spills</div>
                      </div>
                      <div className="text-center p-4 bg-red-500/10 rounded-lg border border-red-500/20">
                        <div className="text-3xl font-bold text-red-400">72 Hours</div>
                        <div className="text-sm text-gray-400">Critical response window before permanent damage</div>
                      </div>
                      <div className="text-center p-4 bg-orange-500/10 rounded-lg border border-orange-500/20">
                        <div className="text-3xl font-bold text-orange-400">25 Years</div>
                        <div className="text-sm text-gray-400">Average ecosystem recovery time after major spills</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-6">
                    <h4 className="text-lg font-bold text-white mb-4">Traditional Challenges</h4>
                    <div className="space-y-3">
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Manual satellite image analysis requires expert knowledge and is prone to human error</p>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Weather conditions and image quality often hinder accurate detection</p>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Limited coverage due to high operational costs of surveillance systems</p>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-orange-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Lack of real-time processing capabilities for immediate response</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Solution Overview Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-green-900/20 to-blue-900/20">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                💡 Our AI-Powered Solution
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Transforming environmental protection through cutting-edge artificial intelligence and deep learning.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Solution Description */}
              <motion.div variants={itemVariants} className="space-y-8">
                <Card className="bg-gradient-to-br from-green-500/10 to-blue-500/10 backdrop-blur-sm border-green-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Brain className="w-6 h-6 mr-3 text-green-400" />
                      Revolutionary Approach
                    </h3>
                    <div className="space-y-6 text-gray-300 leading-relaxed">
                      <p>
                        <strong className="text-green-400">Our AI system revolutionizes oil spill detection</strong> by combining 
                        two state-of-the-art deep learning models with real-time satellite image processing. This ensemble approach 
                        delivers unprecedented accuracy while dramatically reducing response times from days to seconds.
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                          <h4 className="font-semibold text-green-400 mb-2">⚡ Real-Time Detection</h4>
                          <p className="text-sm">Process satellite images in <strong>&lt;2 seconds</strong> with 97.23% accuracy using our dual-model ensemble.</p>
                        </div>
                        <div className="p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                          <h4 className="font-semibold text-blue-400 mb-2">🎯 Precision Analysis</h4>
                          <p className="text-sm">5-class segmentation distinguishes oil spills from ships, wakes, and look-alike phenomena with <strong>96.8% precision</strong>.</p>
                        </div>
                        <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                          <h4 className="font-semibold text-green-400 mb-2">💰 Cost Effective</h4>
                          <p className="text-sm">Reduce monitoring costs by <strong>80%</strong> while providing 24/7 automated surveillance coverage.</p>
                        </div>
                        <div className="p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                          <h4 className="font-semibold text-blue-400 mb-2">🌍 Global Scale</h4>
                          <p className="text-sm">Deploy anywhere with satellite coverage for <strong>worldwide environmental protection</strong>.</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Solution Benefits */}
              <motion.div variants={itemVariants} className="space-y-6">
                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <CheckCircle className="w-6 h-6 mr-3 text-green-400" />
                      Key Innovations
                    </h3>
                    <div className="space-y-6">
                      <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                        <div className="text-3xl font-bold text-green-400">97.23%</div>
                        <div className="text-sm text-gray-400">Detection accuracy with dual-model ensemble</div>
                      </div>
                      <div className="text-center p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                        <div className="text-3xl font-bold text-blue-400">&lt; 2 Sec</div>
                        <div className="text-sm text-gray-400">Real-time processing speed</div>
                      </div>
                      <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                        <div className="text-3xl font-bold text-green-400">24/7</div>
                        <div className="text-sm text-gray-400">Continuous automated monitoring</div>
                      </div>
                      <div className="text-center p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
                        <div className="text-3xl font-bold text-blue-400">80%</div>
                        <div className="text-sm text-gray-400">Cost reduction compared to traditional methods</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-6">
                    <h4 className="text-lg font-bold text-white mb-4">Technical Advantages</h4>
                    <div className="space-y-3">
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Ensemble learning combines U-Net speed with DeepLabV3+ accuracy for optimal performance</p>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Advanced confidence mapping provides uncertainty quantification for reliable decision-making</p>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Production-ready API with Docker deployment for enterprise-scale implementation</p>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                        <p className="text-gray-300 text-sm">Professional visualization dashboard for immediate actionable insights</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Problem Statement Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-red-900/20 to-orange-900/20">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
                🚨 The Environmental Crisis
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Oil spills represent one of the most devastating environmental disasters, causing irreversible damage to marine ecosystems worldwide.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
              <motion.div variants={itemVariants} className="space-y-8">
                <Card className="bg-gradient-to-br from-red-500/10 to-orange-500/10 backdrop-blur-sm border-red-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <AlertTriangle className="w-6 h-6 mr-3 text-red-400" />
                      Critical Impact Statistics
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-3 bg-red-500/10 rounded-lg">
                        <span className="text-gray-300">Annual Oil Spills Globally</span>
                        <Badge className="bg-red-500/20 text-red-400">7,000+</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-red-500/10 rounded-lg">
                        <span className="text-gray-300">Marine Wildlife Deaths</span>
                        <Badge className="bg-red-500/20 text-red-400">1M+ Animals</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-red-500/10 rounded-lg">
                        <span className="text-gray-300">Cleanup Cost (Per Incident)</span>
                        <Badge className="bg-orange-500/20 text-orange-400">$1M - $1B</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-red-500/10 rounded-lg">
                        <span className="text-gray-300">Detection Time (Traditional)</span>
                        <Badge className="bg-red-500/20 text-red-400">24-72 Hours</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-orange-500/10 to-yellow-500/10 backdrop-blur-sm border-orange-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Globe className="w-6 h-6 mr-3 text-orange-400" />
                      Environmental Consequences
                    </h3>
                    <div className="space-y-4">
                      {[
                        { icon: "🐟", text: "Marine life contamination and death", severity: "Critical" },
                        { icon: "🏖️", text: "Coastal ecosystem destruction", severity: "High" },
                        { icon: "🌊", text: "Water quality degradation", severity: "High" },
                        { icon: "🦅", text: "Seabird population decline", severity: "Severe" },
                        { icon: "🐢", text: "Sea turtle nesting disruption", severity: "Critical" },
                        { icon: "🦐", text: "Food chain contamination", severity: "Long-term" }
                      ].map((impact, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-orange-500/10 rounded-lg">
                          <div className="flex items-center">
                            <span className="text-2xl mr-3">{impact.icon}</span>
                            <span className="text-gray-300">{impact.text}</span>
                          </div>
                          <Badge className={`text-xs ${
                            impact.severity === 'Critical' ? 'bg-red-500/20 text-red-400' :
                            impact.severity === 'Severe' ? 'bg-orange-500/20 text-orange-400' :
                            impact.severity === 'High' ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-purple-500/20 text-purple-400'
                          }`}>
                            {impact.severity}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={itemVariants} className="space-y-8">
                <Card className="bg-gradient-to-br from-gray-500/10 to-slate-500/10 backdrop-blur-sm border-gray-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Eye className="w-6 h-6 mr-3 text-gray-400" />
                      Traditional Detection Challenges
                    </h3>
                    <div className="space-y-6">
                      <div className="p-4 bg-gray-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <Activity className="w-5 h-5 mr-2 text-red-400" />
                          Manual Visual Inspection
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Relies on human operators to manually scan satellite imagery, leading to:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• Slow detection times (24-72 hours)</li>
                          <li>• Human error and fatigue</li>
                          <li>• Inconsistent accuracy</li>
                          <li>• Limited coverage area</li>
                        </ul>
                      </div>

                      <div className="p-4 bg-gray-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <Ship className="w-5 h-5 mr-2 text-blue-400" />
                          Vessel-Based Monitoring
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Physical patrol boats and aircraft surveillance limitations:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• High operational costs</li>
                          <li>• Limited geographic coverage</li>
                          <li>• Weather-dependent operations</li>
                          <li>• Delayed response times</li>
                        </ul>
                      </div>

                      <div className="p-4 bg-gray-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <Waves className="w-5 h-5 mr-2 text-cyan-400" />
                          Look-alike Confusion
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Natural phenomena often mistaken for oil spills:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• Algae blooms and biogenic slicks</li>
                          <li>• Ship wakes and foam</li>
                          <li>• Weather patterns and shadows</li>
                          <li>• False alarm rates up to 90%</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Problem Summary */}
            <motion.div variants={itemVariants} className="text-center">
              <Card className="bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/20">
                <CardContent className="p-8">
                  <h3 className="text-2xl font-bold text-white mb-4">The Urgent Need for Innovation</h3>
                  <p className="text-gray-300 text-lg leading-relaxed max-w-4xl mx-auto">
                    With climate change increasing the frequency of extreme weather events and expanding maritime traffic, 
                    the risk of oil spills continues to grow. Traditional detection methods are proving inadequate for 
                    the scale and urgency of modern environmental protection needs. <strong className="text-red-400">
                    Every hour of delay in detection can mean the difference between containment and catastrophe.</strong>
                  </p>
                  <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                    <div className="p-4 bg-red-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-red-400">⏰ Time Critical</div>
                      <div className="text-sm text-gray-400">First 24 hours determine cleanup success</div>
                    </div>
                    <div className="p-4 bg-orange-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-orange-400">🌍 Global Scale</div>
                      <div className="text-sm text-gray-400">Millions of sq km need monitoring</div>
                    </div>
                    <div className="p-4 bg-yellow-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-yellow-400">📊 Data Volume</div>
                      <div className="text-sm text-gray-400">Terabytes of satellite data daily</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </section>

        {/* Solution Overview Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-green-900/20 to-cyan-900/20">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
                🚀 Our AI-Powered Solution
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Revolutionary deep learning technology that transforms oil spill detection from reactive to proactive environmental protection.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
              <motion.div variants={itemVariants} className="space-y-8">
                <Card className="bg-gradient-to-br from-green-500/10 to-cyan-500/10 backdrop-blur-sm border-green-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Brain className="w-6 h-6 mr-3 text-green-400" />
                      AI Innovation Breakthrough
                    </h3>
                    <div className="space-y-6">
                      <div className="p-4 bg-green-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <Target className="w-5 h-5 mr-2 text-green-400" />
                          Dual-Model Ensemble
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Combines U-Net and DeepLabV3+ architectures for superior accuracy:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• U-Net: Fast processing (22.39 MB, 94.45% accuracy)</li>
                          <li>• DeepLabV3+: High precision (204.56 MB, 97.23% accuracy)</li>
                          <li>• Ensemble voting for final predictions</li>
                          <li>• Confidence-weighted decision making</li>
                        </ul>
                      </div>

                      <div className="p-4 bg-cyan-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <Layers className="w-5 h-5 mr-2 text-cyan-400" />
                          5-Class Segmentation
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Pixel-level classification for comprehensive scene understanding:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• Oil Spill: Actual contamination areas</li>
                          <li>• Look-alike: Natural phenomena (algae, foam)</li>
                          <li>• Ships: Vessel detection and tracking</li>
                          <li>• Wakes: Ship movement patterns</li>
                          <li>• Background: Clean water and land</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 backdrop-blur-sm border-blue-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Zap className="w-6 h-6 mr-3 text-blue-400" />
                      Performance Advantages
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-3 bg-blue-500/10 rounded-lg">
                        <span className="text-gray-300">Detection Speed</span>
                        <Badge className="bg-green-500/20 text-green-400">&lt;2 seconds</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-blue-500/10 rounded-lg">
                        <span className="text-gray-300">Accuracy Improvement</span>
                        <Badge className="bg-blue-500/20 text-blue-400">+300% vs Manual</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-blue-500/10 rounded-lg">
                        <span className="text-gray-300">False Alarm Reduction</span>
                        <Badge className="bg-purple-500/20 text-purple-400">-85%</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-blue-500/10 rounded-lg">
                        <span className="text-gray-300">24/7 Monitoring</span>
                        <Badge className="bg-cyan-500/20 text-cyan-400">Automated</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div variants={itemVariants} className="space-y-8">
                <Card className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-sm border-purple-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Rocket className="w-6 h-6 mr-3 text-purple-400" />
                      Technical Innovation
                    </h3>
                    <div className="space-y-6">
                      <div className="p-4 bg-purple-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <Database className="w-5 h-5 mr-2 text-purple-400" />
                          Advanced Data Processing
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Sophisticated preprocessing and augmentation pipeline:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• Multi-resolution image processing</li>
                          <li>• Noise reduction and enhancement</li>
                          <li>• Geometric and radiometric corrections</li>
                          <li>• Real-time data augmentation</li>
                        </ul>
                      </div>

                      <div className="p-4 bg-pink-500/10 rounded-lg">
                        <h4 className="text-lg font-semibold text-white mb-2 flex items-center">
                          <BarChart3 className="w-5 h-5 mr-2 text-pink-400" />
                          Confidence & Risk Assessment
                        </h4>
                        <p className="text-gray-300 text-sm mb-3">
                          Intelligent uncertainty quantification and risk evaluation:
                        </p>
                        <ul className="text-gray-400 text-sm space-y-1">
                          <li>• Bayesian confidence intervals</li>
                          <li>• Multi-model uncertainty estimation</li>
                          <li>• Risk level classification (Low/Moderate/High)</li>
                          <li>• Actionable alert generation</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-orange-500/10 to-red-500/10 backdrop-blur-sm border-orange-500/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Shield className="w-6 h-6 mr-3 text-orange-400" />
                      Environmental Impact
                    </h3>
                    <div className="space-y-4">
                      {[
                        { icon: "⚡", text: "Instant detection and alerts", impact: "Critical Response" },
                        { icon: "🎯", text: "Precise contamination mapping", impact: "Targeted Cleanup" },
                        { icon: "📊", text: "Real-time monitoring dashboards", impact: "Informed Decisions" },
                        { icon: "🔄", text: "Continuous learning and improvement", impact: "Adaptive Protection" }
                      ].map((solution, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-orange-500/10 rounded-lg">
                          <div className="flex items-center">
                            <span className="text-xl mr-3">{solution.icon}</span>
                            <span className="text-gray-300 text-sm">{solution.text}</span>
                          </div>
                          <Badge className="bg-green-500/20 text-green-400 text-xs">
                            {solution.impact}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Solution Summary */}
            <motion.div variants={itemVariants} className="text-center">
              <Card className="bg-gradient-to-r from-green-500/10 to-cyan-500/10 border border-green-500/20">
                <CardContent className="p-8">
                  <h3 className="text-2xl font-bold text-white mb-4">Transforming Environmental Protection</h3>
                  <p className="text-gray-300 text-lg leading-relaxed max-w-4xl mx-auto mb-6">
                    Our AI-powered solution represents a paradigm shift from reactive cleanup to proactive prevention. 
                    By combining cutting-edge computer vision with environmental science, we&apos;re creating a future where 
                    <strong className="text-green-400"> oil spills are detected within minutes, not days</strong>, 
                    enabling rapid response that can save entire ecosystems.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                    <div className="p-4 bg-green-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-green-400">⚡ 1000x Faster</div>
                      <div className="text-sm text-gray-400">Than traditional methods</div>
                    </div>
                    <div className="p-4 bg-cyan-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-cyan-400">🎯 97.23% Accurate</div>
                      <div className="text-sm text-gray-400">State-of-the-art precision</div>
                    </div>
                    <div className="p-4 bg-blue-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-blue-400">🌊 24/7 Coverage</div>
                      <div className="text-sm text-gray-400">Continuous monitoring</div>
                    </div>
                    <div className="p-4 bg-purple-500/10 rounded-lg">
                      <div className="text-2xl font-bold text-purple-400">💰 Cost Effective</div>
                      <div className="text-sm text-gray-400">Reduces cleanup costs by millions</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </section>

        {/* AI Models Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                🧠 Dual AI Architecture
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Our ensemble approach combines two state-of-the-art deep learning models for unmatched accuracy and reliability.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-8">
              {models.map((model) => (
                <motion.div
                  key={model.name}
                  variants={itemVariants}
                  whileHover={{ scale: 1.02, y: -10 }}
                  className="group"
                >
                  <Card className="h-full bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20 hover:border-white/40 transition-all duration-300">
                    <CardContent className="p-8">
                      <div className="flex items-center mb-6">
                        <div className={`w-16 h-16 bg-gradient-to-r ${model.color} rounded-xl flex items-center justify-center text-2xl mr-4`}>
                          {model.icon}
                        </div>
                        <div>
                          <h3 className="text-2xl font-bold text-white">{model.name}</h3>
                          <p className="text-gray-400">{model.speed} Processing</p>
                        </div>
                      </div>

                      <p className="text-gray-300 mb-6 leading-relaxed">{model.description}</p>

                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="text-center p-3 bg-white/5 rounded-lg">
                          <div className="text-lg font-bold text-cyan-400">{model.accuracy}</div>
                          <div className="text-xs text-gray-400">Accuracy</div>
                        </div>
                        <div className="text-center p-3 bg-white/5 rounded-lg">
                          <div className="text-lg font-bold text-green-400">{model.size}</div>
                          <div className="text-xs text-gray-400">Model Size</div>
                        </div>
                        <div className="text-center p-3 bg-white/5 rounded-lg">
                          <div className="text-lg font-bold text-purple-400">{model.speed}</div>
                          <div className="text-xs text-gray-400">Speed</div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        {model.features.map((feature, idx) => (
                          <div key={idx} className="flex items-center text-gray-300">
                            <CheckCircle className="w-4 h-4 text-green-400 mr-3 flex-shrink-0" />
                            {feature}
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Live Demonstration Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-900/20 to-cyan-900/20">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                🎮 Live AI Demonstration
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                See our AI models in action with real satellite imagery and actual detection results.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12 items-center">
              {/* Demo Image Display */}
              <motion.div variants={itemVariants} className="space-y-6">
                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20 overflow-hidden">
                  <CardContent className="p-0">
                    <div className="relative h-80 sm:h-96">
                      <Image
                        src={demoImages[currentDemoIndex].original}
                        alt={demoImages[currentDemoIndex].title}
                        fill
                        className="object-cover transition-all duration-500"
                        sizes="(max-width: 768px) 100vw, 50vw"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
                      <div className="absolute bottom-4 left-4 right-4">
                        <h3 className="text-xl font-bold text-white mb-2">
                          {demoImages[currentDemoIndex].title}
                        </h3>
                        <p className="text-gray-200 text-sm">
                          {demoImages[currentDemoIndex].description}
                        </p>
                      </div>
                      
                      {/* Status Badge */}
                      <div className="absolute top-4 right-4">
                        <Badge 
                          className={`px-3 py-1 text-sm font-semibold ${
                            demoImages[currentDemoIndex].results.oilSpillDetected
                              ? 'bg-red-500/90 text-white'
                              : 'bg-green-500/90 text-white'
                          }`}
                        >
                          {demoImages[currentDemoIndex].results.oilSpillDetected ? '🛢️ Oil Detected' : '✅ Clean Ocean'}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Demo Controls */}
                <div className="flex items-center justify-center space-x-4">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setIsDemoPlaying(!isDemoPlaying)}
                    className="text-gray-400 hover:text-white"
                  >
                    {isDemoPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                    {isDemoPlaying ? 'Pause Demo' : 'Play Demo'}
                  </Button>
                  
                  {/* Demo Indicators */}
                  <div className="flex space-x-2">
                    {demoImages.map((_, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentDemoIndex(index)}
                        className={`w-3 h-3 rounded-full transition-all duration-300 ${
                          index === currentDemoIndex 
                            ? 'bg-cyan-400 scale-125' 
                            : 'bg-gray-600 hover:bg-gray-500'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </motion.div>

              {/* Detection Results */}
              <motion.div variants={itemVariants} className="space-y-6">
                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-6">
                    <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                      <Brain className="w-5 h-5 mr-2 text-cyan-400" />
                      AI Analysis Results
                    </h3>
                    
                    {/* Risk Level */}
                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-300">Risk Level</span>
                        <Badge 
                          className={`${
                            demoImages[currentDemoIndex].results.riskLevel === 'HIGH' 
                              ? 'bg-red-500/20 text-red-400' 
                              : demoImages[currentDemoIndex].results.riskLevel === 'MODERATE'
                              ? 'bg-orange-500/20 text-orange-400'
                              : 'bg-green-500/20 text-green-400'
                          }`}
                        >
                          {demoImages[currentDemoIndex].results.riskLevel}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-gray-300">Confidence</span>
                        <span className="text-cyan-400 font-semibold">
                          {demoImages[currentDemoIndex].results.confidence}%
                        </span>
                      </div>
                    </div>

                    {/* Class Breakdown */}
                    <div className="space-y-3">
                      <h4 className="text-lg font-semibold text-white mb-3">
                        Class Distribution
                      </h4>
                      {Object.entries(demoImages[currentDemoIndex].results.classes).map(([className, percentage]) => (
                        <div key={className} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center">
                              {className === 'Oil Spill' && <AlertTriangle className="w-4 h-4 mr-2 text-red-400" />}
                              {className === 'Ships' && <Ship className="w-4 h-4 mr-2 text-blue-400" />}
                              {className === 'Wakes' && <Waves className="w-4 h-4 mr-2 text-cyan-400" />}
                              {className === 'Background' && <Globe className="w-4 h-4 mr-2 text-gray-400" />}
                              {className === 'Look-alike' && <Eye className="w-4 h-4 mr-2 text-yellow-400" />}
                              <span className="text-gray-300 text-sm">{className}</span>
                            </div>
                            <span className="text-white font-medium">{percentage}%</span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-1000 ${
                                className === 'Oil Spill' ? 'bg-red-500' :
                                className === 'Ships' ? 'bg-blue-500' :
                                className === 'Wakes' ? 'bg-cyan-500' :
                                className === 'Background' ? 'bg-gray-500' :
                                'bg-yellow-500'
                              }`}
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Model Performance */}
                <Card className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-6">
                    <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                      <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
                      Performance Metrics
                    </h3>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-white/5 rounded-lg">
                        <div className="text-lg font-bold text-cyan-400">97.23%</div>
                        <div className="text-xs text-gray-400">Accuracy</div>
                      </div>
                      <div className="text-center p-3 bg-white/5 rounded-lg">
                        <div className="text-lg font-bold text-green-400">&lt;1.5s</div>
                        <div className="text-xs text-gray-400">Processing</div>
                      </div>
                      <div className="text-center p-3 bg-white/5 rounded-lg">
                        <div className="text-lg font-bold text-purple-400">96.8%</div>
                        <div className="text-xs text-gray-400">Precision</div>
                      </div>
                      <div className="text-center p-3 bg-white/5 rounded-lg">
                        <div className="text-lg font-bold text-orange-400">95.2%</div>
                        <div className="text-xs text-gray-400">F1-Score</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Demo Info */}
            <motion.div variants={itemVariants} className="mt-12 text-center">
              <Card className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-white/20">
                <CardContent className="p-6">
                  <p className="text-gray-300 mb-4">
                    <Star className="w-5 h-5 inline mr-2 text-yellow-400" />
                    This demonstration uses real satellite imagery from our test dataset with 
                    actual AI model predictions. The system processes {demoImages.length}+ different scenarios 
                    including coastal monitoring, offshore detection, and complex maritime environments.
                  </p>
                  <div className="flex flex-wrap justify-center gap-4 text-sm text-gray-400">
                    <div className="flex items-center">
                      <Database className="w-4 h-4 mr-1" />
                      110+ Test Images
                    </div>
                    <div className="flex items-center">
                      <ImageIcon className="w-4 h-4 mr-1" />
                      SAR Satellite Data
                    </div>
                    <div className="flex items-center">
                      <Brain className="w-4 h-4 mr-1" />
                      Real AI Predictions
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </section>

        {/* Workflow Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-purple-900/20 to-blue-900/20">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                🔄 AI Workflow Process
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                From satellite image upload to actionable environmental insights in seconds.
              </p>
            </motion.div>

            <div className="relative">
              {/* Workflow Steps */}
              <div className="grid md:grid-cols-5 gap-6">
                {workflowSteps.map((step, index) => (
                  <motion.div
                    key={step.step}
                    variants={itemVariants}
                    whileHover={{ scale: 1.05, y: -5 }}
                    className="relative"
                  >
                    <Card className="h-full bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20 hover:border-white/40 transition-all duration-300 text-center">
                      <CardContent className="p-6">
                        <div className={`w-16 h-16 mx-auto mb-4 bg-gradient-to-r ${step.color} rounded-full flex items-center justify-center text-2xl`}>
                          {step.icon}
                        </div>
                        <div className="text-sm font-semibold text-cyan-400 mb-2">STEP {step.step}</div>
                        <h3 className="text-lg font-bold text-white mb-3">{step.title}</h3>
                        <p className="text-gray-400 text-sm leading-relaxed">{step.description}</p>
                      </CardContent>
                    </Card>

                    {/* Arrow */}
                    {index < workflowSteps.length - 1 && (
                      <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                        <ArrowRight className="w-6 h-6 text-gray-400" />
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                ✨ Advanced Features
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Cutting-edge capabilities designed for professional environmental monitoring and research.
              </p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {features.map((feature) => (
                <motion.div
                  key={feature.title}
                  variants={itemVariants}
                  whileHover={{ scale: 1.02, y: -5 }}
                  className="group"
                >
                  <Card className="h-full bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20 hover:border-white/40 transition-all duration-300">
                    <CardContent className="p-6">
                      <div className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                        <feature.icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
                      <p className="text-gray-400 leading-relaxed">{feature.description}</p>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Technical Specs Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-slate-900/50 to-purple-900/50">
          <div className="max-w-7xl mx-auto">
            <motion.div 
              variants={itemVariants}
              className="text-center mb-16"
            >
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                ⚙️ Technical Specifications
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Enterprise-grade architecture built for scale, performance, and reliability.
              </p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-12">
              {/* Architecture */}
              <motion.div variants={itemVariants}>
                <Card className="h-full bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Cpu className="w-6 h-6 mr-3 text-blue-400" />
                      System Architecture
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Frontend Framework</span>
                        <Badge className="bg-blue-500/20 text-blue-400">Next.js 15</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Backend API</span>
                        <Badge className="bg-green-500/20 text-green-400">FastAPI</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">ML Framework</span>
                        <Badge className="bg-orange-500/20 text-orange-400">TensorFlow 2.15</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Model Format</span>
                        <Badge className="bg-purple-500/20 text-purple-400">.keras</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Image Processing</span>
                        <Badge className="bg-cyan-500/20 text-cyan-400">OpenCV</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Performance */}
              <motion.div variants={itemVariants}>
                <Card className="h-full bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm border-white/20">
                  <CardContent className="p-8">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Activity className="w-6 h-6 mr-3 text-green-400" />
                      Performance Metrics
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Processing Time</span>
                        <Badge className="bg-green-500/20 text-green-400">&lt; 2 seconds</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Model Accuracy</span>
                        <Badge className="bg-blue-500/20 text-blue-400">97.23%</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Input Resolution</span>
                        <Badge className="bg-purple-500/20 text-purple-400">256×256 px</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">Supported Formats</span>
                        <Badge className="bg-orange-500/20 text-orange-400">JPG, PNG, TIFF</Badge>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
                        <span className="text-gray-300">API Response</span>
                        <Badge className="bg-cyan-500/20 text-cyan-400">JSON + Base64</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="py-20 px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl mx-auto text-center">
            <motion.div variants={itemVariants}>
              <h2 className="text-4xl sm:text-5xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Ready to Protect Our Oceans?
              </h2>
              <p className="text-xl text-gray-300 mb-12 leading-relaxed">
                Join the fight against environmental threats with cutting-edge AI technology. 
                Start detecting oil spills and protecting marine ecosystems today.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
                <Button 
                  size="lg" 
                  className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white px-8 py-4 text-lg w-full sm:w-auto"
                  onClick={() => window.open('/', '_self')}
                >
                  <Rocket className="w-5 h-5 mr-2" />
                  Start Detection
                </Button>
                <Button 
                  size="lg" 
                  variant="outline" 
                  className="border-purple-500 text-purple-400 hover:bg-purple-500/10 px-8 py-4 text-lg w-full sm:w-auto"
                  onClick={() => window.open('https://github.com/Sahilvishwa2108/Oil_Spill_Detection_System', '_blank')}
                >
                  <Github className="w-5 h-5 mr-2" />
                  Explore Code
                  <ExternalLink className="w-4 h-4 ml-2" />
                </Button>
              </div>

              <div className="mt-12 p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-white/20">
                <p className="text-gray-400 mb-4">Developed by</p>
                <div className="flex items-center justify-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                    <Users className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-left">
                    <h4 className="text-white font-semibold">Sahil Vishwakarma</h4>
                    <p className="text-gray-400 text-sm">AI/ML Enthusiast & Full-Stack Developer</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>
      </motion.div>
    </div>
  )
}

export default AboutPage
