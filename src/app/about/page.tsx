"use client"

import { motion } from "framer-motion"
import { useState, useEffect } from "react"
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
  RotateCcw
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ThemeToggle } from "@/components/theme-toggle"

const AboutPage = () => {
  const [currentTechIndex, setCurrentTechIndex] = useState(0)
  const [isAnimating, setIsAnimating] = useState(true)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

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
                    <p className="text-gray-400 text-sm">AI/ML Engineer & Full-Stack Developer</p>
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
