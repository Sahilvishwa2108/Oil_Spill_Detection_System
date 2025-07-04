"use client"

import * as React from "react"
import Image from "next/image"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  ChevronLeft, 
  ChevronRight, 
  Play, 
  Download, 
  Sparkles,
  ImageIcon,
  Zap,
  Target,
  Layers
} from "lucide-react"
import { 
  ALL_TEST_IMAGES,
  DIFFICULTY_COLORS,
  CATEGORY_ICONS,
  TestImage
} from "@/data/test-images-cloudinary"

interface ImageDockProps {
  onImageSelect: (imageUrl: string, imageName: string) => void
  onPredictionTabActivate?: () => void
}

export function ImageDock({ onImageSelect, onPredictionTabActivate }: ImageDockProps) {
  const [currentIndex, setCurrentIndex] = React.useState(0)
  const [selectedImageIndex, setSelectedImageIndex] = React.useState<number | null>(null)
  const [isProcessing, setIsProcessing] = React.useState(false)

  const itemsPerView = 5
  const totalItems = ALL_TEST_IMAGES.length
  const maxIndex = Math.max(0, totalItems - itemsPerView)

  const handlePrevious = () => {
    setCurrentIndex(prev => Math.max(0, prev - 1))
  }

  const handleNext = () => {
    setCurrentIndex(prev => Math.min(maxIndex, prev + 1))
  }

  const handleImageClick = async (image: TestImage, index: number) => {
    setSelectedImageIndex(index)
    setIsProcessing(true)
    
    try {
      await onImageSelect(image.url, image.name)
      
      // Automatically switch to prediction tab after a short delay
      setTimeout(() => {
        onPredictionTabActivate?.()
      }, 500)
    } catch (error) {
      console.error('Error selecting image:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleDownload = async (image: TestImage) => {
    try {
      const response = await fetch(image.url)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${image.name}.jpg`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Download failed:', error)
    }
  }

  const visibleImages = ALL_TEST_IMAGES.slice(currentIndex, currentIndex + itemsPerView)

  return (
    <div className="relative">
      {/* Futuristic Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6 text-center"
      >
        <div className="flex items-center justify-center gap-2 mb-2">
          <motion.div
            animate={{ 
              rotate: 360,
              scale: [1, 1.2, 1]
            }}
            transition={{ 
              rotate: { duration: 3, repeat: Infinity, ease: "linear" },
              scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
            }}
          >
            <Sparkles className="w-6 h-6 text-cyan-500" />
          </motion.div>
          <h3 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
            AI Test Gallery
          </h3>
          <motion.div
            animate={{ 
              rotate: -360,
              scale: [1, 1.2, 1]
            }}
            transition={{ 
              rotate: { duration: 3, repeat: Infinity, ease: "linear" },
              scale: { duration: 2, repeat: Infinity, ease: "easeInOut", delay: 1 }
            }}
          >
            <Sparkles className="w-6 h-6 text-purple-500" />
          </motion.div>
        </div>
        <p className="text-sm text-muted-foreground">
          Neural network training dataset • Click to analyze with AI
        </p>
      </motion.div>

      {/* Main Dock Container */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="relative"
      >
        <Card className="bg-gradient-to-r from-slate-50 via-blue-50 to-cyan-50 dark:from-slate-900 dark:via-blue-950 dark:to-cyan-950 border-2 border-blue-200 dark:border-blue-800 overflow-hidden">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-5 h-5 text-blue-600" />
                <span className="text-sm font-medium">
                  {currentIndex + 1}-{Math.min(currentIndex + itemsPerView, totalItems)} of {totalItems}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePrevious}
                  disabled={currentIndex === 0}
                  className="h-8 w-8 p-0"
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleNext}
                  disabled={currentIndex >= maxIndex}
                  className="h-8 w-8 p-0"
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Image Dock */}
            <div className="relative overflow-hidden rounded-xl">
              <motion.div 
                className="flex gap-4 p-4 bg-white/50 dark:bg-black/20 rounded-xl backdrop-blur-sm"
                animate={{ x: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              >
                <AnimatePresence mode="wait">
                  {visibleImages.map((image, index) => {
                    const globalIndex = currentIndex + index
                    const isSelected = selectedImageIndex === globalIndex
                    const isActive = isProcessing && isSelected
                    
                    return (
                      <motion.div
                        key={image.id}
                        initial={{ opacity: 0, scale: 0.8, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.8, y: -20 }}
                        transition={{ duration: 0.4, delay: index * 0.1 }}
                        className="relative group cursor-pointer"
                        onClick={() => handleImageClick(image, globalIndex)}
                      >
                        {/* Magical Background Effect */}
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-xl blur-lg"
                          animate={{
                            scale: isActive ? [1, 1.1, 1] : 1,
                            opacity: isActive ? [0.3, 0.6, 0.3] : 0
                          }}
                          transition={{ duration: 1.5, repeat: isActive ? Infinity : 0 }}
                        />
                        
                        {/* Image Container */}
                        <motion.div
                          className={`relative w-40 h-40 rounded-xl overflow-hidden border-2 ${
                            isSelected 
                              ? 'border-cyan-400 dark:border-cyan-600' 
                              : 'border-transparent group-hover:border-blue-300 dark:group-hover:border-blue-700'
                          }`}
                          whileHover={{ scale: 1.05, y: -5 }}
                          whileTap={{ scale: 0.95 }}
                          animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                          transition={{ 
                            hover: { duration: 0.2 },
                            tap: { duration: 0.1 },
                            scale: { duration: 1, repeat: isActive ? Infinity : 0 }
                          }}
                        >
                          <Image
                            src={image.url}
                            alt={image.name}
                            fill
                            className="object-cover transition-all duration-300 group-hover:scale-105"
                            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                            priority={index < 3}
                            loading={index < 3 ? "eager" : "lazy"}
                            placeholder="blur"
                            blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
                          />
                          
                          {/* Gradient Overlay */}
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                          
                          {/* Badges */}
                          <div className="absolute top-2 left-2 flex gap-1">
                            <Badge className={`${DIFFICULTY_COLORS[image.difficulty as keyof typeof DIFFICULTY_COLORS]} text-xs`}>
                              {image.difficulty}
                            </Badge>
                          </div>
                          
                          <div className="absolute top-2 right-2">
                            <div className="text-lg">
                              {CATEGORY_ICONS[image.category as keyof typeof CATEGORY_ICONS] || '📊'}
                            </div>
                          </div>
                          
                          {/* Processing Indicator */}
                          {isActive && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full"
                              />
                            </div>
                          )}
                          
                          {/* Hover Controls */}
                          <motion.div
                            className="absolute bottom-2 left-2 right-2 flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                            initial={{ y: 10 }}
                            animate={{ y: 0 }}
                          >
                            <Button
                              size="sm"
                              className="h-8 px-3 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 text-white border-0"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleImageClick(image, globalIndex)
                              }}
                            >
                              <Play className="w-3 h-3 mr-1" />
                              Analyze
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-8 w-8 p-0 bg-white/90 hover:bg-white"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleDownload(image)
                              }}
                            >
                              <Download className="w-3 h-3" />
                            </Button>
                          </motion.div>
                        </motion.div>
                        
                        {/* Image Info */}
                        <motion.div
                          className="mt-3 text-center"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.1 * index }}
                        >
                          <div className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-1">
                            {image.name}
                          </div>
                          <div className="text-xs text-muted-foreground mb-2">
                            {image.description}
                          </div>
                          <Badge 
                            variant={image.expectedResult.includes('Oil Spill') ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            {image.expectedResult}
                          </Badge>
                        </motion.div>
                      </motion.div>
                    )
                  })}
                </AnimatePresence>
              </motion.div>
            </div>

            {/* AI Enhancement Indicators */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-4 flex items-center justify-center gap-4 text-xs text-muted-foreground"
            >
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3 text-yellow-500" />
                <span>Neural Processing</span>
              </div>
              <div className="flex items-center gap-1">
                <Target className="w-3 h-3 text-green-500" />
                <span>Ensemble Detection</span>
              </div>
              <div className="flex items-center gap-1">
                <Layers className="w-3 h-3 text-blue-500" />
                <span>Multi-Model Analysis</span>
              </div>
            </motion.div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Futuristic Navigation Dots */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="flex justify-center mt-4 gap-2"
      >
        {Array.from({ length: Math.ceil(totalItems / itemsPerView) }, (_, i) => (
          <motion.button
            key={i}
            className={`w-2 h-2 rounded-full transition-all duration-300 ${
              Math.floor(currentIndex / itemsPerView) === i
                ? 'bg-cyan-500 scale-125'
                : 'bg-gray-300 dark:bg-gray-700 hover:bg-gray-400 dark:hover:bg-gray-600'
            }`}
            whileHover={{ scale: 1.2 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setCurrentIndex(i * itemsPerView)}
          />
        ))}
      </motion.div>
    </div>
  )
}
