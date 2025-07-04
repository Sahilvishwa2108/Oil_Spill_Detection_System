"use client"

import * as React from "react"
import Image from "next/image"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { 
  ImageIcon, 
  Download, 
  Play, 
  Shuffle,
  Filter,
  Grid3X3,
  List
} from "lucide-react"
import { 
  ALL_TEST_IMAGES, 
  getRandomTestImages,
  getImagesByCategory,
  DIFFICULTY_COLORS, 
  CATEGORY_ICONS,
  CATEGORIES,
  DIFFICULTIES,
  TestImage
} from "@/data/test-images-cloudinary"

interface TestImageGalleryProps {
  onImageSelect: (imageUrl: string, imageName: string) => void
  selectedCategory?: string
  onCategoryChange?: (category: string) => void
}

export function TestImageGallery({ 
  onImageSelect, 
  selectedCategory = "all",
  onCategoryChange 
}: TestImageGalleryProps) {
  const [viewMode, setViewMode] = React.useState<'grid' | 'list'>('grid')
  const [selectedDifficulty, setSelectedDifficulty] = React.useState<string>('all')
  const [shuffledImages, setShuffledImages] = React.useState<TestImage[]>([])
  const [isShuffled, setIsShuffled] = React.useState(false)

  // Use the imported categories and difficulties
  const categories = CATEGORIES
  const difficulties = DIFFICULTIES

  // Use the new filtering functions
  const filteredImages = React.useMemo(() => {
    let images = ALL_TEST_IMAGES
    
    // Filter by category
    if (selectedCategory !== 'all') {
      images = getImagesByCategory(selectedCategory)
    }
    
    // Filter by difficulty
    if (selectedDifficulty !== 'all') {
      images = images.filter(img => img.difficulty === selectedDifficulty)
    }
    
    return images
  }, [selectedCategory, selectedDifficulty])

  // Reset shuffle when filters change
  React.useEffect(() => {
    setIsShuffled(false)
  }, [selectedCategory, selectedDifficulty])

  // Initialize shuffled images
  React.useEffect(() => {
    setShuffledImages(getRandomTestImages(20))
  }, [])

  const handleImageLoad = async (imageUrl: string, imageName: string) => {
    try {
      // Convert URL to File object for the prediction
      const response = await fetch(imageUrl)
      const blob = await response.blob()
      const file = new File([blob], `${imageName}.jpg`, { type: 'image/jpeg' })
      
      // Create URL for preview
      const localUrl = URL.createObjectURL(file)
      onImageSelect(localUrl, imageName)
    } catch (error) {
      console.error('Error loading test image:', error)
      // Fallback to direct URL
      onImageSelect(imageUrl, imageName)
    }
  }

  // Handle shuffle
  const handleShuffle = () => {
    setShuffledImages(getRandomTestImages(20))
    setIsShuffled(true)
  }

  // Use shuffled images when in shuffle mode, otherwise use filtered images
  const displayImages = isShuffled ? shuffledImages : filteredImages

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
        duration: 0.4,
        ease: "easeOut"
      }
    }
  }

  return (
    <Card className="h-full">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <ImageIcon className="w-5 h-5 text-blue-600" />
              </motion.div>
              AI Test Gallery
            </CardTitle>
            <CardDescription>
              Pre-selected test images for AI model evaluation
            </CardDescription>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant={viewMode === 'grid' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('grid')}
            >
              <Grid3X3 className="w-4 h-4" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('list')}
            >
              <List className="w-4 h-4" />
            </Button>
            <Button
              variant={isShuffled ? 'default' : 'outline'}
              size="sm"
              onClick={handleShuffle}
              className="ml-2"
            >
              <Shuffle className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2 pt-4">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Category:</span>
            {categories.map((category) => (
              <Button
                key={category}
                variant={selectedCategory === category ? 'default' : 'outline'}
                size="sm"
                onClick={() => onCategoryChange?.(category)}
                className="h-7 text-xs"
              >
                {category === 'all' ? 'All' : `${CATEGORY_ICONS[category as keyof typeof CATEGORY_ICONS] || '📊'} ${category}`}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <span className="text-sm text-muted-foreground">Difficulty:</span>
          {difficulties.map((difficulty) => (
            <Button
              key={difficulty}
              variant={selectedDifficulty === difficulty ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedDifficulty(difficulty)}
              className="h-7 text-xs"
            >
              {difficulty === 'all' ? 'All' : difficulty}
            </Button>
          ))}
        </div>
        
        {/* Status indicator */}
        <div className="flex items-center justify-between pt-2 border-t">
          <div className="text-sm text-muted-foreground">
            Showing {displayImages.length} of {ALL_TEST_IMAGES.length} images
            {isShuffled && <span className="ml-2 text-blue-600">• Shuffled</span>}
          </div>
          <div className="text-xs text-muted-foreground">
            📊 {ALL_TEST_IMAGES.length} total available
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <ScrollArea className="h-[400px] pr-4">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className={`gap-4 ${
              viewMode === 'grid' 
                ? 'grid grid-cols-2 lg:grid-cols-3' 
                : 'flex flex-col space-y-3'
            }`}
          >
            {displayImages.map((image: TestImage) => (
              <motion.div
                key={image.id}
                variants={itemVariants}
                whileHover={{ scale: viewMode === 'grid' ? 1.02 : 1.01 }}
                whileTap={{ scale: 0.98 }}
                className="cursor-pointer"
                onClick={() => handleImageLoad(image.url, image.name)}
              >
                <Card className="overflow-hidden border-2 hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-200">
                  {viewMode === 'grid' ? (
                    <>
                      <div className="aspect-square relative overflow-hidden">
                        <Image
                          src={image.url}
                          alt={image.name}
                          fill
                          className="object-cover transition-transform duration-300 hover:scale-105"
                          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                          loading="lazy"
                          placeholder="blur"
                          blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkqGx0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
                        />
                        <div className="absolute top-2 left-2">
                          <Badge className={DIFFICULTY_COLORS[image.difficulty as keyof typeof DIFFICULTY_COLORS]}>
                            {image.difficulty}
                          </Badge>
                        </div>
                        <div className="absolute top-2 right-2">
                          <div className="text-lg">
                            {CATEGORY_ICONS[image.category as keyof typeof CATEGORY_ICONS] || '📊'}
                          </div>
                        </div>
                      </div>
                      <CardContent className="p-3">
                        <div className="font-medium text-sm">{image.name}</div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {image.description}
                        </div>
                        <div className="mt-2">
                          <Badge 
                            variant={image.expectedResult.includes('Oil Spill') ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            Expected: {image.expectedResult}
                          </Badge>
                        </div>
                      </CardContent>
                    </>
                  ) : (
                    <CardContent className="p-4">
                      <div className="flex items-center gap-4">
                        <div className="relative w-16 h-16 rounded-lg overflow-hidden">
                          <Image
                            src={image.url}
                            alt={image.name}
                            fill
                            className="object-cover"
                            sizes="64px"
                          />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium">{image.name}</span>
                            <div className="text-sm">
                              {CATEGORY_ICONS[image.category as keyof typeof CATEGORY_ICONS] || '📊'}
                            </div>
                            <Badge className={DIFFICULTY_COLORS[image.difficulty as keyof typeof DIFFICULTY_COLORS]}>
                              {image.difficulty}
                            </Badge>
                          </div>
                          <div className="text-sm text-muted-foreground mb-2">
                            {image.description}
                          </div>
                          <Badge 
                            variant={image.expectedResult.includes('Oil Spill') ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            Expected: {image.expectedResult}
                          </Badge>
                        </div>
                        <Button size="sm" variant="outline">
                          <Play className="w-4 h-4" />
                        </Button>
                      </div>
                    </CardContent>
                  )}
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </ScrollArea>

        {/* Quick Actions */}
        <div className="flex justify-between items-center mt-4 pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            {displayImages.length} test images available
          </div>
          <div className="flex gap-2">
            <Button 
              size="sm" 
              variant="outline"
              onClick={handleShuffle}
            >
              <Shuffle className="w-4 h-4 mr-1" />
              {isShuffled ? 'Shuffle Again' : 'Random Test'}
            </Button>
            <Button size="sm" variant="outline">
              <Download className="w-4 h-4 mr-1" />
              Download All
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
