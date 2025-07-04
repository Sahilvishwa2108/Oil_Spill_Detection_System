"use client"

import * as React from "react"
import { motion } from "framer-motion"

// Lazy loading wrapper with fallback
interface LazyWrapperProps {
  children: React.ReactNode
  fallback?: React.ReactNode
  className?: string
}

export function LazyWrapper({ children, fallback, className }: LazyWrapperProps) {
  const [isVisible, setIsVisible] = React.useState(false)
  const ref = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          observer.disconnect()
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px'
      }
    )

    if (ref.current) {
      observer.observe(ref.current)
    }

    return () => observer.disconnect()
  }, [])

  return (
    <div ref={ref} className={className}>
      {isVisible ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {children}
        </motion.div>
      ) : (
        fallback || (
          <div className="lazy-loading h-32 rounded-lg animate-pulse">
            <div className="h-full bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 rounded-lg"></div>
          </div>
        )
      )}
    </div>
  )
}

// Performance-optimized component wrapper
interface PerformanceWrapperProps {
  children: React.ReactNode
  threshold?: number
  enableGpu?: boolean
}

export function PerformanceWrapper({ 
  children, 
  threshold = 0.1, 
  enableGpu = true 
}: PerformanceWrapperProps) {
  const [shouldRender, setShouldRender] = React.useState(false)
  const ref = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setShouldRender(entry.isIntersecting)
      },
      { threshold }
    )

    if (ref.current) {
      observer.observe(ref.current)
    }

    return () => observer.disconnect()
  }, [threshold])

  return (
    <div 
      ref={ref} 
      style={{
        transform: enableGpu ? 'translateZ(0)' : undefined,
        willChange: shouldRender ? 'auto' : 'transform'
      }}
    >
      {shouldRender && children}
    </div>
  )
}
