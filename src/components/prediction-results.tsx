"use client"

import * as React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Clock, Zap, Target, Activity } from "lucide-react"
import { PredictionResult } from "@/types/api"

interface PredictionResultsProps {
  result: PredictionResult
  originalImage?: string
}

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

  const confidencePercentage = Math.round((result.confidence_score || 0) * 100)
  const hasOilSpill = confidencePercentage > 50

  return (
    <div className="space-y-6">
      {/* Results Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5" />
            Detection Results
          </CardTitle>
          <CardDescription>
            Analysis completed using {result.selected_model}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Activity className="w-4 h-4" />
                <span className="text-sm font-medium">Detection Status</span>
              </div>
              <Badge variant={hasOilSpill ? "destructive" : "secondary"} className="text-lg px-4 py-2">
                {hasOilSpill ? "Oil Spill Detected" : "No Oil Spill"}
              </Badge>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Zap className="w-4 h-4" />
                <span className="text-sm font-medium">Confidence</span>
              </div>
              <div className="space-y-2">
                <div className="text-2xl font-bold">{confidencePercentage}%</div>
                <Progress value={confidencePercentage} className="h-2" />
              </div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">Processing Time</span>
              </div>
              <div className="text-2xl font-bold">
                {result.processing_time?.toFixed(2)}s
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Image Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {originalImage && (
          <Card>
            <CardHeader>
              <CardTitle>Original Image</CardTitle>
            </CardHeader>
            <CardContent>              <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={originalImage}
                  alt="Original"
                  className="w-full h-full object-cover"
                />
              </div>
            </CardContent>
          </Card>
        )}

        {result.prediction_mask && (
          <Card>
            <CardHeader>
              <CardTitle>Detection Mask</CardTitle>
              <CardDescription>
                Red areas indicate detected oil spills
              </CardDescription>
            </CardHeader>
            <CardContent>              <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={result.prediction_mask}
                  alt="Prediction mask"
                  className="w-full h-full object-cover"
                />
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
