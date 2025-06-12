"use client"

import * as React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Clock, Zap, Target, Activity, Users } from "lucide-react"
import { PredictionResult, EnsemblePredictionResult } from "@/types/api"

interface PredictionResultsProps {
  result: EnsemblePredictionResult
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

  const confidencePercentage = Math.round((result.ensemble_confidence || 0) * 100)
  const hasOilSpill = confidencePercentage > 50

  return (
    <div className="space-y-6">
      {/* Ensemble Results Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5" />
            Ensemble Detection Results
          </CardTitle>
          <CardDescription>
            Combined analysis from {result.individual_predictions?.length || 2} AI models
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Activity className="w-4 h-4" />
                <span className="text-sm font-medium">Final Detection</span>
              </div>
              <Badge variant={hasOilSpill ? "destructive" : "secondary"} className="text-lg px-4 py-2">
                {result.ensemble_prediction || "Analysis Complete"}
              </Badge>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Zap className="w-4 h-4" />
                <span className="text-sm font-medium">Ensemble Confidence</span>
              </div>
              <div className="space-y-2">
                <div className="text-2xl font-bold">{confidencePercentage}%</div>
                <Progress value={confidencePercentage} className="h-2" />
              </div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm font-medium">Total Processing Time</span>
              </div>
              <div className="text-2xl font-bold">
                {result.total_processing_time?.toFixed(2)}s
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Model Results */}
      {result.individual_predictions && result.individual_predictions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5" />
              Individual Model Results
            </CardTitle>
            <CardDescription>
              Detailed results from each AI model
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {result.individual_predictions.map((modelResult, index) => {
                const modelConfidence = Math.round((modelResult.confidence || 0) * 100)
                const modelHasOilSpill = modelConfidence > 50
                
                return (
                  <div key={index} className="p-4 border rounded-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium">{modelResult.model_name}</h4>
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
                    
                    <div className="text-sm text-muted-foreground">
                      Processing time: {modelResult.processing_time?.toFixed(2)}s
                    </div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Image Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {originalImage && (
          <Card>
            <CardHeader>
              <CardTitle>Original Image</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={originalImage}
                  alt="Original"
                  className="w-full h-full object-cover"
                />
              </div>
            </CardContent>
          </Card>
        )}        {/* Individual Model Prediction Masks */}
        {result.individual_predictions?.map((modelResult, index) => (
          modelResult.prediction_mask && (
            <Card key={`mask-${index}`}>
              <CardHeader>
                <CardTitle>{modelResult.model_name} Detection</CardTitle>
                <CardDescription>
                  Highlighted areas show detected oil spills
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${modelResult.prediction_mask}`}
                    alt={`${modelResult.model_name} prediction mask`}
                    className="w-full h-full object-cover"
                  />
                </div>
              </CardContent>
            </Card>
          )
        ))}

        {/* Ensemble Prediction Mask */}
        {result.ensemble_mask && (
          <Card>
            <CardHeader>
              <CardTitle>Ensemble Detection</CardTitle>
              <CardDescription>
                Combined detection from all models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-square relative bg-muted rounded-lg overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${result.ensemble_mask}`}
                  alt="Ensemble prediction mask"
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
