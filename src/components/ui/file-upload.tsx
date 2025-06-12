"use client"

import * as React from "react"
import { useDropzone } from "react-dropzone"
import { Upload, X, Image as ImageIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface FileUploadProps {
  onFilesChange: (files: File[]) => void
  multiple?: boolean
  maxFiles?: number
  maxSize?: number // in MB
  className?: string
  disabled?: boolean
  onError?: (error: string) => void
}

export function FileUpload({ 
  onFilesChange, 
  multiple = false, 
  maxFiles = 1,
  maxSize = 10, // 10MB default
  className,
  disabled = false,
  onError
}: FileUploadProps) {
  const [files, setFiles] = React.useState<File[]>([])
  const [uploadError, setUploadError] = React.useState<string>("")

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxSize * 1024 * 1024) {
      return `File "${file.name}" is too large. Maximum size is ${maxSize}MB.`
    }

    // Check file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff']
    if (!allowedTypes.includes(file.type)) {
      return `File "${file.name}" is not a supported image format.`
    }

    return null
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    },
    multiple,
    maxFiles,
    disabled,
    onDrop: (acceptedFiles, rejectedFiles) => {
      setUploadError("")
      
      // Handle rejected files
      if (rejectedFiles.length > 0) {
        const error = rejectedFiles[0].errors[0]?.message || "File was rejected"
        setUploadError(error)
        onError?.(error)
        return
      }

      // Validate accepted files
      const validFiles: File[] = []
      for (const file of acceptedFiles) {
        const error = validateFile(file)
        if (error) {
          setUploadError(error)
          onError?.(error)
          return
        }
        validFiles.push(file)
      }

      const newFiles = multiple ? [...files, ...validFiles] : validFiles
      setFiles(newFiles)
      onFilesChange(newFiles)
    }
  })

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    setFiles(newFiles)
    onFilesChange(newFiles)
  }

  return (
    <div className={cn("w-full", className)}>
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
          isDragActive 
            ? "border-primary bg-primary/5" 
            : "border-muted-foreground/25 hover:border-primary/50",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-4">
          <Upload className="w-12 h-12 text-muted-foreground" />
          <div>
            <p className="text-lg font-medium">
              {isDragActive 
                ? "Drop the images here..." 
                : "Drag & drop images here, or click to select"
              }
            </p>            <p className="text-sm text-muted-foreground mt-2">
              Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF • Max size: {maxSize}MB
              {multiple && ` • Max ${maxFiles} files`}
            </p>
          </div>
        </div>
      </div>

      {uploadError && (
        <div className="mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive">{uploadError}</p>
        </div>
      )}

      {files.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-medium mb-3">Selected Files:</h4>
          <div className="space-y-2">
            {files.map((file, index) => (
              <div 
                key={index} 
                className="flex items-center justify-between p-3 bg-muted rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <ImageIcon className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="p-1 hover:bg-destructive/10 rounded-full transition-colors"
                  disabled={disabled}
                >
                  <X className="w-4 h-4 text-destructive" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
