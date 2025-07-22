/**
 * Web Worker for Goalkeeper Detection Inference
 * Runs ML inference in background thread to keep UI responsive
 */

// Import ONNX.js for web workers
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js');

// Configure ONNX Runtime to use CDN for WASM files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';

class InferenceWorker {
    constructor() {
        this.model = null;
        this.isLoaded = false;
        this.inputSize = 224;
        this.classes = ['goalkeeper', 'not_goalkeeper'];
    }

    /**
     * Load the ONNX model
     */
    async loadModel(modelPath) {
        try {
            console.log('Worker: Loading ONNX model from:', modelPath);
            
            this.model = await ort.InferenceSession.create(modelPath);
            this.isLoaded = true;
            
            console.log('Worker: Model loaded successfully');
            return { success: true };
        } catch (error) {
            console.error('Worker: Failed to load model:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Preprocess image data to tensor format
     * Expects imageData as {data: Uint8ClampedArray, width: number, height: number}
     */
    preprocessImageData(imageData, cropInfo) {
        const { width, height, data } = imageData;
        const { left, top, cropWidth, cropHeight } = cropInfo;
        
        // Create tensor data [1, 3, 224, 224] (NCHW format)
        const tensorData = new Float32Array(1 * 3 * this.inputSize * this.inputSize);
        const pixelCount = this.inputSize * this.inputSize;
        
        // Process each pixel in the cropped region
        for (let y = 0; y < this.inputSize; y++) {
            for (let x = 0; x < this.inputSize; x++) {
                // Calculate source coordinates (scaling from crop region to input size)
                const srcX = Math.floor((x / this.inputSize) * cropWidth) + left;
                const srcY = Math.floor((y / this.inputSize) * cropHeight) + top;
                
                // Ensure coordinates are within bounds
                const boundedSrcX = Math.max(0, Math.min(width - 1, srcX));
                const boundedSrcY = Math.max(0, Math.min(height - 1, srcY));
                
                // Get pixel from source image data
                const srcIndex = (boundedSrcY * width + boundedSrcX) * 4; // RGBA
                const tensorIndex = y * this.inputSize + x;
                
                // Normalize to 0-1 and arrange in CHW format
                tensorData[tensorIndex] = data[srcIndex] / 255.0;     // R channel
                tensorData[pixelCount + tensorIndex] = data[srcIndex + 1] / 255.0; // G channel
                tensorData[pixelCount * 2 + tensorIndex] = data[srcIndex + 2] / 255.0; // B channel
            }
        }
        
        return tensorData;
    }

    /**
     * Run inference on tensor data
     */
    async predict(tensorData) {
        if (!this.isLoaded || !this.model) {
            throw new Error('Model not loaded');
        }

        try {
            // Create input tensor
            const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, this.inputSize, this.inputSize]);
            
            // Run inference
            const feeds = {};
            feeds[this.model.inputNames[0]] = inputTensor;
            const results = await this.model.run(feeds);
            
            // Get output probabilities
            const outputData = results[this.model.outputNames[0]].data;
            
            // Apply softmax
            const probabilities = this.softmax(Array.from(outputData));
            
            // Get prediction
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            const predictedClass = this.classes[maxIndex];
            const confidence = probabilities[maxIndex];
            
            return {
                predictedClass,
                confidence,
                probabilities: {
                    goalkeeper: probabilities[0],
                    not_goalkeeper: probabilities[1]
                }
            };
        } catch (error) {
            console.error('Worker: Prediction failed:', error);
            throw error;
        }
    }

    /**
     * Apply softmax to convert logits to probabilities
     */
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expValues = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expValues.reduce((sum, val) => sum + val, 0);
        return expValues.map(val => val / sumExp);
    }
}

// Initialize worker instance
const worker = new InferenceWorker();

// Handle messages from main thread
self.onmessage = async function(event) {
    const { id, type, data } = event.data;
    
    try {
        let result;
        
        switch (type) {
            case 'loadModel':
                result = await worker.loadModel(data.modelPath);
                break;
                
            case 'predict':
                // data contains: { imageData, cropInfo, confidenceThreshold }
                const tensorData = worker.preprocessImageData(data.imageData, data.cropInfo);
                const prediction = await worker.predict(tensorData);
                
                // Apply confidence threshold
                const isGoalkeeper = prediction.predictedClass === 'goalkeeper' && 
                                   prediction.confidence >= data.confidenceThreshold;
                
                result = {
                    ...prediction,
                    isGoalkeeper
                };
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
        
        // Send successful response
        self.postMessage({
            id,
            success: true,
            result
        });
        
    } catch (error) {
        // Send error response
        self.postMessage({
            id,
            success: false,
            error: error.message
        });
    }
};