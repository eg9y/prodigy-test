/**
 * Interface for communicating with the inference web worker
 * Provides async methods that send messages to worker and wait for responses
 */

class WorkerInterface {
    constructor() {
        this.worker = null;
        this.messageId = 0;
        this.pendingMessages = new Map();
        this.isLoaded = false;
        this.confidenceThreshold = 0.5;
    }

    /**
     * Initialize the web worker
     */
    async init(workerPath = 'inference-worker.js') {
        return new Promise((resolve, reject) => {
            try {
                this.worker = new Worker(workerPath);
                
                // Handle messages from worker
                this.worker.onmessage = (event) => {
                    this.handleWorkerMessage(event.data);
                };
                
                // Handle worker errors
                this.worker.onerror = (error) => {
                    console.error('Worker error:', error);
                    reject(new Error(`Worker error: ${error.message}`));
                };
                
                resolve();
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Handle messages from worker
     */
    handleWorkerMessage(message) {
        const { id, success, result, error } = message;
        
        if (this.pendingMessages.has(id)) {
            const { resolve, reject } = this.pendingMessages.get(id);
            this.pendingMessages.delete(id);
            
            if (success) {
                resolve(result);
            } else {
                reject(new Error(error));
            }
        }
    }

    /**
     * Send message to worker and wait for response
     */
    async sendMessage(type, data) {
        if (!this.worker) {
            throw new Error('Worker not initialized');
        }
        
        return new Promise((resolve, reject) => {
            const id = ++this.messageId;
            
            // Store promise handlers
            this.pendingMessages.set(id, { resolve, reject });
            
            // Send message to worker
            this.worker.postMessage({
                id,
                type,
                data
            });
            
            // Set timeout for response
            setTimeout(() => {
                if (this.pendingMessages.has(id)) {
                    this.pendingMessages.delete(id);
                    reject(new Error('Worker response timeout'));
                }
            }, 10000); // 10 second timeout
        });
    }

    /**
     * Load model in worker
     */
    async loadModel(modelPath) {
        const result = await this.sendMessage('loadModel', { modelPath });
        this.isLoaded = result.success;
        return result;
    }

    /**
     * Calculate crop information for image preprocessing
     */
    calculateCropInfo(width, height, cropHeightRatio = 0.3, cropWidthRatio = 0.5) {
        const cropHeight = Math.floor(height * cropHeightRatio);
        const cropWidth = Math.floor(width * cropWidthRatio);
        
        const left = Math.floor((width - cropWidth) / 2);
        const top = height - cropHeight;
        
        return {
            left,
            top,
            cropWidth,
            cropHeight
        };
    }

    /**
     * Process image element
     */
    async processImage(imageElement) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }
        
        // Create canvas to get image data
        const canvas = document.createElement('canvas');
        canvas.width = imageElement.naturalWidth;
        canvas.height = imageElement.naturalHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const cropInfo = this.calculateCropInfo(canvas.width, canvas.height);
        
        const result = await this.sendMessage('predict', {
            imageData: {
                data: imageData.data,
                width: imageData.width,
                height: imageData.height
            },
            cropInfo,
            confidenceThreshold: this.confidenceThreshold
        });
        
        return result;
    }

    /**
     * Process video frame
     */
    async processFrame(videoElement) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }
        
        // Create canvas to capture video frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const cropInfo = this.calculateCropInfo(canvas.width, canvas.height);
        
        const result = await this.sendMessage('predict', {
            imageData: {
                data: imageData.data,
                width: imageData.width,
                height: imageData.height
            },
            cropInfo,
            confidenceThreshold: this.confidenceThreshold
        });
        
        return result;
    }

    /**
     * Set confidence threshold
     */
    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
    }

    /**
     * Terminate worker
     */
    terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
            this.isLoaded = false;
            this.pendingMessages.clear();
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WorkerInterface;
} else if (typeof window !== 'undefined') {
    window.WorkerInterface = WorkerInterface;
}