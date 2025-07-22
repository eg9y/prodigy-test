/**
 * Main Gameplay Analyzer for Goalkeeper Detection
 * Orchestrates video processing with ML inference for gameplay analysis
 */

class GameplayAnalyzer {
    constructor() {
        this.workerInterface = new WorkerInterface();
        this.videoProcessor = null;
        this.isInitialized = false;
        
        // Configuration
        this.config = {
            modelPath: '../models/goalkeeper_model.onnx',
            frameInterval: 1.0,        // Extract frame every N seconds
            confidenceThreshold: 0.5,  // Minimum confidence for goalkeeper detection
            maxFrames: null,           // Limit total frames (null = unlimited)
            skipFirstSeconds: 0,       // Skip first N seconds of video
            useWebWorker: true         // Use web worker for background processing
        };
        
        // Event callbacks
        this.onProgress = null;
        this.onFrameProcessed = null;
        this.onComplete = null;
        this.onError = null;
    }

    /**
     * Initialize the analyzer
     */
    async init(config = {}) {
        try {
            console.log('🚀 Initializing Gameplay Analyzer...');
            
            // Merge configuration
            this.config = { ...this.config, ...config };
            
            // Initialize worker interface
            await this.workerInterface.init();
            console.log('✅ Worker interface initialized');
            
            // Load model
            await this.workerInterface.loadModel(this.config.modelPath);
            console.log('✅ Model loaded successfully');
            
            // Set confidence threshold
            this.workerInterface.setConfidenceThreshold(this.config.confidenceThreshold);
            
            // Initialize video processor with appropriate detector
            const detector = this.config.useWebWorker ? this.workerInterface : new GoalkeeperDetector();
            if (!this.config.useWebWorker) {
                await detector.loadModel(this.config.modelPath);
                detector.setConfidenceThreshold(this.config.confidenceThreshold);
            }
            
            this.videoProcessor = new VideoProcessor(detector);
            this.setupVideoProcessorCallbacks();
            
            this.isInitialized = true;
            console.log('🎯 Gameplay Analyzer ready!');
            
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize:', error);
            if (this.onError) this.onError(error);
            throw error;
        }
    }

    /**
     * Setup video processor callbacks
     */
    setupVideoProcessorCallbacks() {
        this.videoProcessor.onProgress = (progress) => {
            if (this.onProgress) {
                this.onProgress({
                    type: 'frame_progress',
                    ...progress
                });
            }
        };

        this.videoProcessor.onFrameProcessed = (frameData) => {
            if (this.onFrameProcessed) {
                this.onFrameProcessed(frameData);
            }
        };

        this.videoProcessor.onComplete = (results) => {
            const stats = this.videoProcessor.getStats();
            if (this.onComplete) {
                this.onComplete({
                    results,
                    stats,
                    summary: this.generateSummary(results, stats)
                });
            }
        };
    }

    /**
     * Analyze gameplay video file
     */
    async analyzeVideo(videoFile, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Analyzer not initialized. Call init() first.');
        }

        const processingOptions = {
            frameInterval: options.frameInterval || this.config.frameInterval,
            maxFrames: options.maxFrames || this.config.maxFrames,
            skipFirstSeconds: options.skipFirstSeconds || this.config.skipFirstSeconds
        };

        console.log('🎮 Starting gameplay analysis...');
        console.log('Options:', processingOptions);

        try {
            const results = await this.videoProcessor.processVideo(videoFile, processingOptions);
            return this.formatResults(results);
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            if (this.onError) this.onError(error);
            throw error;
        }
    }

    /**
     * Analyze video element (for live processing)
     */
    async analyzeVideoElement(videoElement, options = {}) {
        if (!this.isInitialized) {
            throw new Error('Analyzer not initialized. Call init() first.');
        }

        const processingOptions = {
            frameInterval: options.frameInterval || this.config.frameInterval,
            maxFrames: options.maxFrames || this.config.maxFrames,
            skipFirstSeconds: options.skipFirstSeconds || this.config.skipFirstSeconds
        };

        try {
            const results = await this.videoProcessor.processVideoElement(videoElement, processingOptions);
            return this.formatResults(results);
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            if (this.onError) this.onError(error);
            throw error;
        }
    }

    /**
     * Process single frame (for real-time analysis)
     */
    async analyzeSingleFrame(videoElement) {
        if (!this.isInitialized) {
            throw new Error('Analyzer not initialized. Call init() first.');
        }

        try {
            const detector = this.config.useWebWorker ? this.workerInterface : this.videoProcessor.detector;
            const result = await detector.processFrame(videoElement);
            
            return {
                timestamp: videoElement.currentTime,
                ...result
            };
        } catch (error) {
            console.error('❌ Frame analysis failed:', error);
            if (this.onError) this.onError(error);
            throw error;
        }
    }

    /**
     * Format results for output
     */
    formatResults(results) {
        const stats = this.videoProcessor.getStats();
        
        return {
            metadata: {
                processedAt: new Date().toISOString(),
                frameInterval: this.config.frameInterval,
                confidenceThreshold: this.config.confidenceThreshold,
                ...stats
            },
            frames: results,
            summary: this.generateSummary(results, stats),
            goalkeeperMoments: results
                .filter(f => f.isGoalkeeper)
                .map(f => ({
                    timestamp: f.timestamp,
                    confidence: f.confidence,
                    timeString: this.formatTime(f.timestamp)
                }))
        };
    }

    /**
     * Generate analysis summary
     */
    generateSummary(results, stats) {
        const goalkeeperFrames = results.filter(f => f.isGoalkeeper);
        const segments = this.identifyGoalkeeperSegments(goalkeeperFrames);
        
        return {
            totalAnalysisTime: stats.duration,
            goalkeeperDetected: goalkeeperFrames.length > 0,
            goalkeeperSegments: segments.length,
            longestGoalkeeperPeriod: segments.length > 0 ? 
                Math.max(...segments.map(s => s.duration)) : 0,
            averageConfidence: parseFloat(stats.averageConfidence),
            recommendation: this.generateRecommendation(stats, segments)
        };
    }

    /**
     * Identify continuous goalkeeper segments
     */
    identifyGoalkeeperSegments(goalkeeperFrames) {
        if (goalkeeperFrames.length === 0) return [];
        
        const segments = [];
        let currentSegment = {
            start: goalkeeperFrames[0].timestamp,
            end: goalkeeperFrames[0].timestamp,
            frames: [goalkeeperFrames[0]]
        };
        
        for (let i = 1; i < goalkeeperFrames.length; i++) {
            const frame = goalkeeperFrames[i];
            const timeDiff = frame.timestamp - currentSegment.end;
            
            // If gap is less than 2x frame interval, consider it same segment
            if (timeDiff <= this.config.frameInterval * 2) {
                currentSegment.end = frame.timestamp;
                currentSegment.frames.push(frame);
            } else {
                // Close current segment and start new one
                currentSegment.duration = currentSegment.end - currentSegment.start;
                segments.push(currentSegment);
                
                currentSegment = {
                    start: frame.timestamp,
                    end: frame.timestamp,
                    frames: [frame]
                };
            }
        }
        
        // Add final segment
        currentSegment.duration = currentSegment.end - currentSegment.start;
        segments.push(currentSegment);
        
        return segments;
    }

    /**
     * Generate recommendation based on analysis
     */
    generateRecommendation(stats, segments) {
        const goalkeeperPercentage = parseFloat(stats.goalkeeperPercentage);
        
        if (goalkeeperPercentage > 50) {
            return 'High goalkeeper activity detected. Player spent significant time in goalkeeper role.';
        } else if (goalkeeperPercentage > 20) {
            return 'Moderate goalkeeper activity detected. Player occasionally acted as goalkeeper.';
        } else if (goalkeeperPercentage > 5) {
            return 'Low goalkeeper activity detected. Brief periods of goalkeeper role.';
        } else if (segments.length > 0) {
            return 'Minimal goalkeeper activity detected. Very brief goalkeeper moments.';
        } else {
            return 'No goalkeeper activity detected in this gameplay session.';
        }
    }

    /**
     * Format time in MM:SS format
     */
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    /**
     * Export analysis results
     */
    exportResults(format = 'json') {
        if (!this.videoProcessor) {
            throw new Error('No analysis results to export');
        }
        
        return this.videoProcessor.exportResults(format);
    }

    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        if (this.workerInterface && newConfig.confidenceThreshold !== undefined) {
            this.workerInterface.setConfidenceThreshold(newConfig.confidenceThreshold);
        }
    }

    /**
     * Stop current analysis
     */
    stop() {
        if (this.videoProcessor) {
            this.videoProcessor.stop();
        }
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.videoProcessor) {
            this.videoProcessor.stop();
        }
        if (this.workerInterface) {
            this.workerInterface.terminate();
        }
        this.isInitialized = false;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GameplayAnalyzer;
} else if (typeof window !== 'undefined') {
    window.GameplayAnalyzer = GameplayAnalyzer;
}