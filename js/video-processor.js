/**
 * Video Processing Module for Goalkeeper Detection
 * Extracts frames from gameplay videos at intervals and processes them
 */

class VideoProcessor {
	constructor(goalkeeperDetector) {
		this.detector = goalkeeperDetector;
		this.isProcessing = false;
		this.processedFrames = [];
		this.onFrameProcessed = null; // Callback for each processed frame
		this.onProgress = null; // Callback for progress updates
		this.onComplete = null; // Callback when processing is complete

		// Processing settings
		this.frameInterval = 1.0; // Extract frame every N seconds
		this.maxFrames = null; // Limit total frames (null = no limit)
		this.skipFirstSeconds = 0; // Skip first N seconds of video
	}

	/**
	 * Process a video file
	 */
	async processVideo(videoFile, options = {}) {
		if (this.isProcessing) {
			throw new Error("Already processing a video");
		}

		// Apply options
		this.frameInterval = options.frameInterval || 1.0;
		this.maxFrames = options.maxFrames || null;
		this.skipFirstSeconds = options.skipFirstSeconds || 0;

		this.isProcessing = true;
		this.processedFrames = [];

		try {
			console.log("🎥 Starting video processing...");
			console.log(`Frame interval: ${this.frameInterval}s`);
			console.log(`Max frames: ${this.maxFrames || "unlimited"}`);

			const results = await this._extractAndProcessFrames(videoFile);

			console.log("✅ Video processing complete!");
			console.log(`Processed ${results.length} frames`);

			if (this.onComplete) {
				this.onComplete(results);
			}

			return results;
		} finally {
			this.isProcessing = false;
		}
	}

	/**
	 * Process frames from a video element that's already loaded
	 */
	async processVideoElement(videoElement, options = {}) {
		if (this.isProcessing) {
			throw new Error("Already processing a video");
		}

		// Apply options
		this.frameInterval = options.frameInterval || 1.0;
		this.maxFrames = options.maxFrames || null;
		this.skipFirstSeconds = options.skipFirstSeconds || 0;

		this.isProcessing = true;
		this.processedFrames = [];

		try {
			console.log("🎥 Starting video element processing...");
			const results = await this._processVideoElementFrames(videoElement);

			console.log("✅ Video processing complete!");
			console.log(`Processed ${results.length} frames`);

			if (this.onComplete) {
				this.onComplete(results);
			}

			return results;
		} finally {
			this.isProcessing = false;
		}
	}

	/**
	 * Extract and process frames from video file
	 */
	async _extractAndProcessFrames(videoFile) {
		return new Promise((resolve, reject) => {
			const video = document.createElement("video");
			video.preload = "metadata";

			video.onloadedmetadata = async () => {
				try {
					const results = await this._processVideoElementFrames(video);
					resolve(results);
				} catch (error) {
					reject(error);
				}
			};

			video.onerror = () => {
				reject(new Error("Failed to load video file"));
			};

			// Load video file
			video.src = URL.createObjectURL(videoFile);
		});
	}

	/**
	 * Process frames from a video element
	 */
	async _processVideoElementFrames(videoElement) {
		const duration = videoElement.duration;
		const startTime = this.skipFirstSeconds;
		const endTime = duration;

		const timePoints = this._generateTimePoints(startTime, endTime);
		const results = [];

		console.log(
			`Extracting ${timePoints.length} frames from video (${duration.toFixed(1)}s)`,
		);

		for (let i = 0; i < timePoints.length; i++) {
			const time = timePoints[i];

			try {
				// Seek to time point
				await this._seekToTime(videoElement, time);

				// Process frame
				const result = await this.detector.processFrame(videoElement);

				console.log("RESULTO:", result);

				const frameData = {
					timestamp: time,
					frameIndex: i,
					...result,
				};

				results.push(frameData);
				this.processedFrames.push(frameData);

				// Call progress callback
				if (this.onProgress) {
					this.onProgress({
						processedFrames: i + 1,
						totalFrames: timePoints.length,
						percentage: (i + 1) / timePoints.length,
						current: i + 1,     // Keep for backward compatibility
						total: timePoints.length,  // Keep for backward compatibility
						timestamp: time,
						result: frameData,
					});
				}

				// Call frame processed callback
				if (this.onFrameProcessed) {
					this.onFrameProcessed(frameData);
				}

				console.log(
					`Frame ${i + 1}/${timePoints.length} at ${time.toFixed(1)}s: ${result.predictedClass} (${(result.confidence * 100).toFixed(1)}%)`,
				);
			} catch (error) {
				console.error(`Error processing frame at ${time}s:`, error);
			}
		}

		return results;
	}

	/**
	 * Generate time points for frame extraction
	 */
	_generateTimePoints(startTime, endTime) {
		const timePoints = [];
		let currentTime = startTime;

		while (currentTime <= endTime) {
			timePoints.push(currentTime);
			currentTime += this.frameInterval;

			// Respect max frames limit
			if (this.maxFrames && timePoints.length >= this.maxFrames) {
				break;
			}
		}

		return timePoints;
	}

	/**
	 * Seek video to specific time
	 */
	async _seekToTime(videoElement, time) {
		return new Promise((resolve, reject) => {
			const timeout = setTimeout(() => {
				reject(new Error(`Timeout seeking to ${time}s`));
			}, 5000);

			const onSeeked = () => {
				videoElement.removeEventListener("seeked", onSeeked);
				clearTimeout(timeout);
				resolve();
			};

			videoElement.addEventListener("seeked", onSeeked);
			videoElement.currentTime = time;
		});
	}

	/**
	 * Get processing statistics
	 */
	getStats() {
		if (this.processedFrames.length === 0) {
			return null;
		}

		const goalkeeperFrames = this.processedFrames.filter((f) => f.isGoalkeeper);
		const totalFrames = this.processedFrames.length;

		return {
			totalFrames,
			goalkeeperFrames: goalkeeperFrames.length,
			nonGoalkeeperFrames: totalFrames - goalkeeperFrames.length,
			goalkeeperPercentage: (
				(goalkeeperFrames.length / totalFrames) *
				100
			).toFixed(1),
			averageConfidence: (
				this.processedFrames.reduce((sum, f) => sum + f.confidence, 0) /
				totalFrames
			).toFixed(3),
			duration:
				this.processedFrames.length > 0
					? this.processedFrames[this.processedFrames.length - 1].timestamp -
						this.processedFrames[0].timestamp
					: 0,
		};
	}

	/**
	 * Export results to various formats
	 */
	exportResults(format = "json") {
		const stats = this.getStats();

		switch (format.toLowerCase()) {
			case "json":
				return {
					metadata: {
						processedAt: new Date().toISOString(),
						frameInterval: this.frameInterval,
						...stats,
					},
					frames: this.processedFrames,
				};

			case "csv": {
				const csvHeader =
					"timestamp,frameIndex,isGoalkeeper,confidence,predictedClass\n";
				const csvRows = this.processedFrames
					.map(
						(frame) =>
							`${frame.timestamp},${frame.frameIndex},${frame.isGoalkeeper},${frame.confidence},${frame.predictedClass}`,
					)
					.join("\n");
				return csvHeader + csvRows;
			}

			case "summary":
				return {
					summary: stats,
					goalkeeperMoments: this.processedFrames
						.filter((f) => f.isGoalkeeper)
						.map((f) => ({
							timestamp: f.timestamp,
							confidence: f.confidence,
						})),
				};

			default:
				throw new Error(`Unsupported export format: ${format}`);
		}
	}

	/**
	 * Stop processing (if running)
	 */
	stop() {
		this.isProcessing = false;
	}
}

// Export for use in other modules
if (typeof module !== "undefined" && module.exports) {
	module.exports = VideoProcessor;
} else if (typeof window !== "undefined") {
	window.VideoProcessor = VideoProcessor;
}
