#!/usr/bin/env node

/**
 * Debug script to test individual video analysis components
 */

require('dotenv').config();
const LocalVideoAnalyzer = require('./server/services/LocalVideoAnalyzer');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

// Test video URL
const TEST_VIDEO_URL = 'https://www.tiktok.com/@aliasherbals/video/7522345762324237623';

async function downloadTestVideo() {
    console.log('📥 Downloading test video...');
    
    // Download a small test video
    const videoPath = path.join(__dirname, 'temp', 'test_debug.mp4');
    
    // Use the TikTok scraper to get download URL
    const TikTokSingleVideoScraper = require('./server/services/TikTokSingleVideoScraper');
    const match = TEST_VIDEO_URL.match(/@([^/]+)\/video\/(\d+)/);
    const username = match[1];
    const videoId = match[2];
    
    try {
        const videoData = await TikTokSingleVideoScraper.scrapeVideo(username, videoId);
        const downloadUrl = videoData.mediaUrls?.[0] || videoData.videoUrl || videoData.downloadAddr;
        
        if (!downloadUrl) {
            throw new Error('No download URL available');
        }
        
        // Download the video
        const writer = require('fs').createWriteStream(videoPath);
        const response = await axios({
            method: 'GET',
            url: downloadUrl,
            responseType: 'stream',
            headers: { 'User-Agent': 'RumiAI/1.0' }
        });
        
        response.data.pipe(writer);
        
        await new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });
        
        console.log('✅ Video downloaded:', videoPath);
        return videoPath;
        
    } catch (error) {
        console.error('❌ Failed to download video:', error.message);
        throw error;
    }
}

async function testComponent(name, testFn) {
    console.log(`\n🧪 Testing ${name}...`);
    const startTime = Date.now();
    
    try {
        const result = await testFn();
        const duration = (Date.now() - startTime) / 1000;
        console.log(`✅ ${name} completed in ${duration.toFixed(2)}s`);
        return { name, success: true, duration, result };
    } catch (error) {
        const duration = (Date.now() - startTime) / 1000;
        console.error(`❌ ${name} failed after ${duration.toFixed(2)}s:`, error.message);
        return { name, success: false, duration, error: error.message };
    }
}

async function runDebugTests() {
    console.log('🚀 Starting Video Analysis Debug Tests');
    console.log('=====================================\n');
    
    try {
        // Download test video
        const videoPath = await downloadTestVideo();
        const videoId = 'debug_test';
        
        // Initialize analyzer
        const analyzer = new LocalVideoAnalyzer();
        
        // Test each component individually with timeout
        const tests = [
            {
                name: 'Video Metadata Extraction',
                fn: () => analyzer.extractVideoMetadata(videoPath)
            },
            {
                name: 'Whisper Transcription',
                fn: () => analyzer.runWhisper(videoPath, videoId)
            },
            {
                name: 'YOLO + DeepSort (10 frames)',
                fn: async () => {
                    // Override frame skip for faster testing
                    process.env.YOLO_FRAME_SKIP = '30';
                    return analyzer.runYOLOWithDeepSort(videoPath, videoId);
                }
            },
            {
                name: 'MediaPipe Human Detection',
                fn: () => analyzer.runMediaPipe(videoPath, videoId)
            },
            {
                name: 'Enhanced Human Analysis',
                fn: () => analyzer.runEnhancedHumanAnalysis(videoPath, videoId)
            },
            {
                name: 'OCR Text Detection',
                fn: () => analyzer.runOCR(videoPath, videoId)
            },
            {
                name: 'Scene Detection',
                fn: () => analyzer.runSceneDetect(videoPath, videoId)
            },
            {
                name: 'CLIP Scene Labeling',
                fn: () => analyzer.runCLIP(videoPath, videoId)
            },
            {
                name: 'NSFW Content Moderation',
                fn: () => analyzer.runNSFW(videoPath, videoId)
            }
        ];
        
        const results = [];
        
        for (const test of tests) {
            const result = await testComponent(test.name, test.fn);
            results.push(result);
            
            // Add a small delay between tests
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        // Summary
        console.log('\n📊 Test Summary');
        console.log('===============\n');
        
        const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);
        const successCount = results.filter(r => r.success).length;
        
        results.forEach(r => {
            const status = r.success ? '✅' : '❌';
            console.log(`${status} ${r.name}: ${r.duration.toFixed(2)}s`);
        });
        
        console.log(`\n🏁 Total time: ${totalDuration.toFixed(2)}s`);
        console.log(`📈 Success rate: ${successCount}/${results.length}`);
        
        // Identify slowest components
        const sorted = [...results].sort((a, b) => b.duration - a.duration);
        console.log('\n🐌 Slowest components:');
        sorted.slice(0, 3).forEach((r, i) => {
            console.log(`${i + 1}. ${r.name}: ${r.duration.toFixed(2)}s`);
        });
        
        // Clean up
        try {
            await fs.unlink(videoPath);
        } catch (e) {
            // Ignore
        }
        
    } catch (error) {
        console.error('❌ Debug test failed:', error);
        process.exit(1);
    }
}

// Run the debug tests
runDebugTests();