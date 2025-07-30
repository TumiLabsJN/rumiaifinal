#!/usr/bin/env node

/**
 * Debug script to test individual video analysis components with local video
 */

require('dotenv').config();
const LocalVideoAnalyzer = require('./server/services/LocalVideoAnalyzer');
const path = require('path');

// Use existing video file
const VIDEO_PATH = '/home/jorge/RumiAIv2-clean/temp/7522345762324237623_1.mp4';
const VIDEO_ID = '7522345762324237623';

async function testComponent(name, testFn, timeout = 30000) {
    console.log(`\n🧪 Testing ${name}...`);
    const startTime = Date.now();
    
    const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error(`Timeout after ${timeout/1000}s`)), timeout)
    );
    
    try {
        const result = await Promise.race([testFn(), timeoutPromise]);
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
    console.log('🚀 Starting Video Analysis Debug Tests (Local Video)');
    console.log('=================================================\n');
    console.log(`📹 Using video: ${VIDEO_PATH}`);
    console.log(`🆔 Video ID: ${VIDEO_ID}\n`);
    
    try {
        // Get analyzer instance (already initialized)
        const analyzer = LocalVideoAnalyzer;
        
        // Test each component individually with timeout
        const tests = [
            {
                name: 'Video Metadata Extraction',
                fn: () => analyzer.extractVideoMetadata(VIDEO_PATH),
                timeout: 10000
            },
            {
                name: 'Whisper Transcription',
                fn: () => analyzer.runWhisper(VIDEO_PATH, VIDEO_ID),
                timeout: 60000
            },
            {
                name: 'YOLO + DeepSort (sample frames)',
                fn: async () => {
                    // Override frame skip for faster testing
                    process.env.YOLO_FRAME_SKIP = '30';
                    return analyzer.runYOLOWithDeepSort(VIDEO_PATH, VIDEO_ID);
                },
                timeout: 60000
            },
            {
                name: 'MediaPipe Human Detection',
                fn: () => analyzer.runMediaPipe(VIDEO_PATH, VIDEO_ID),
                timeout: 60000
            },
            {
                name: 'Enhanced Human Analysis',
                fn: () => analyzer.runEnhancedHumanAnalysis(VIDEO_PATH, VIDEO_ID),
                timeout: 60000
            },
            {
                name: 'OCR Text Detection (Optimized)',
                fn: () => analyzer.runOCR(VIDEO_PATH, VIDEO_ID),
                timeout: 30000
            },
            {
                name: 'Scene Detection',
                fn: () => analyzer.runSceneDetect(VIDEO_PATH, VIDEO_ID),
                timeout: 30000
            },
            {
                name: 'CLIP Scene Labeling',
                fn: () => analyzer.runCLIP(VIDEO_PATH, VIDEO_ID),
                timeout: 60000
            },
            {
                name: 'NSFW Content Moderation',
                fn: () => analyzer.runNSFW(VIDEO_PATH, VIDEO_ID),
                timeout: 30000
            }
        ];
        
        const results = [];
        
        for (const test of tests) {
            const result = await testComponent(test.name, test.fn, test.timeout);
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
        
        // Show any failures
        const failures = results.filter(r => !r.success);
        if (failures.length > 0) {
            console.log('\n⚠️  Failed components:');
            failures.forEach(f => {
                console.log(`- ${f.name}: ${f.error}`);
            });
        }
        
    } catch (error) {
        console.error('❌ Debug test failed:', error);
        process.exit(1);
    }
}

// Run the debug tests
runDebugTests();