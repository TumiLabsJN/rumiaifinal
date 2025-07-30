#!/usr/bin/env node

/**
 * Debug script to test Apify TikTok scraper performance
 */

require('dotenv').config();
const TikTokSingleVideoScraper = require('./server/services/TikTokSingleVideoScraper');
const { ApifyClient } = require('apify-client');

// Test video
const TEST_VIDEO_URL = 'https://www.tiktok.com/@aliasherbals/video/7522345762324237623';

async function testApifyDirect() {
    console.log('🚀 Testing Apify TikTok Scraper Performance');
    console.log('==========================================\n');
    
    const match = TEST_VIDEO_URL.match(/@([^/]+)\/video\/(\d+)/);
    const username = match[1];
    const videoId = match[2];
    
    console.log(`📱 Video: @${username}/video/${videoId}`);
    console.log(`🔑 Apify token: ${process.env.APIFY_TOKEN ? 'Configured' : 'Missing'}\n`);
    
    if (!process.env.APIFY_TOKEN) {
        console.error('❌ APIFY_TOKEN not found in .env file');
        return;
    }
    
    // Test 1: Minimal configuration (no downloads)
    console.log('📊 Test 1: Minimal scrape (metadata only, no downloads)');
    console.log('-------------------------------------------------------');
    const startTime1 = Date.now();
    
    try {
        const client = new ApifyClient({ token: process.env.APIFY_TOKEN });
        const input1 = {
            postURLs: [TEST_VIDEO_URL],
            resultsPerPage: 1,
            shouldDownloadVideos: false,  // No video download
            shouldDownloadCovers: false,  // No cover download
            shouldDownloadSubtitles: false, // No subtitle download
            proxyConfiguration: {
                useApifyProxy: true
            }
        };
        
        console.log('🚀 Starting minimal Apify run...');
        const run1 = await client.actor('clockworks/tiktok-scraper').start(input1);
        console.log(`📋 Run ID: ${run1.id}`);
        
        // Poll with detailed timing
        let pollCount = 0;
        while (true) {
            pollCount++;
            const runStatus = await client.run(run1.id).get();
            const elapsed = Math.round((Date.now() - startTime1) / 1000);
            
            console.log(`   Poll #${pollCount} (${elapsed}s): ${runStatus.status}`);
            
            if (runStatus.status === 'SUCCEEDED') {
                console.log(`✅ Completed in ${elapsed} seconds (${pollCount} polls)\n`);
                
                // Get results
                const { items } = await client.dataset(runStatus.defaultDatasetId).listItems();
                if (items && items.length > 0) {
                    const video = items[0];
                    console.log('📊 Video metadata retrieved:');
                    console.log(`   - Views: ${video.playCount || video.viewCount || 0}`);
                    console.log(`   - Likes: ${video.diggCount || video.likeCount || 0}`);
                    console.log(`   - Duration: ${video.videoMeta?.duration || 0}s`);
                    console.log(`   - Has download URL: ${!!(video.videoUrl || video.downloadAddr)}`);
                }
                break;
            } else if (runStatus.status === 'FAILED' || runStatus.status === 'ABORTED') {
                console.error(`❌ Run failed: ${runStatus.statusMessage}`);
                break;
            }
            
            await new Promise(resolve => setTimeout(resolve, 3000));
        }
        
    } catch (error) {
        console.error('❌ Test 1 failed:', error.message);
    }
    
    // Test 2: With video download
    console.log('\n📊 Test 2: Full scrape (with video download)');
    console.log('--------------------------------------------');
    const startTime2 = Date.now();
    
    try {
        const scraper = TikTokSingleVideoScraper;
        await scraper.scrapeVideo(username, videoId);
        
        const elapsed2 = Math.round((Date.now() - startTime2) / 1000);
        console.log(`✅ Full scrape completed in ${elapsed2} seconds\n`);
        
    } catch (error) {
        console.error('❌ Test 2 failed:', error.message);
    }
    
    // Test 3: Check Apify account limits
    console.log('\n📊 Test 3: Checking Apify account status');
    console.log('----------------------------------------');
    
    try {
        const client = new ApifyClient({ token: process.env.APIFY_TOKEN });
        const user = await client.user().get();
        
        console.log('👤 Account info:');
        console.log(`   - Username: ${user.username}`);
        console.log(`   - Plan: ${user.plan?.name || 'Unknown'}`);
        console.log(`   - Monthly usage: $${user.plan?.monthlyUsageUsd || 0}`);
        
        // Check if there are any rate limits
        if (user.limits) {
            console.log('\n⚠️  Account limits:');
            Object.entries(user.limits).forEach(([key, value]) => {
                console.log(`   - ${key}: ${value}`);
            });
        }
        
    } catch (error) {
        console.error('❌ Could not fetch account info:', error.message);
    }
    
    // Summary
    console.log('\n📊 Performance Analysis');
    console.log('======================');
    console.log('The TikTok scraper delay is likely due to:');
    console.log('1. Apify actor initialization time (cold start)');
    console.log('2. Proxy connection overhead');
    console.log('3. TikTok rate limiting or anti-scraping measures');
    console.log('4. Media file download time (if enabled)');
    console.log('\nConsider using cached data or implementing a queue system for better performance.');
}

// Run the test
testApifyDirect();