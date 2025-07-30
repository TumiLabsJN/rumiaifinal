#!/usr/bin/env node
/**
 * Compatibility wrapper for Node.js integration.
 * 
 * Ensures smooth transition from v1 to v2.
 * AUTOMATED - NO HUMAN INTERVENTION REQUIRED.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

class RumiAICompatibilityWrapper {
    constructor() {
        this.v2Runner = path.join(__dirname, 'rumiai_runner.py');
        this.pythonCmd = process.env.PYTHON_CMD || 'python3';
    }

    /**
     * Check if v2 is available
     */
    isV2Available() {
        return fs.existsSync(this.v2Runner);
    }

    /**
     * Run RumiAI with automatic v1/v2 detection
     */
    async run(videoInput, options = {}) {
        console.log('🔍 RumiAI Compatibility Wrapper');
        
        if (!this.isV2Available()) {
            throw new Error('RumiAI v2 not found. Please run installation first.');
        }

        // Determine if input is URL or ID
        const isUrl = videoInput.startsWith('http');
        console.log(`📊 Input type: ${isUrl ? 'URL' : 'Video ID'}`);

        // Build command
        const args = [this.v2Runner, videoInput];
        
        if (options.outputFormat) {
            args.push('--output-format', options.outputFormat);
        }

        // Run v2
        return this.runPython(args, options);
    }

    /**
     * Run Python script with monitoring
     */
    runPython(args, options = {}) {
        return new Promise((resolve, reject) => {
            console.log(`🚀 Running: ${this.pythonCmd} ${args.join(' ')}`);
            
            const child = spawn(this.pythonCmd, args, {
                cwd: options.cwd || process.cwd(),
                env: { ...process.env, ...options.env }
            });

            let stdout = '';
            let stderr = '';
            let lastProgress = null;

            // Monitor stdout
            child.stdout.on('data', (data) => {
                const output = data.toString();
                stdout += output;

                // Parse progress updates
                const progressMatch = output.match(/📊\s+(\w+)\.{3}\s+\((\d+)%\)/);
                if (progressMatch) {
                    const [, stage, percent] = progressMatch;
                    lastProgress = { stage, percent: parseInt(percent) };
                    
                    if (options.onProgress) {
                        options.onProgress(lastProgress);
                    }
                }

                // Forward output
                if (options.verbose || output.includes('❌') || output.includes('🔴')) {
                    process.stdout.write(output);
                }
            });

            // Monitor stderr
            child.stderr.on('data', (data) => {
                stderr += data.toString();
                if (options.verbose) {
                    process.stderr.write(data);
                }
            });

            // Handle completion
            child.on('close', (code) => {
                if (code === 0) {
                    // Try to parse JSON result
                    try {
                        const lines = stdout.split('\n');
                        const jsonStart = lines.findIndex(line => line.trim().startsWith('{'));
                        
                        if (jsonStart >= 0) {
                            const jsonLines = lines.slice(jsonStart);
                            const jsonStr = jsonLines.join('\n');
                            const result = JSON.parse(jsonStr);
                            resolve(result);
                        } else {
                            // No JSON found, return raw output
                            resolve({
                                success: true,
                                output: stdout,
                                progress: lastProgress
                            });
                        }
                    } catch (error) {
                        // Failed to parse, return raw
                        resolve({
                            success: true,
                            output: stdout,
                            error: 'Failed to parse JSON output'
                        });
                    }
                } else {
                    // Error occurred
                    const errorInfo = {
                        code,
                        error: stderr || 'Unknown error',
                        output: stdout
                    };

                    // Map exit codes
                    switch (code) {
                        case 1:
                            errorInfo.type = 'general';
                            break;
                        case 2:
                            errorInfo.type = 'invalid_arguments';
                            break;
                        case 3:
                            errorInfo.type = 'api_failure';
                            break;
                        case 4:
                            errorInfo.type = 'ml_failure';
                            break;
                        default:
                            errorInfo.type = 'unknown';
                    }

                    reject(errorInfo);
                }
            });

            // Handle errors
            child.on('error', (error) => {
                reject({
                    code: -1,
                    error: error.message,
                    type: 'spawn_error'
                });
            });
        });
    }

    /**
     * Run migration from v1 to v2
     */
    async migrate(options = {}) {
        console.log('🔄 Running v1 to v2 migration...');
        
        const migrationScript = path.join(__dirname, 'migrate_to_v2.py');
        const args = [migrationScript];
        
        if (options.dryRun) {
            args.push('--dry-run');
        }
        
        if (options.noBackup) {
            args.push('--no-backup');
        }
        
        return this.runPython(args, { verbose: true });
    }

    /**
     * Check system compatibility
     */
    async checkCompatibility() {
        console.log('🔍 Checking system compatibility...');
        
        const checks = {
            python: false,
            v2_installed: false,
            dependencies: false
        };

        // Check Python
        try {
            const { stdout } = await this.runCommand(this.pythonCmd, ['--version']);
            checks.python = stdout.includes('Python 3');
            console.log(`✅ Python: ${stdout.trim()}`);
        } catch (error) {
            console.log('❌ Python not found');
        }

        // Check v2 installation
        checks.v2_installed = this.isV2Available();
        console.log(checks.v2_installed ? '✅ RumiAI v2 installed' : '❌ RumiAI v2 not found');

        // Check dependencies
        try {
            await this.runPython(['-c', 'import rumiai_v2']);
            checks.dependencies = true;
            console.log('✅ Dependencies installed');
        } catch (error) {
            console.log('❌ Dependencies missing');
        }

        return checks;
    }

    /**
     * Helper to run commands
     */
    runCommand(command, args) {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args);
            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (data) => stdout += data);
            child.stderr.on('data', (data) => stderr += data);

            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(stderr || stdout));
                }
            });
        });
    }
}

// CLI interface
if (require.main === module) {
    const wrapper = new RumiAICompatibilityWrapper();
    const args = process.argv.slice(2);

    if (args.length === 0 || args[0] === '--help') {
        console.log(`
RumiAI Compatibility Wrapper

Usage:
  compatibility_wrapper.js <video_url_or_id>    Process video
  compatibility_wrapper.js --check               Check compatibility
  compatibility_wrapper.js --migrate             Migrate v1 to v2
  compatibility_wrapper.js --help                Show this help

Options:
  --verbose          Show detailed output
  --dry-run          Simulate migration without changes
  --no-backup        Skip backups during migration
`);
        process.exit(0);
    }

    // Handle commands
    (async () => {
        try {
            if (args[0] === '--check') {
                const result = await wrapper.checkCompatibility();
                process.exit(Object.values(result).every(v => v) ? 0 : 1);
            } else if (args[0] === '--migrate') {
                const options = {
                    dryRun: args.includes('--dry-run'),
                    noBackup: args.includes('--no-backup')
                };
                await wrapper.migrate(options);
            } else {
                // Process video
                const result = await wrapper.run(args[0], {
                    verbose: args.includes('--verbose'),
                    outputFormat: 'json',
                    onProgress: (progress) => {
                        console.log(`Progress: ${progress.stage} ${progress.percent}%`);
                    }
                });
                
                console.log(JSON.stringify(result, null, 2));
            }
        } catch (error) {
            console.error('❌ Error:', error);
            process.exit(error.code || 1);
        }
    })();
}

module.exports = RumiAICompatibilityWrapper;