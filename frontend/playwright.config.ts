import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,  // Tests depend on shared backend state
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,  // Sequential to avoid race conditions
  reporter: [
    ['html'],
    ['list'],
  ],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  // Visual regression test settings
  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.05, // Allow 5% pixel difference
      threshold: 0.2, // Per-pixel tolerance
    },
  },
  // Snapshot path configuration for visual tests
  snapshotPathTemplate: '{testDir}/__screenshots__/{testFilePath}/{arg}{ext}',
  projects: [
    // Main test suite on Chromium
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // Cross-browser testing on Firefox
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
      // Only run core functionality tests on Firefox (skip visual regression)
      testIgnore: ['**/visual-regression.spec.ts'],
    },
    // Mobile testing
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
      testMatch: ['**/visual-regression.spec.ts'],
    },
  ],
  webServer: [
    {
      command: 'cd ../backend && python app.py',
      url: 'http://localhost:5000/api/problems',
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],
});
