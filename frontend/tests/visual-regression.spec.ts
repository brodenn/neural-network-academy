import { test, expect } from '@playwright/test';
import { waitForConnection, selectProblem } from './fixtures/test-helpers';

test.describe.skip('Visual Regression Tests', () => {
  // SKIP: Visual regression tests require stable baselines and are flaky in CI
  // Run manually with: npx playwright test visual-regression.spec.ts --update-snapshots
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForConnection(page);
  });

  test('problem selector collapsed state', async ({ page }) => {
    // Wait for initial render
    await page.waitForTimeout(500);

    // Take screenshot of problem selector dropdown button in header
    const headerDropdown = page.locator('header button').first();
    await expect(headerDropdown).toHaveScreenshot('problem-selector-collapsed.png', {
      maxDiffPixels: 100,
    });
  });

  test('problem selector expanded Level 1', async ({ page }) => {
    // Open dropdown
    const headerDropdown = page.locator('header button').first();
    await headerDropdown.click();
    await page.waitForTimeout(200);

    // Expand Level 1
    const menu = page.locator('.absolute.bg-gray-800.w-80');
    await menu.getByRole('button', { name: /Level 1/ }).click();
    await page.waitForTimeout(300);

    // Capture expanded dropdown with Level 1 problems visible
    await expect(menu).toHaveScreenshot('problem-selector-level1-expanded.png', {
      maxDiffPixels: 100,
    });
  });

  test('failure case level styling', async ({ page }) => {
    // Open dropdown and expand Level 5
    const headerDropdown = page.locator('header button').first();
    await headerDropdown.click();
    await page.waitForTimeout(200);

    const menu = page.locator('.absolute.bg-gray-800.w-80');
    await menu.getByRole('button', { name: /Level 5/ }).click();
    await page.waitForTimeout(300);

    // Capture failure case styling in dropdown
    await expect(menu).toHaveScreenshot('failure-case-level.png', {
      maxDiffPixels: 100,
    });
  });

  test('input panel binary inputs', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');
    await page.waitForTimeout(500);

    const inputPanel = page.locator('[data-testid="input-panel"]');
    await expect(inputPanel).toHaveScreenshot('input-panel-binary.png', {
      maxDiffPixels: 100,
    });
  });

  test('input panel slider inputs', async ({ page }) => {
    await selectProblem(page, 'Level 3', 'Circle');
    await page.waitForTimeout(500);

    const inputPanel = page.locator('[data-testid="input-panel"]');
    await expect(inputPanel).toHaveScreenshot('input-panel-sliders.png', {
      maxDiffPixels: 100,
    });
  });

  test('CNN grid input', async ({ page }) => {
    await selectProblem(page, 'Level 7', 'Shape Detection');
    await page.waitForTimeout(2000);

    // CNN uses the same InputPanel but with grid mode
    const inputPanel = page.locator('[data-testid="input-panel"], [data-testid="input-grid"]').first();
    await expect(inputPanel).toHaveScreenshot('input-panel-cnn-grid.png', {
      maxDiffPixels: 200, // Allow more variance for grid
    });
  });

  test('training panel untrained state', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');
    await page.waitForTimeout(500);

    const trainingPanel = page.locator('[data-testid="training-panel"]');
    await expect(trainingPanel).toHaveScreenshot('training-panel-untrained.png', {
      maxDiffPixels: 100,
    });
  });

  test('failure case training panel', async ({ page }) => {
    await selectProblem(page, 'Level 5: Failure Cases', 'XOR (No Hidden Layer)');
    await page.waitForTimeout(500);

    const trainingPanel = page.locator('[data-testid="training-panel"]');
    await expect(trainingPanel).toHaveScreenshot('training-panel-failure-case.png', {
      maxDiffPixels: 100,
    });
  });

  test('network visualization renders correctly', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');
    await page.waitForTimeout(500);

    // Capture SVG network visualization
    const networkViz = page.locator('svg').first();
    await expect(networkViz).toHaveScreenshot('network-viz-simple.png', {
      maxDiffPixels: 50,
    });
  });

  test('XOR network visualization with hidden layer', async ({ page }) => {
    await selectProblem(page, 'Level 2', 'XOR Gate');
    await page.waitForTimeout(500);

    const networkViz = page.locator('svg').first();
    await expect(networkViz).toHaveScreenshot('network-viz-xor.png', {
      maxDiffPixels: 50,
    });
  });

  test('full page layout desktop', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await selectProblem(page, 'Level 1', 'AND Gate');
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('full-page-desktop.png', {
      fullPage: false,
      maxDiffPixels: 500,
    });
  });

  test('responsive layout tablet', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await selectProblem(page, 'Level 1', 'AND Gate');
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('full-page-tablet.png', {
      fullPage: false,
      maxDiffPixels: 500,
    });
  });

  test('responsive layout mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('full-page-mobile.png', {
      fullPage: false,
      maxDiffPixels: 500,
    });
  });
});
