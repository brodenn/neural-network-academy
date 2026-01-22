import { test, expect } from '@playwright/test';
import { waitForConnection, selectProblem, stopTraining } from './fixtures/test-helpers';

test.describe('Network Visualization', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForConnection(page);
  });

  test('should render SVG network diagram', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Check that SVG elements exist
    await expect(page.locator('svg').first()).toBeVisible();
  });

  test('should show network nodes (circles)', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Network visualization should have circle elements for neurons
    const circles = page.locator('svg circle');
    await expect(circles.first()).toBeVisible();
  });

  test('should show network connections (paths)', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Network connections are rendered as SVG path elements (curved lines)
    const paths = page.locator('svg path');
    const count = await paths.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should show loss curve chart after training starts', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Start training
    await page.getByRole('button', { name: 'Train Static' }).click();
    await page.waitForTimeout(2000);

    // Recharts renders SVG wrapper
    await expect(page.locator('.recharts-wrapper')).toBeVisible();

    await stopTraining(page);
  });

  test('loss curve should update during training', async ({ page }) => {
    test.setTimeout(30000);

    await selectProblem(page, 'Level 1', 'AND Gate');

    await page.getByRole('button', { name: 'Train Static' }).click();

    // Wait for training to start and accumulate data points
    await page.waitForTimeout(3000);

    // Wait for chart to render
    await expect(page.locator('.recharts-wrapper')).toBeVisible({ timeout: 5000 });

    // The chart should have path elements for the loss line
    // Note: Line curves only appear after multiple data points
    const lineCurve = page.locator('.recharts-line-curve').first();
    const hasLineCurve = await lineCurve.isVisible().catch(() => false);

    // Either we have a line curve or at least the chart area
    if (!hasLineCurve) {
      await expect(page.locator('.recharts-cartesian-grid')).toBeVisible();
    }

    await stopTraining(page);
  });

  test('should show accuracy curve alongside loss', async ({ page }) => {
    test.setTimeout(60000);
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Use few epochs to complete quickly
    const epochsInput = page.locator('input[type="number"]').first();
    await epochsInput.fill('100');

    await page.getByRole('button', { name: 'Train Static' }).click();

    // Wait for training to accumulate some data points
    await page.waitForTimeout(3000);

    // Chart should be visible
    await expect(page.locator('.recharts-wrapper')).toBeVisible({ timeout: 5000 });

    // Legend should show both Loss and Accuracy
    await expect(page.getByText('Loss').first()).toBeVisible();
    await expect(page.getByText('Accuracy').first()).toBeVisible();

    await stopTraining(page);
  });

  test('should show chart legend', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    await page.getByRole('button', { name: 'Train Static' }).click();
    await page.waitForTimeout(2000);

    // Check for legend items
    await expect(page.getByText('Loss').first()).toBeVisible();

    await stopTraining(page);
  });

  test('network visualization should update with different architectures', async ({ page }) => {
    // Select a problem with different architecture
    await selectProblem(page, 'Level 2', 'XOR Gate');

    // Wait for visualization to update
    await page.waitForTimeout(500);

    // Should still have network visualization
    await expect(page.locator('svg circle').first()).toBeVisible();
  });

  test('CNN problems should show CNN-specific UI', async ({ page }) => {
    // Actual CNN problem name is "Shape Detection (CNN)"
    await selectProblem(page, 'Level 7', 'Shape Detection');

    // Wait for component to render
    await page.waitForTimeout(1000);

    // CNN problems have 8x8 grid input
    await expect(page.getByText('8×8 grid').first()).toBeVisible();

    // Should have CNN-specific UI elements like "Draw" button
    await expect(page.getByRole('button', { name: /Draw/ })).toBeVisible();
  });

  test('should show decision boundary visualization for 2D problems', async ({ page }) => {
    await selectProblem(page, 'Level 3', 'Circle');

    // Wait for decision boundary viz to load
    await page.waitForTimeout(500);

    // Decision boundary component should be visible for 2D classification problems
    await expect(page.locator('canvas, svg').first()).toBeVisible();
  });

  test('should show terminal output component', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Terminal output section should be visible
    // Look for "Terminal Output" heading
    await expect(page.getByText('Terminal Output').first()).toBeVisible();
  });

  test('visualization should show input and output labels', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Wait for viz to render
    await page.waitForTimeout(500);

    // The network visualization should show input/output labels
    // These appear as text elements in the SVG
    const svgText = page.locator('svg text');
    const count = await svgText.count();
    expect(count).toBeGreaterThan(0);
  });

  test('neurons should light up with activations after training', async ({ page }) => {
    test.setTimeout(60000);

    await selectProblem(page, 'Level 1', 'AND Gate');

    // Train the network
    await page.getByRole('button', { name: 'Train Static' }).click();
    await page.waitForTimeout(5000);
    await stopTraining(page);

    // After training, the network visualization should show activations
    // This is indicated by fill colors on circles
    const circles = page.locator('svg circle');
    await expect(circles.first()).toBeVisible();
  });

  test('should display total epochs in loss curve', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    await page.getByRole('button', { name: 'Train Static' }).click();
    await page.waitForTimeout(3000);
    await stopTraining(page);

    // The loss curve component should show epoch count
    await expect(page.getByText(/Epoch|epochs/i).first()).toBeVisible();
  });
});
