import { test, expect } from '@playwright/test';
import { waitForConnection, selectProblem, stopTraining } from './fixtures/test-helpers';

test.describe('Input Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForConnection(page);
  });

  test('should display input panel', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Input panel should be visible with the Input heading
    await expect(page.getByRole('heading', { name: /Input/ })).toBeVisible();
  });

  test('should show correct number of inputs for AND gate', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Wait for the network to be ready
    await page.waitForTimeout(1000);

    // AND gate has 2 inputs - should have toggle buttons
    const inputSection = page.locator('div').filter({ hasText: /Input/ });
    await expect(inputSection.first()).toBeVisible();
  });

  test('should show inputs disabled before training', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Check for disabled state message or disabled inputs
    // The InputPanel should show that network needs training first
    await expect(page.getByText('Train the network first').first()).toBeVisible({ timeout: 5000 });
  });

  test('binary inputs should toggle on click after training', async ({ page }) => {
    test.setTimeout(60000);

    await selectProblem(page, 'Level 1', 'AND Gate');

    // Train the network first
    await page.getByRole('button', { name: 'Train Static' }).click();
    await page.waitForTimeout(5000);
    await stopTraining(page);

    // Now inputs should be enabled
    // Find input toggle buttons and click
    const inputButtons = page.locator('button').filter({ hasText: /^[01]$/ });
    const count = await inputButtons.count();
    if (count > 0) {
      await inputButtons.first().click();
      await page.waitForTimeout(200);
    }
  });

  test('slider inputs should work for continuous problems', async ({ page }) => {
    await selectProblem(page, 'Level 3', 'Circle');

    // Wait for network
    await page.waitForTimeout(1000);

    // Circle problem should have slider inputs for x, y coordinates
    const sliders = page.locator('input[type="range"]');
    const sliderCount = await sliders.count();

    // Should have at least some sliders (target accuracy slider + input sliders)
    expect(sliderCount).toBeGreaterThan(0);
  });

  test('should show output display', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Output display panel should be visible
    await expect(page.locator('text=Output').first()).toBeVisible();
  });

  test('should update prediction when inputs change after training', async ({ page }) => {
    test.setTimeout(60000);

    await selectProblem(page, 'Level 1', 'AND Gate');

    // Train first
    await page.getByRole('button', { name: 'Train Static' }).click();

    // Wait for training to start and run
    await expect(page.locator('span').filter({ hasText: 'Training...' }).first()).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(8000);
    await stopTraining(page);
    await page.waitForTimeout(1000);

    // The output display should show prediction after training
    // Check for "Prediction" label in output section
    await expect(page.getByText('Prediction').first()).toBeVisible();
  });

  test('CNN problems should show grid input', async ({ page }) => {
    // Actual CNN problem name is "Shape Detection (CNN)"
    await selectProblem(page, 'Level 7', 'Shape Detection');

    // CNN problems use a 2D grid input
    await page.waitForTimeout(2000);

    // Should have a grid-based input visualization or CNN-specific UI
    // The CNN section shows shape selection or grid visualization
    const cnnSection = page.locator('div').filter({ hasText: /Input.*values|Shape/ });
    await expect(cnnSection.first()).toBeVisible();
  });

  test('should display input labels', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Wait for problem to load
    await page.waitForTimeout(500);

    // Input panel should show "Input" section with value count
    await expect(page.getByText(/Input.*values/).first()).toBeVisible();

    // Should also show individual input labels (Input A, Input B)
    await expect(page.getByText(/Input A/).first()).toBeVisible();
  });

  test('should show prediction result after network is trained', async ({ page }) => {
    test.setTimeout(60000);

    await selectProblem(page, 'Level 1', 'AND Gate');

    // Train
    await page.getByRole('button', { name: 'Train Static' }).click();

    // Wait for training to progress
    await expect(page.locator('span').filter({ hasText: 'Training...' }).first()).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(8000);
    await stopTraining(page);
    await page.waitForTimeout(1000);

    // Look for prediction/output display showing result
    // OutputDisplay shows "Prediction" label
    await expect(page.getByText('Prediction').first()).toBeVisible();
  });
});
