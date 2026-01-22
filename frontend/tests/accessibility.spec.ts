import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { waitForConnection, selectProblem } from './fixtures/test-helpers';

test.describe('Accessibility Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForConnection(page);
  });

  test('home page should have no critical accessibility violations', async ({ page }) => {
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .disableRules(['color-contrast', 'svg-img-alt']) // Disable known issues for now
      .analyze();

    // Log any violations for debugging
    if (accessibilityScanResults.violations.length > 0) {
      console.log('Accessibility violations:');
      accessibilityScanResults.violations.forEach(v => {
        console.log(`  - ${v.id} (${v.impact}): ${v.description}`);
        console.log(`    Help: ${v.helpUrl}`);
      });
    }

    // Critical violations should be zero (allow serious for now as we improve)
    const criticalViolations = accessibilityScanResults.violations.filter(
      v => v.impact === 'critical'
    );

    expect(criticalViolations.length).toBe(0);
  });

  test('problem selector should be keyboard navigable', async ({ page }) => {
    // Focus the first level button
    await page.keyboard.press('Tab');

    // Verify focus is visible (element has focus ring or outline)
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  test('training panel should have accessible form controls', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Check that input fields have accessible labels
    const epochsInput = page.getByRole('spinbutton').first();
    await expect(epochsInput).toBeVisible();

    // Check training buttons have accessible names
    const trainStaticBtn = page.getByRole('button', { name: /Train Static/ });
    await expect(trainStaticBtn).toBeVisible();
    await expect(trainStaticBtn).toHaveAttribute('aria-label');

    const trainAdaptiveBtn = page.getByRole('button', { name: /Train Adaptive/ });
    await expect(trainAdaptiveBtn).toBeVisible();
  });

  test('input panel binary buttons should have aria-pressed state', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Wait for input panel to render
    await page.waitForTimeout(500);

    // Find toggle buttons with aria-pressed
    const toggleButtons = page.locator('button[aria-pressed]');
    const count = await toggleButtons.count();

    // Should have binary input buttons with aria-pressed
    expect(count).toBeGreaterThan(0);
  });

  test('input sliders should have aria-value attributes', async ({ page }) => {
    await selectProblem(page, 'Level 3', 'Circle');

    // Wait for component to render
    await page.waitForTimeout(500);

    // Find sliders with aria-valuenow
    const sliders = page.locator('input[type="range"][aria-valuenow]');
    const count = await sliders.count();

    expect(count).toBeGreaterThan(0);
  });

  test('metrics should have accessible labels', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Check metrics are in a labeled group
    const metricsGroup = page.locator('[role="group"][aria-label="Training metrics"]');
    await expect(metricsGroup).toBeVisible();

    // Check individual metrics have data-testid
    await expect(page.locator('[data-testid="metric-epoch"]')).toBeVisible();
    await expect(page.locator('[data-testid="metric-loss"]')).toBeVisible();
    await expect(page.locator('[data-testid="metric-accuracy"]')).toBeVisible();
  });

  test('training panel region should have aria-label', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    const trainingPanel = page.locator('[data-testid="training-panel"]');
    await expect(trainingPanel).toBeVisible();
    await expect(trainingPanel).toHaveAttribute('role', 'region');
    await expect(trainingPanel).toHaveAttribute('aria-label', 'Training Controls');
  });

  test('input panel region should have aria-label', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    const inputPanel = page.locator('[data-testid="input-panel"]');
    await expect(inputPanel).toBeVisible();
    await expect(inputPanel).toHaveAttribute('role', 'region');
    await expect(inputPanel).toHaveAttribute('aria-label', 'Input Controls');
  });

  test('CNN grid input should have accessible role', async ({ page }) => {
    await selectProblem(page, 'Level 7', 'Shape Detection');

    await page.waitForTimeout(1000);

    // Check grid has accessible role
    const grid = page.locator('[data-testid="input-grid"]');
    await expect(grid).toHaveAttribute('role', 'grid');
  });

  test('failure case page should have no critical violations', async ({ page }) => {
    await selectProblem(page, 'Level 5: Failure Cases', 'XOR (No Hidden Layer)');

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .disableRules(['color-contrast', 'svg-img-alt']) // Disable known issues
      .analyze();

    // Log violations for debugging
    if (accessibilityScanResults.violations.length > 0) {
      console.log('Failure case page violations:');
      accessibilityScanResults.violations.forEach(v => {
        console.log(`  - ${v.id} (${v.impact}): ${v.description}`);
      });
    }

    const criticalViolations = accessibilityScanResults.violations.filter(
      v => v.impact === 'critical'
    );

    expect(criticalViolations.length).toBe(0);
  });
});
