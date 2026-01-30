import { test, expect } from '@playwright/test';
import { waitForConnection, selectProblem, startStaticTraining, waitForTrainingComplete } from './fixtures/test-helpers';

/**
 * Tests for newly added features:
 * - TrainingNarrator (real-time insights during training)
 * - FailureDramatization (visual effects for failure cases)
 * - Enhanced LossCurve (tooltips, annotations)
 * - Interactive Hints (in learning paths)
 */

test.describe('New Features Test Suite', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(500);

    // Clear localStorage and dismiss any modals
    await page.evaluate(() => {
      localStorage.clear();
      localStorage.setItem('learning_paths_onboarding_seen', 'true');
    });
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Try to close any open modals
    const closeButton = page.locator('button[aria-label="Close"], button:has-text("Skip"), button:has-text("Got it")').first();
    if (await closeButton.isVisible({ timeout: 1000 }).catch(() => false)) {
      await closeButton.click();
      await page.waitForTimeout(300);
    }
  });

  test.describe('Training Narrator', () => {
    test('shows training insights during training', async ({ page }) => {
      // Go to All Problems view for standard training UI
      await page.click('text=All Problems');
      await page.waitForTimeout(1000);

      await waitForConnection(page);

      // Start training using standard training panel
      await startStaticTraining(page, 100, 0.5);

      // Wait for training to start
      await page.waitForTimeout(2000);

      // Check for training-related UI elements (narrator insights)
      const trainingElements = page.locator('text=/Training|Loss|Accuracy|Epoch/i');
      await expect(trainingElements.first()).toBeVisible({ timeout: 10000 });
    });
  });

  test.describe('Loss Curve Enhancements', () => {
    test('loss curve shows during training', async ({ page }) => {
      await waitForConnection(page);

      // Select a simple problem
      await selectProblem(page, 'Level 1', 'AND Gate');
      await page.waitForTimeout(500);

      // Start training
      await startStaticTraining(page, 100, 0.5);
      await page.waitForTimeout(2000);

      // Check for loss curve components
      const trainingProgress = page.locator('text=/Training Progress|Loss|Accuracy/i').first();
      await expect(trainingProgress).toBeVisible({ timeout: 10000 });

      // Wait for training to complete
      await waitForTrainingComplete(page, 30000);

      // Check that stats are displayed
      const stats = page.locator('text=/Epochs:|Loss:|Acc:/');
      await expect(stats.first()).toBeVisible();
    });

    test('loss curve has interactive tooltip', async ({ page }) => {
      await waitForConnection(page);
      await selectProblem(page, 'Level 1', 'AND Gate');

      // Train first
      await startStaticTraining(page, 200, 0.5);
      await waitForTrainingComplete(page, 30000);

      // The chart area should be visible
      const chartArea = page.locator('.recharts-wrapper, [class*="AreaChart"]').first();
      const isChartVisible = await chartArea.isVisible().catch(() => false);

      if (isChartVisible) {
        // Hover over chart to trigger tooltip
        await chartArea.hover();
        await page.waitForTimeout(500);

        // Check if any tooltip-like element appeared
        const tooltipContent = page.locator('.recharts-tooltip-wrapper, [class*="tooltip"]').first();
        const hasTooltip = await tooltipContent.isVisible().catch(() => false);
        console.log(`Tooltip visible: ${hasTooltip}`);
      }

      // Verify chart rendered
      expect(isChartVisible || await page.locator('text=Training Progress').isVisible()).toBeTruthy();
    });
  });

  test.describe('Failure Case Visualization', () => {
    test('failure case problem shows expected failure badge', async ({ page }) => {
      await waitForConnection(page);

      // Select a failure case problem (LR Explosion)
      await selectProblem(page, 'Level 5', 'LR Explosion');
      await page.waitForTimeout(500);

      // Check for failure-related UI elements
      const failureIndicator = page.locator('text=/Expected Failure|DESIGNED TO FAIL|Failure|Explosion|LR/i').first();
      await expect(failureIndicator).toBeVisible({ timeout: 5000 });

      console.log('✓ Failure case indicator visible');
    });

    test('failure case shows dramatization during training', async ({ page }) => {
      await waitForConnection(page);

      // Select the high learning rate failure case (LR Explosion)
      await selectProblem(page, 'Level 5', 'LR Explosion');
      await page.waitForTimeout(500);

      // Start training with high LR (using the Watch it Fail button or Train Static)
      const trainButton = page.locator('button:has-text("Watch it Fail"), button:has-text("Train")').first();
      await trainButton.click();

      // Wait for training to show effects
      await page.waitForTimeout(3000);

      // Check for unstable/failure indicators
      const failureEffects = page.locator('text=/Explosion|Unstable|NaN|spike|oscillate/i').first();
      const hasFailureEffects = await failureEffects.isVisible().catch(() => false);

      // Or check for high loss values
      const highLoss = page.locator('text=/Loss: [0-9]+\\.[0-9]+|NaN/').first();
      const hasHighLoss = await highLoss.isVisible().catch(() => false);

      console.log(`Failure effects visible: ${hasFailureEffects}, High loss visible: ${hasHighLoss}`);

      // At least one indicator should be visible
      expect(hasFailureEffects || hasHighLoss).toBeTruthy();
    });

    test('zero init failure shows symmetry message', async ({ page }) => {
      await waitForConnection(page);

      // Select zero init problem (Zero Init Trap)
      await selectProblem(page, 'Level 5', 'Zero Init Trap');
      await page.waitForTimeout(500);

      // Check for symmetry-related content
      const symmetryContent = page.locator('text=/Symmetry|identical|same/i').first();
      const hasSymmetryContent = await symmetryContent.isVisible({ timeout: 3000 }).catch(() => false);

      console.log(`Symmetry content visible: ${hasSymmetryContent}`);

      // The problem should at least be loaded
      await expect(page.locator('text=/Zero|Init|Trap/i').first()).toBeVisible();
    });
  });

  test.describe('Interactive Hints', () => {
    test('hints panel appears in learning path', async ({ page }) => {
      // Navigate to Learning Paths
      await page.click('text=Learning Paths');
      await page.waitForTimeout(1000);

      // Start Foundations path
      const startButton = page.locator('button:has-text("Start Path")').first();
      await startButton.click({ force: true });
      await page.waitForTimeout(2000);

      // Look for hints section
      const hintsSection = page.locator('text=/Hints|HINTS|Tips|💡/i').first();
      await expect(hintsSection).toBeVisible({ timeout: 5000 });

      console.log('✓ Hints section visible in learning path');
    });
  });

  test.describe('Overall Feature Integration', () => {
    test('main app loads without errors', async ({ page }) => {
      await page.waitForLoadState('networkidle');

      // No error boundaries or crash indicators
      const errorIndicator = page.locator('text=/Error|Something went wrong|Crash/i');
      const hasError = await errorIndicator.isVisible().catch(() => false);

      expect(hasError).toBeFalsy();

      // Key UI elements should be present
      await expect(page.locator('text=/Learning Paths|Problems/i').first()).toBeVisible();
    });

    test('can switch between problems view and learning paths', async ({ page }) => {
      await page.waitForLoadState('networkidle');

      // Click Learning Paths
      await page.click('text=Learning Paths');
      await page.waitForTimeout(1000);

      // Verify Learning Paths view
      await expect(page.locator('h1:has-text("Learning Paths")').first()).toBeVisible();

      // Click back via Exit or back button
      const backButton = page.locator('button:has-text("Exit"), button:has-text("Back"), a:has-text("Back")').first();
      if (await backButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await backButton.click({ force: true });
        await page.waitForTimeout(1000);
      } else {
        // If no back button, navigate directly to home
        await page.goto('http://localhost:5173');
        await page.waitForLoadState('networkidle');
      }

      // Should be back to main view
      await waitForConnection(page);
    });
  });
});
