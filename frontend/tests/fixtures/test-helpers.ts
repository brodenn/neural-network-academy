import { Page, expect } from '@playwright/test';

/**
 * Wait for the WebSocket connection to be established
 */
export async function waitForConnection(page: Page) {
  await expect(page.getByText('Connected')).toBeVisible({ timeout: 10000 });
}

/**
 * Escape special regex characters in a string
 */
function escapeRegex(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Select a problem by opening the dropdown menu, expanding its level, and clicking on the problem
 */
export async function selectProblem(page: Page, level: string, problem: string) {
  // Open the dropdown menu by clicking the problem selector button (in header)
  const headerDropdown = page.locator('header button').first();
  await headerDropdown.click();
  // Wait for dropdown to appear
  await page.waitForTimeout(200);

  // Click on the level header to expand it (inside the dropdown menu)
  // w-80 distinguishes problem selector dropdown from keyboard shortcuts dropdown
  const dropdownMenu = page.locator('.absolute.bg-gray-800.w-80');
  const levelButton = dropdownMenu.getByRole('button', { name: new RegExp(escapeRegex(level)) });
  await levelButton.click();
  // Wait for level expansion animation
  await page.waitForTimeout(200);

  // Click on the problem (inside the dropdown menu, look for button with category badge)
  // Problem buttons have format: "Problem Name category ★★"
  const problemButton = dropdownMenu.locator('button').filter({ hasText: new RegExp(`^${escapeRegex(problem)}`) }).first();
  await problemButton.click();
  // Wait for problem to load
  await page.waitForTimeout(500);
}

/**
 * Start static training with given parameters
 */
export async function startStaticTraining(page: Page, epochs = 1000, learningRate = 0.5) {
  const epochsInput = page.locator('input[type="number"]').first();
  const lrInput = page.locator('input[type="number"]').nth(1);

  await epochsInput.fill(epochs.toString());
  await lrInput.fill(learningRate.toString());

  // Find and click the Train Static button (might say "Watch it Fail!" for failure cases)
  const trainButton = page.getByRole('button', { name: /Train Static/ });
  await trainButton.click();
}

/**
 * Start adaptive training
 */
export async function startAdaptiveTraining(page: Page, targetAccuracy = 0.99) {
  // Set target accuracy via slider
  const slider = page.locator('input[type="range"]').first();
  await slider.fill((targetAccuracy * 100).toString());

  await page.getByRole('button', { name: 'Train Adaptive' }).click();
}

/**
 * Stop training
 */
export async function stopTraining(page: Page) {
  try {
    const stopButton = page.getByRole('button', { name: /Stop Training/ });
    // Wait for button to be visible and stable before clicking
    const isVisible = await stopButton.isVisible().catch(() => false);
    if (isVisible) {
      // Use force:true to click even if element is being detached
      await stopButton.click({ force: true, timeout: 2000 }).catch(() => {
        // Ignore click errors - training may have completed
      });
      await page.waitForTimeout(500);
    }
  } catch {
    // Training may have already stopped
  }
}

/**
 * Reset the network
 */
export async function resetNetwork(page: Page) {
  await page.getByRole('button', { name: 'Reset Network' }).click();
  await page.waitForTimeout(500);
}

/**
 * Get current accuracy from the display
 */
export async function getCurrentAccuracy(page: Page): Promise<number> {
  // Try to find accuracy in multiple ways

  // Method 1: Look for the summary line text directly
  // Format: "Epochs: X | Loss: X.XX | Acc: XX.X%"
  const summaryLine = page.locator('text=/Epochs:.*Acc:/');
  if (await summaryLine.isVisible().catch(() => false)) {
    const summaryText = await summaryLine.textContent();
    const accMatch = summaryText?.match(/Acc:\s*([\d.]+)%/);
    if (accMatch) {
      return parseFloat(accMatch[1]);
    }
  }

  // Method 2: Fall back to searching full page text
  const allText = await page.textContent('body');

  // Try multiple patterns
  const patterns = [
    /Acc:\s*([\d.]+)%/,           // Summary line
    /Accuracy[\s\S]{0,15}?([\d.]+)%/, // Stats panel
  ];

  for (const pattern of patterns) {
    const match = allText?.match(pattern);
    if (match) {
      const value = parseFloat(match[1]);
      // Return even if 0 - that's a valid accuracy value
      return value;
    }
  }

  return 0;
}

/**
 * Get current epoch from the display
 */
export async function getCurrentEpoch(page: Page): Promise<number> {
  // Look for Epoch in multiple places:
  // 1. Summary line: "Epochs: XXX | Loss..."
  // 2. Stats panel: "Epoch" label with value
  const allText = await page.textContent('body');

  // Try summary line first (more reliable during training)
  const patterns = [
    /Epochs:\s*(\d+)/,           // Summary line
    /Epoch\s*(\d+)/,             // Direct adjacency
    /Epoch[\s\S]{0,10}?(\d+)/,   // With small gap
  ];

  for (const pattern of patterns) {
    const match = allText?.match(pattern);
    if (match && parseInt(match[1], 10) > 0) {
      return parseInt(match[1], 10);
    }
  }

  // Return 0 if nothing matched
  return 0;
}

/**
 * Wait for training to reach a certain accuracy
 */
export async function waitForAccuracy(page: Page, targetAccuracy: number, timeout = 60000) {
  const startTime = Date.now();
  while (Date.now() - startTime < timeout) {
    const accuracy = await getCurrentAccuracy(page);
    if (accuracy >= targetAccuracy) return accuracy;
    await page.waitForTimeout(500);
  }
  throw new Error(`Training did not reach ${targetAccuracy}% accuracy within ${timeout}ms`);
}

/**
 * Wait for training to complete (either success or stopped)
 */
export async function waitForTrainingComplete(page: Page, timeout = 60000) {
  // Wait for "Ready" status in header (more reliable than checking Training... absence)
  await expect(page.getByText('Ready').first()).toBeVisible({ timeout });
}

/**
 * Check if training is in progress
 */
export async function isTrainingInProgress(page: Page): Promise<boolean> {
  // Check for the header status badge
  const trainingBadge = page.locator('span').filter({ hasText: 'Training...' }).first();
  return await trainingBadge.isVisible();
}
