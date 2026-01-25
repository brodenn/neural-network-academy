import { test, expect } from '@playwright/test';

test('manually check all steps in Foundations path', async ({ page }) => {
  // Navigate to app
  await page.goto('http://localhost:5173');
  await page.waitForLoadState('networkidle');

  // Click Learning Paths
  await page.click('text=Learning Paths');
  await page.waitForTimeout(1000);

  // Take screenshot of path selector
  await page.screenshot({ path: 'step-0-path-selector.png', fullPage: true });

  // Click Start Path on Foundations
  await page.click('button:has-text("Start Path")');
  await page.waitForTimeout(3000);

  // Take screenshot of step 1
  await page.screenshot({ path: 'step-1-and-gate.png', fullPage: true });

  // Check if problem is loaded
  const inputText = await page.locator('text=Input').first().locator('..').textContent();
  console.log(`\nStep 1 Input Panel: ${inputText}`);

  // Check architecture
  const archText = await page.locator('text=/Arch:.*\\[.*\\]/').textContent().catch(() => 'not found');
  console.log(`Step 1 Architecture: ${archText}`);

  // Check if we can find input buttons
  const inputButtons = await page.locator('button[class*="bg-gray"]').count();
  console.log(`Step 1 Input Buttons: ${inputButtons}`);

  // Check if step info is visible
  const stepTitle = await page.locator('h3').filter({ hasText: 'AND Gate' }).textContent().catch(() => 'not found');
  console.log(`Step 1 Title: ${stepTitle}`);

  // Wait and check for any console errors
  await page.waitForTimeout(1000);

  console.log('\n=== Test Complete ===');
  console.log('Check step-1-and-gate.png to see if problem loaded correctly');
});
