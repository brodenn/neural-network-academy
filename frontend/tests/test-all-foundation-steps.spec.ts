import { test, expect } from '@playwright/test';

test.describe('Foundations Learning Path - All Steps', () => {
  test('verify all 7 steps load correctly', async ({ page }) => {
    // Navigate to app
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Click Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Click Start Path on Foundations
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(2000);

    const steps = [
      { num: 1, name: 'AND Gate', inputCount: 2 },
      { num: 2, name: 'OR Gate', inputCount: 2 },
      { num: 3, name: 'NOT Gate', inputCount: 1 },
      { num: 4, name: 'XOR', inputCount: 2 },  // Might fail intentionally
      { num: 5, name: 'XOR', inputCount: 2 },
      { num: 6, name: 'XNOR', inputCount: 2 },
      { num: 7, name: 'Parity', inputCount: 5 },  // 5-bit parity
    ];

    for (const step of steps) {
      console.log(`\n=== Checking Step ${step.num}: ${step.name} ===`);

      // Wait for step to load
      await page.waitForTimeout(1500);

      // Take screenshot
      await page.screenshot({ path: `step-${step.num}-${step.name.toLowerCase().replace(' ', '-')}.png`, fullPage: true });

      // Check that we're on the right step
      const stepText = await page.locator('text=/Step \\d+ of 7/').textContent();
      console.log(`Step indicator: ${stepText}`);
      expect(stepText).toContain(`Step ${step.num} of 7`);

      // Check that problem title is visible
      const titleVisible = await page.locator(`h3:has-text("${step.name}")`).count();
      console.log(`Title "${step.name}" visible: ${titleVisible > 0}`);
      expect(titleVisible).toBeGreaterThan(0);

      // Check that inputs are loaded (not showing "Select a problem")
      const inputPanel = await page.locator('text=Input').first().locator('..').textContent();
      console.log(`Input panel: ${inputPanel?.substring(0, 50)}...`);
      expect(inputPanel).not.toContain('Select a problem to configure inputs');

      // Check architecture is displayed
      const arch = await page.locator('text=/Arch: \\[.*\\]/').textContent().catch(() => 'not found');
      console.log(`Architecture: ${arch}`);
      expect(arch).not.toBe('not found');

      // Move to next step if not last
      if (step.num < 7) {
        // Click next step in progress bar (step number + 1)
        await page.locator(`text="Step ${step.num + 1}"`).click();
      }
    }

    console.log('\n=== All steps loaded successfully! ===');
  });
});
