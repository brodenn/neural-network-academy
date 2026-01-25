import { test, expect } from '@playwright/test';

test.describe('Verify Problem Loading in Path', () => {
  test('problem selection API is called when entering path', async ({ page }) => {
    // Track API calls
    const apiCalls: string[] = [];
    page.on('request', (request) => {
      const url = request.url();
      if (url.includes('localhost:5000/api/')) {
        apiCalls.push(`${request.method()} ${url.split('localhost:5000')[1]}`);
      }
    });

    // Navigate to app
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Clear previous calls
    apiCalls.length = 0;

    // Click Learning Paths tab
    await page.click('div[role="tab"]:has-text("Learning Paths")');
    await page.waitForTimeout(500);

    // Click Start Path button on Foundations card
    await page.locator('div').filter({ hasText: /^Foundations/ })
      .locator('button:has-text("Start Path")').first().click();

    // Wait for path to load
    await page.waitForTimeout(2000);

    // Log all API calls
    console.log('\n=== API Calls ===');
    apiCalls.forEach(call => console.log(call));

    // Check if problem selection was called
    const problemSelectCalls = apiCalls.filter(call =>
      call.includes('POST /api/problems/') && call.includes('/select')
    );

    console.log(`\nProblem select calls: ${problemSelectCalls.length}`);
    expect(problemSelectCalls.length).toBeGreaterThan(0);
    expect(problemSelectCalls[0]).toContain('/api/problems/and/select');
  });

  test('verify inputs are loaded after entering path', async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Navigate to Foundations path
    await page.click('div[role="tab"]:has-text("Learning Paths")');
    await page.waitForTimeout(500);
    await page.locator('div').filter({ hasText: /^Foundations/ })
      .locator('button:has-text("Start Path")').first().click();

    // Wait for problem to load
    await page.waitForTimeout(2000);

    // Take screenshot for debugging
    await page.screenshot({ path: 'verify-path-loaded.png', fullPage: true });

    // Check that inputs are NOT showing "Select a problem"
    const inputPanel = await page.locator('div').filter({ hasText: 'Input' }).first();
    const inputText = await inputPanel.textContent();

    console.log(`\nInput panel text: ${inputText}`);
    expect(inputText).not.toContain('Select a problem to configure inputs');

    // Check that we have input buttons (AND gate has 2 inputs)
    const inputButtons = await page.locator('button').filter({ hasText: /^[01]$/ }).count();
    console.log(`Input buttons found: ${inputButtons}`);
    expect(inputButtons).toBeGreaterThanOrEqual(2);
  });
});
