import { test, expect } from '@playwright/test';

test.describe('Debug Learning Path Loading', () => {
  test('verify problem loads when entering path', async ({ page }) => {
    // Listen to network requests
    const requests: { url: string; method: string }[] = [];
    page.on('request', (request) => {
      const url = request.url();
      if (url.includes('localhost:5000')) {
        requests.push({ url, method: request.method() });
      }
    });

    // Go to app
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Click Learning Paths
    await page.click('div[role="tab"]:has-text("Learning Paths")');
    await page.waitForTimeout(500);

    // Click Foundations path "Start Learning" button
    const foundationsCard = page.locator('div').filter({ hasText: /^Foundations/ }).first();
    await foundationsCard.locator('button:has-text("Start Learning")').click();
    await page.waitForTimeout(1000);

    // Log all requests
    console.log('\n=== Network Requests ===');
    requests.forEach(req => {
      console.log(`${req.method} ${req.url}`);
    });

    // Check if problem selection endpoint was called
    const problemSelectCalls = requests.filter(r =>
      r.url.includes('/api/problems/') && r.url.includes('/select') && r.method === 'POST'
    );
    console.log(`\nProblem select calls: ${problemSelectCalls.length}`);
    problemSelectCalls.forEach(r => console.log(`  - ${r.url}`));

    // Check current page state
    const inputPanelText = await page.locator('div:has-text("Input")').first().textContent();
    const outputPanelText = await page.locator('div:has-text("Output")').first().textContent();

    console.log('\n=== Panel States ===');
    console.log(`Input panel: ${inputPanelText}`);
    console.log(`Output panel: ${outputPanelText}`);

    // Check if current problem info is loaded
    const currentProblemTitle = await page.locator('h3:has-text("AND Gate")').count();
    console.log(`\nAND Gate title found: ${currentProblemTitle > 0}`);

    // Verify problem was selected
    expect(problemSelectCalls.length).toBeGreaterThan(0);

    // Verify input panel is not showing "Select a problem"
    expect(inputPanelText).not.toContain('Select a problem to configure inputs');
  });

  test('check step 1 details', async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Navigate to Foundations path
    await page.click('div[role="tab"]:has-text("Learning Paths")');
    await page.waitForTimeout(500);

    const foundationsCard = page.locator('div').filter({ hasText: /^Foundations/ }).first();
    await foundationsCard.locator('button:has-text("Start Learning")').click();
    await page.waitForTimeout(2000);

    // Take screenshot
    await page.screenshot({ path: 'path-step1-debug.png', fullPage: true });

    // Check what's displayed
    const stepTitle = await page.locator('text="Step 1 of"').textContent();
    const problemTitle = await page.locator('h3').filter({ hasText: 'AND Gate' }).first().textContent();

    console.log('\n=== Step 1 Info ===');
    console.log(`Step title: ${stepTitle}`);
    console.log(`Problem title: ${problemTitle}`);

    // Check network architecture display
    const archText = await page.locator('text=/Arch:.*\\[.*\\]/').textContent().catch(() => 'not found');
    console.log(`Architecture: ${archText}`);

    // Check if inputs are available
    const inputButtons = await page.locator('button:has-text("0")').count();
    console.log(`Input buttons found: ${inputButtons}`);
  });
});
