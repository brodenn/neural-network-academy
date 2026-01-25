import { test, expect } from '@playwright/test';

test.describe('Final Learning Path Verification', () => {
  test('verify Foundations path problem loads completely', async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    console.log('\n=== Testing Foundations Learning Path ===\n');

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Verify path card is visible
    const foundationsCard = await page.locator('text=Foundations').first();
    await expect(foundationsCard).toBeVisible();
    console.log('✓ Foundations path card visible');

    // Click Start Path
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(3000);

    // Verify we're in the path detail view
    await expect(page.locator('text="Step 1 of 7"')).toBeVisible();
    console.log('✓ Entered path detail view');

    // Verify step info is loaded
    const stepTitle = await page.locator('h3').filter({ hasText: 'AND Gate' }).first();
    await expect(stepTitle).toBeVisible();
    console.log('✓ Step title displayed: AND Gate - Single Neuron Capability');

    // Verify problem is fully loaded - check for specific elements
    const checks = [
      { name: 'Step progress bar', locator: 'text="Step 1"' },
      { name: 'Learning objectives', locator: 'text="LEARNING OBJECTIVES"' },
      { name: 'Required accuracy', locator: 'text="95%"' },
      { name: 'Network architecture', locator: 'text=/Arch: \\[2-1\\]/' },
      { name: 'Input controls', locator: 'text="Input (2 values)"' },
      { name: 'Training panel', locator: 'text="Training Controls"' },
      { name: 'Output display', locator: 'text="Output"' },
      { name: 'Network visualization', locator: 'text="Network Architecture"' },
    ];

    for (const check of checks) {
      const element = page.locator(check.locator).first();
      await expect(element).toBeVisible({ timeout: 5000 });
      console.log(`✓ ${check.name} visible`);
    }

    // Verify inputs are interactive (not placeholder text)
    const inputPanel = await page.locator('div').filter({ hasText: 'Input' }).first().textContent();
    expect(inputPanel).toContain('Input (2 values)');
    expect(inputPanel).not.toContain('Select a problem to configure inputs');
    console.log('✓ Input panel fully loaded with controls');

    // Take screenshot
    await page.screenshot({ path: 'final-path-verification.png', fullPage: true });
    console.log('✓ Screenshot saved: final-path-verification.png');

    console.log('\n=== SUCCESS: Foundations path loads completely! ===\n');
  });

  test('verify API returns all path problems correctly', async ({ page }) => {
    console.log('\n=== Verifying All Paths via API ===\n');

    const paths = [
      { id: 'foundations', expectedSteps: 7 },
      { id: 'deep-learning-basics', expectedSteps: 10 }, // Updated: added linear and two_blobs
      { id: 'multi-class-mastery', expectedSteps: 4 },
      { id: 'convolutional-vision', expectedSteps: 3 },
      { id: 'pitfall-prevention', expectedSteps: 6 },
      { id: 'research-frontier', expectedSteps: 4 },
    ];

    for (const path of paths) {
      const response = await page.request.get(`http://localhost:5000/api/paths/${path.id}`);
      expect(response.ok()).toBeTruthy();

      const data = await response.json();

      console.log(`\n${data.name}:`);
      console.log(`  Expected ${path.expectedSteps} steps, got ${data.steps.length}`);
      expect(data.steps).toHaveLength(path.expectedSteps);

      // Verify each step has required fields
      for (const step of data.steps) {
        expect(step.stepNumber).toBeGreaterThan(0);
        expect(step.problemId).toBeTruthy();
        expect(step.problemId).not.toBe('undefined');
        expect(step.title).toBeTruthy();
        expect(Array.isArray(step.learningObjectives)).toBeTruthy();
        expect(Array.isArray(step.hints)).toBeTruthy();
        expect(typeof step.requiredAccuracy).toBe('number');

        console.log(`    ${step.stepNumber}. ${step.problemId} ✓`);
      }
    }

    console.log('\n✓ All 6 paths have valid steps with correct problem IDs');
    console.log('✓ Total steps across all paths: 32');
  });

  test('verify problem can be selected via API', async ({ page }) => {
    console.log('\n=== Testing Problem Selection API ===\n');

    // Test selecting AND gate
    const response = await page.request.post('http://localhost:5000/api/problems/and_gate/select');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.success).toBeTruthy();
    expect(data.problem_id).toBe('and_gate');

    console.log('✓ AND gate selected successfully via API');
    console.log(`✓ Response: ${JSON.stringify(data, null, 2)}`);

    // Verify network state updated
    const networkResponse = await page.request.get('http://localhost:5000/api/network');
    expect(networkResponse.ok()).toBeTruthy();

    const networkData = await networkResponse.json();
    expect(networkData.architecture).toEqual([2, 1]);
    console.log('✓ Network architecture updated to [2, 1] for AND gate');
  });
});
