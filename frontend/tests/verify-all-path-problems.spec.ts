import { test, expect } from '@playwright/test';

test.describe('Verify All Learning Path Problems', () => {
  test('Foundations path has all problems defined correctly', async ({ page }) => {
    // Check API returns all steps with valid problem IDs
    const response = await page.request.get('http://localhost:5000/api/paths/foundations');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    console.log('\n=== Foundations Path Steps ===');

    expect(data.id).toBe('foundations');
    expect(data.steps).toHaveLength(7);

    const expectedSteps = [
      { num: 1, problemId: 'and_gate', title: 'AND Gate' },
      { num: 2, problemId: 'or_gate', title: 'OR Gate' },
      { num: 3, problemId: 'not_gate', title: 'NOT Gate' },
      { num: 4, problemId: 'fail_xor_no_hidden', title: 'Failure' },
      { num: 5, problemId: 'xor', title: 'XOR' },
      { num: 6, problemId: 'xnor', title: 'XNOR' },
      { num: 7, problemId: 'xor_5bit', title: '5-Bit Parity' }
    ];

    for (let i = 0; i < data.steps.length; i++) {
      const step = data.steps[i];
      const expected = expectedSteps[i];

      console.log(`Step ${step.stepNumber}: ${step.title} (${step.problemId})`);

      expect(step.stepNumber).toBe(expected.num);
      expect(step.problemId).toBe(expected.problemId);
      expect(step.title).toContain(expected.title);
      expect(step.learningObjectives).toBeInstanceOf(Array);
      expect(step.hints).toBeInstanceOf(Array);
      expect(typeof step.requiredAccuracy).toBe('number');
    }

    console.log('\n✓ All 7 steps have valid problem IDs and metadata');
  });

  test('verify step 1 problem loads in browser', async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Navigate to Foundations path
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(2000);

    // Verify AND gate loaded
    await expect(page.locator('h3:has-text("AND Gate")')).toBeVisible();
    await expect(page.locator('text=/Arch: \\[2-1\\]/')).toBeVisible();
    await expect(page.locator('text=Input A')).toBeVisible();
    await expect(page.locator('text=Input B')).toBeVisible();

    // Verify input panel is not showing "Select a problem"
    const inputPanel = await page.locator('text=Input').first().locator('..').textContent();
    expect(inputPanel).not.toContain('Select a problem to configure inputs');

    console.log('\n✓ Step 1 (AND Gate) loads correctly in browser');

    // Take final screenshot
    await page.screenshot({ path: 'foundations-step1-verified.png', fullPage: true });
  });

  test('verify all other paths have valid problem IDs', async ({ page }) => {
    const paths = [
      'deep-learning-basics',
      'multi-class-mastery',
      'convolutional-vision',
      'pitfall-prevention',
      'research-frontier'
    ];

    for (const pathId of paths) {
      console.log(`\n=== Checking ${pathId} ===`);

      const response = await page.request.get(`http://localhost:5000/api/paths/${pathId}`);
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      console.log(`  Path: ${data.name}`);
      console.log(`  Steps: ${data.steps.length}`);

      for (const step of data.steps) {
        console.log(`    ${step.stepNumber}. ${step.problemId} - ${step.title}`);
        expect(step.problemId).toBeTruthy();
        expect(step.problemId).not.toBe('undefined');
      }
    }

    console.log('\n✓ All paths have valid problem IDs');
  });
});
