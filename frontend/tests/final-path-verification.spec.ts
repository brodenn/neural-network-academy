import { test, expect } from '@playwright/test';

test.describe('Final Learning Path Verification', () => {
  test('verify Foundations path problem loads completely', async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('networkidle');

    // Dismiss onboarding modal if present
    await page.evaluate(() => localStorage.setItem('learning_paths_onboarding_seen', 'true'));

    console.log('\n=== Testing Foundations Learning Path ===\n');

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Close any modal overlays that might be blocking
    const closeButton = page.locator('button[aria-label="Close"], button:has-text("Skip"), button:has-text("Got it"), button:has-text("Get Started")').first();
    if (await closeButton.isVisible({ timeout: 1500 }).catch(() => false)) {
      await closeButton.click();
      await page.waitForTimeout(500);
    }

    // Verify path card is visible
    const foundationsCard = await page.locator('text=Foundations').first();
    await expect(foundationsCard).toBeVisible();
    console.log('✓ Foundations path card visible');

    // Click Start Path
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(3000);

    // Verify we're in the path detail view
    await expect(page.locator('text=Step 1 of 9')).toBeVisible();
    console.log('✓ Entered path detail view');

    // Verify step info is loaded - Step 1 is now a prediction quiz
    const stepTitle = await page.locator('h3').filter({ hasText: /Does AND Need|Predict the Outcome/ }).first();
    await expect(stepTitle).toBeVisible();
    console.log('✓ Step title displayed for step 1 (prediction quiz)');

    // Verify problem is fully loaded - check for specific elements
    // Step 1 is now a prediction_quiz, so it shows quiz UI instead of training controls
    const checks = [
      { name: 'Step progress bar', locator: 'text=Step 1' },
      { name: 'Learning objectives', locator: 'text=LEARNING OBJECTIVES' },
      { name: 'Network visualization', locator: 'text=Network Architecture' },
    ];

    for (const check of checks) {
      const element = page.locator(check.locator).first();
      await expect(element).toBeVisible({ timeout: 5000 });
      console.log(`✓ ${check.name} visible`);
    }

    // Step 1 is a prediction quiz - check for quiz or training content
    const hasQuiz = await page.locator('text=/Predict the Outcome|What will happen|Does the AND/i').first().isVisible({ timeout: 3000 }).catch(() => false);
    const hasTraining = await page.locator('text="Training Controls"').isVisible({ timeout: 1000 }).catch(() => false);
    expect(hasQuiz || hasTraining).toBeTruthy();
    console.log(`✓ Step 1 shows ${hasQuiz ? 'prediction quiz' : 'training'} UI`);

    // Take screenshot
    await page.screenshot({ path: 'final-path-verification.png', fullPage: true });
    console.log('✓ Screenshot saved: final-path-verification.png');

    console.log('\n=== SUCCESS: Foundations path loads completely! ===\n');
  });

  test('verify API returns all path problems correctly', async ({ page }) => {
    console.log('\n=== Verifying All Paths via API ===\n');

    const paths = [
      { id: 'foundations', expectedSteps: 9 },
      { id: 'training-mastery', expectedSteps: 9 },
      { id: 'boundaries-and-classes', expectedSteps: 9 },
      { id: 'convolutional-vision', expectedSteps: 5 },
      { id: 'advanced-challenges', expectedSteps: 7 },
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

    console.log('\n✓ All 5 paths have valid steps with correct problem IDs');
    console.log('✓ Total steps across all paths: 39');
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
    // architecture may be an object with layer_sizes or a flat array
    const layerSizes = networkData.architecture?.layer_sizes ?? networkData.architecture;
    expect(layerSizes).toEqual([2, 1]);
    console.log('✓ Network architecture updated to [2, 1] for AND gate');
  });
});
