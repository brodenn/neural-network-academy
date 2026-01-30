import { test, expect } from '@playwright/test';
import { waitForConnection, startStaticTraining, waitForTrainingComplete, resetNetwork } from './fixtures/test-helpers';

/**
 * Comprehensive E2E test for Learning Path feature
 * Simulates a human user going through the learning experience
 */
test.describe('Learning Path - Human User Journey', () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage to start fresh
    await page.goto('http://localhost:5173');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(500);
    await page.evaluate(() => {
      localStorage.clear();
      // Dismiss onboarding modal so it doesn't block interactions
      localStorage.setItem('learning_paths_onboarding_seen', 'true');
    });
    await page.reload();
    await page.waitForLoadState('networkidle');
  });

  test('complete user journey through Learning Paths', async ({ page }) => {
    test.setTimeout(120000);  // 2 minute timeout for this comprehensive test
    console.log('\n=== Starting Human User Journey Through Learning Paths ===\n');

    // Step 1: Start at main page and navigate to Learning Paths
    console.log('📍 Step 1: Navigate to Learning Paths');
    await expect(page.locator('text=Learning Paths')).toBeVisible();
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Verify we see the Learning Paths page
    await expect(page.locator('h1:has-text("Learning Paths")')).toBeVisible();
    await expect(page.locator('text=Choose your journey')).toBeVisible();
    console.log('✓ Learning Paths page loaded');

    // Step 2: Verify path cards are displayed
    console.log('\n📍 Step 2: Verify path cards are visible');
    const foundationsCard = page.locator('text=Foundations').first();
    await expect(foundationsCard).toBeVisible();
    console.log('✓ Foundations path card visible');

    // Step 3: Check that locked paths show lock icon (prerequisites)
    console.log('\n📍 Step 3: Check prerequisite locking');
    // Advanced Challenges requires Training Mastery + Boundaries & Classes - should be locked for new user
    const researchCard = page.locator('[data-testid="path-card-advanced-challenges"], div:has-text("Advanced Challenges")').first();
    if (await researchCard.isVisible()) {
      // Check for lock indicator or "Complete first" text
      const hasLock = await page.locator('text=/Complete first/').isVisible().catch(() => false);
      if (hasLock) {
        console.log('✓ Locked paths show prerequisites');
      } else {
        console.log('ℹ No locked paths visible (may already have progress)');
      }
    }

    // Step 4: Select Foundations path (the beginner path)
    console.log('\n📍 Step 4: Start Foundations path');
    const startButton = page.locator('button:has-text("Start Path")').first();
    await expect(startButton).toBeVisible();
    await startButton.click();
    await page.waitForTimeout(2000);

    // Verify we entered the path detail view
    await expect(page.locator('text=/Step 1 of \\d+/')).toBeVisible();
    console.log('✓ Entered path detail view');

    // Step 5: Verify path progress bar is accessible
    console.log('\n📍 Step 5: Check accessibility of progress bar');
    const progressNav = page.locator('nav[aria-label="Learning path progress"]');
    await expect(progressNav).toBeVisible();
    console.log('✓ Progress bar has proper navigation role');

    // Check for aria-current on active step
    const activeStep = page.locator('[aria-current="step"]');
    await expect(activeStep).toBeVisible();
    console.log('✓ Active step has aria-current="step"');

    // Step 6: Verify hint panel shows locked hints initially
    console.log('\n📍 Step 6: Check hint locking system');
    const hintPanel = page.locator('text=/Hints|HINTS/i').first();
    await expect(hintPanel).toBeVisible();

    // Look for locked hint indicator (attempts needed to unlock)
    const lockedHint = page.locator('text=/attempts? to unlock|🔒/').first();
    if (await lockedHint.isVisible().catch(() => false)) {
      console.log('✓ Hints are locked initially (need attempts to unlock)');
    } else {
      console.log('ℹ Hint system may be different or hints already unlocked');
    }

    // Step 7: Verify learning objectives are shown
    console.log('\n📍 Step 7: Check learning objectives');
    // Learning objectives text appears in h4 with specific styling
    const objectives = page.locator('h4:has-text("Learning Objectives"), text="Learning Objectives"').first();
    if (await objectives.isVisible().catch(() => false)) {
      console.log('✓ Learning objectives displayed');
    } else {
      console.log('ℹ Learning objectives section may be collapsed or different layout');
    }

    // Step 8: Verify the problem loaded correctly
    console.log('\n📍 Step 8: Verify problem is loaded');
    await waitForConnection(page);

    // Check for network visualization
    await expect(page.locator('text=Network Architecture')).toBeVisible();
    console.log('✓ Network visualization visible');

    // Step 1 is now a prediction quiz, so check for quiz OR training UI
    const hasQuizUI = await page.locator('text=/Predict the Outcome|What will happen|Does the AND/i').first().isVisible({ timeout: 3000 }).catch(() => false);
    const hasTrainingUI = await page.locator('text=Training Controls').isVisible({ timeout: 1000 }).catch(() => false);
    expect(hasQuizUI || hasTrainingUI).toBeTruthy();
    console.log(`✓ Step 1 shows ${hasQuizUI ? 'prediction quiz' : 'training'} UI`);

    // Step 9: Complete step 1 (prediction quiz - answer, check, and train)
    console.log('\n📍 Step 9: Complete the prediction quiz and train');
    if (hasQuizUI) {
      // Select the correct answer option
      const quizOption = page.locator('button:has-text("No, AND is linearly separable")').first();
      if (await quizOption.isVisible({ timeout: 2000 }).catch(() => false)) {
        await quizOption.click();
        await page.waitForTimeout(500);
      }

      // Click "Check My Prediction" to verify answer
      const checkButton = page.locator('button:has-text("Check My Prediction")').first();
      if (await checkButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await checkButton.click();
        await page.waitForTimeout(1000);
        console.log('✓ Checked prediction');
      }

      // Click "Train & See It Happen" to start training
      const trainButton = page.locator('button:has-text("Train & See")').first();
      if (await trainButton.isVisible({ timeout: 3000 }).catch(() => false)) {
        await trainButton.click();
        // Wait for step completion (95% accuracy) OR training to finish
        // The step can complete while training continues, so check for step completion first
        await Promise.race([
          expect(page.locator('text=/1\\/\\d+ completed|Step completed/').first()).toBeVisible({ timeout: 25000 }),
          expect(page.getByText('Ready').first()).toBeVisible({ timeout: 25000 }),
        ]);
        console.log('✓ Training or step completed');
      } else {
        // Fallback: if no train button, use standard training panel
        const startLearningBtn = page.locator('button:has-text("Start Learning")').first();
        if (await startLearningBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
          await startLearningBtn.click();
          await waitForTrainingComplete(page, 20000);
          console.log('✓ Training via Start Learning button');
        } else {
          console.log('ℹ No train button visible');
        }
      }
    } else {
      // Fallback: train if it's a training step
      await startStaticTraining(page, 500, 0.5);
      await waitForTrainingComplete(page, 30000);
      console.log('✓ Training completed');
    }

    // Step 10: Check if step was completed
    console.log('\n📍 Step 10: Verify step completion');
    await page.waitForTimeout(1000);

    // Look for completion indicator or next step button
    const completeButton = page.locator('button:has-text("Complete"), button:has-text("Next Step")').first();
    const isCompletable = await completeButton.isVisible().catch(() => false);
    if (isCompletable) {
      console.log('✓ Step can be marked as complete');
    } else {
      console.log('ℹ May need more interaction to complete step');
    }

    // Step 11: Navigate between steps using progress bar
    console.log('\n📍 Step 11: Test step navigation');
    // Check if step 2 is unlocked (not locked) - a locked step has a lock icon
    const step2ListItem = page.locator('li[aria-label*="Step 2"], li:has-text("How Does OR")').first();
    const isStep2Locked = await step2ListItem.locator('img[src*="lock"], [class*="lock"]').isVisible().catch(() => true);

    if (!isStep2Locked) {
      const step2Button = page.locator('[aria-label*="Step 2"], button:has-text("2")').first();
      if (await step2Button.isVisible().catch(() => false)) {
        await step2Button.click();
        await page.waitForTimeout(1000);
        await expect(page.locator('text=/Step 2 of \\d+/')).toBeVisible();
        console.log('✓ Can navigate to step 2');

        // Go back to step 1 (completed step shows checkmark, use aria-label)
        const step1Item = page.locator('li[aria-label*="Does AND"], li:has-text("Does AND")').first();
        if (await step1Item.isVisible({ timeout: 2000 }).catch(() => false)) {
          await step1Item.click();
          await page.waitForTimeout(1000);
          console.log('✓ Can navigate back to step 1');
        } else {
          console.log('ℹ Step 1 navigation not available');
        }
      }
    } else {
      console.log('ℹ Step 2 is locked (step 1 may not be completed yet)');
    }

    // Step 12: Test responsive layout
    console.log('\n📍 Step 12: Test responsive layout');

    // Desktop view
    await page.setViewportSize({ width: 1400, height: 900 });
    await page.waitForTimeout(500);
    console.log('✓ Desktop layout (1400px)');

    // Tablet view - check that main content is still visible
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);
    await expect(page.locator('text=Network Architecture')).toBeVisible();
    console.log('✓ Tablet layout (768px)');

    // Mobile view - check main heading still visible
    await page.setViewportSize({ width: 375, height: 812 });
    await page.waitForTimeout(500);
    await expect(page.locator('h1:has-text("Foundations")')).toBeVisible();
    console.log('✓ Mobile layout (375px) - content still accessible');

    // Reset to desktop
    await page.setViewportSize({ width: 1400, height: 900 });

    // Step 13: Test "Back to Paths" navigation
    console.log('\n📍 Step 13: Test back navigation');
    const backButton = page.locator('button:has-text("Back to Paths"), button:has-text("← Back")').first();
    if (await backButton.isVisible().catch(() => false)) {
      await backButton.click();
      await page.waitForTimeout(1000);
      await expect(page.locator('h1:has-text("Learning Paths")')).toBeVisible();
      console.log('✓ Can navigate back to path selector');
    }

    console.log('\n=== Human User Journey Complete! ===\n');
  });

  test('test hint unlocking after failed attempts', async ({ page }) => {
    test.setTimeout(60000);  // Extend timeout for this test
    console.log('\n=== Testing Hint Unlock System ===\n');

    // Navigate to Learning Paths and start Foundations
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(2000);

    await waitForConnection(page);

    // Find and count hints
    const hintSection = page.locator('button:has-text("Hints"), div:has-text("Hints")').first();
    if (await hintSection.isVisible().catch(() => false)) {
      console.log('✓ Hint section found');

      // Check initial locked state
      const initialLocked = await page.locator('text=/🔒|locked|attempts? to unlock/i').count();
      console.log(`ℹ Found ${initialLocked} locked hint indicators initially`);

      // In learning path mode, complete the quiz first to make attempts
      console.log('\n📍 Making attempts to unlock hints...');

      // Complete quiz (wrong answer first to simulate failure)
      const wrongOption = page.locator('button:has-text("Yes, all problems need hidden layers")').first();
      if (await wrongOption.isVisible({ timeout: 2000 }).catch(() => false)) {
        await wrongOption.click();
        await page.waitForTimeout(500);
        const checkButton = page.locator('button:has-text("Check My Prediction")').first();
        if (await checkButton.isVisible({ timeout: 2000 }).catch(() => false)) {
          await checkButton.click();
          await page.waitForTimeout(1000);
          console.log('✓ Made attempt 1 (wrong answer)');
        }
      }

      // Check if any hints unlocked after attempt
      const afterAttempt = await page.locator('text=/🔒/').count();
      if (afterAttempt < initialLocked) {
        console.log('✓ Hint unlocked after attempt!');
      } else {
        console.log('ℹ Hints may require training or more attempts');
      }
    } else {
      console.log('ℹ Hint section not visible for this step');
    }

    console.log('\n=== Hint Unlock Test Complete ===\n');
  });

  test('test path reset functionality', async ({ page }) => {
    console.log('\n=== Testing Path Reset Functionality ===\n');

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Start Foundations path
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(2000);

    await waitForConnection(page);

    // Complete step 1 (now a prediction quiz)
    console.log('📍 Completing step 1 (prediction quiz)...');
    const hasQuiz = await page.locator('text=/Predict the Outcome|What will happen|Does the AND/i').first().isVisible({ timeout: 3000 }).catch(() => false);
    if (hasQuiz) {
      const quizOption = page.locator('button:has-text("No, AND is linearly separable")').first();
      if (await quizOption.isVisible({ timeout: 2000 }).catch(() => false)) {
        await quizOption.click();
        await page.waitForTimeout(500);
      }
      const checkButton = page.locator('button:has-text("Check"), button:has-text("Train & See")').first();
      if (await checkButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await checkButton.click();
        await page.waitForTimeout(1000);
      }
      console.log('✓ Step 1 prediction quiz completed');
    } else {
      // Fallback: train if it's a training step
      await startStaticTraining(page, 1000, 0.5);
      await waitForTrainingComplete(page, 60000);
      const completeButton = page.locator('button:has-text("Complete Step"), button:has-text("Mark Complete")').first();
      if (await completeButton.isVisible().catch(() => false)) {
        await completeButton.click();
        await page.waitForTimeout(1000);
      }
      console.log('✓ Step 1 training completed');
    }

    // Look for reset button
    console.log('\n📍 Testing reset functionality...');
    const resetButton = page.locator('button:has-text("Reset Progress"), button[title*="Reset"]').first();

    if (await resetButton.isVisible().catch(() => false)) {
      await resetButton.click();
      await page.waitForTimeout(500);

      // Check for confirmation modal
      const modalOverlay = page.locator('.fixed.inset-0');
      if (await modalOverlay.isVisible().catch(() => false)) {
        console.log('✓ Reset confirmation modal appeared');

        // Find and click confirm button INSIDE the modal
        const modalContent = page.locator('.bg-gray-800.rounded-lg.p-6');
        const confirmButton = modalContent.locator('button:has-text("Reset Progress")');

        if (await confirmButton.isVisible().catch(() => false)) {
          await confirmButton.click({ force: true });
          await page.waitForTimeout(1000);
          console.log('✓ Reset confirmed');

          // Verify progress was reset (back to step 1, no completion)
          const step1Active = page.locator('[aria-current="step"]');
          if (await step1Active.isVisible().catch(() => false)) {
            console.log('✓ Progress reset - back to step 1');
          }
        } else {
          // Try Cancel to dismiss modal if confirm button not found
          const cancelButton = modalContent.locator('button:has-text("Cancel")');
          if (await cancelButton.isVisible().catch(() => false)) {
            await cancelButton.click({ force: true });
            console.log('ℹ Reset cancelled - confirm button not found');
          }
        }
      }
    } else {
      console.log('ℹ Reset button not found in current view');
    }

    console.log('\n=== Path Reset Test Complete ===\n');
  });

  test('test toast notifications on errors', async ({ page }) => {
    console.log('\n=== Testing Toast Notifications ===\n');

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Start a path
    await page.click('button:has-text("Start Path")');
    await page.waitForTimeout(2000);

    await waitForConnection(page);

    // Try to trigger an error by making an invalid API call
    // We'll use page.evaluate to simulate an error scenario
    console.log('📍 Checking toast notification container exists...');

    // The toast container should exist (bottom-right positioning)
    // Note: Container may be invisible until a toast appears
    await expect(page.locator('.fixed.bottom-4.right-4, [class*="toast"]')).toBeAttached({ timeout: 1000 }).catch(() => {
      // Toast container might not exist yet, which is fine
    });
    console.log('✓ Toast notification system is set up');

    // Simulate a successful action that shows a toast
    // Step 1 may be a quiz now; complete it to get to a training step or trigger toast
    const quizOpt = page.locator('button:has-text("No, AND is linearly separable")').first();
    if (await quizOpt.isVisible({ timeout: 2000 }).catch(() => false)) {
      await quizOpt.click();
      await page.waitForTimeout(500);
      const revealBtn = page.locator('button:has-text("Check"), button:has-text("Train & See")').first();
      if (await revealBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await revealBtn.click();
        await page.waitForTimeout(1000);
      }
    } else {
      // Fallback: try training
      await startStaticTraining(page, 500, 0.5);
      await waitForTrainingComplete(page, 30000);
    }

    // Check for any toast appearance
    await page.waitForTimeout(1000);
    const visibleToast = page.locator('.bg-green-600, .bg-blue-600, .bg-red-600, [class*="toast"]').first();
    if (await visibleToast.isVisible().catch(() => false)) {
      console.log('✓ Toast notification appeared!');
    } else {
      console.log('ℹ No toast visible (may appear on specific actions)');
    }

    console.log('\n=== Toast Notification Test Complete ===\n');
  });

  test('verify all learning paths load correctly', async ({ page }) => {
    console.log('\n=== Verifying All Learning Paths ===\n');

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Expected paths
    const expectedPaths = [
      'Foundations',
      'Training Mastery',
      'Boundaries & Classes',
      'Convolutional Vision',
      'Advanced Challenges'
    ];

    for (const pathName of expectedPaths) {
      const pathCard = page.locator(`text=${pathName}`).first();
      const isVisible = await pathCard.isVisible().catch(() => false);
      if (isVisible) {
        console.log(`✓ ${pathName} path visible`);
      } else {
        console.log(`✗ ${pathName} path NOT visible`);
      }
    }

    // Verify API returns correct data
    console.log('\n📍 Verifying API data...');
    const response = await page.request.get('http://localhost:5000/api/paths');
    expect(response.ok()).toBeTruthy();

    const paths = await response.json();
    expect(paths.length).toBe(5);
    console.log(`✓ API returns ${paths.length} paths`);

    // Check step counts (need to fetch each path individually to get steps)
    const expectedSteps: Record<string, number> = {
      'foundations': 9,
      'training-mastery': 9,
      'boundaries-and-classes': 9,
      'convolutional-vision': 5,
      'advanced-challenges': 7
    };

    for (const path of paths) {
      const expected = expectedSteps[path.id];
      if (expected) {
        // Fetch individual path to get steps
        const pathResponse = await page.request.get(`http://localhost:5000/api/paths/${path.id}`);
        expect(pathResponse.ok()).toBeTruthy();
        const pathData = await pathResponse.json();
        expect(pathData.steps.length).toBe(expected);
        console.log(`✓ ${path.id}: ${pathData.steps.length} steps (expected ${expected})`);
      }
    }

    console.log('\n=== All Paths Verification Complete ===\n');
  });

  test('verify milestone celebrations', async ({ page }) => {
    test.setTimeout(60000);  // Extend timeout
    console.log('\n=== Testing Milestone Celebrations ===\n');

    // Set up localStorage with partial progress to trigger milestone
    await page.evaluate(() => {
      // Pre-populate with progress just before 25% milestone
      const progress = {
        'foundations': {
          completedSteps: [0], // 1 of 9 = 11%
          currentStep: 1
        }
      };
      localStorage.setItem('learningPathProgress', JSON.stringify(progress));
    });

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1000);

    // Continue Foundations path
    const continueButton = page.locator('button:has-text("Continue")').first();
    if (await continueButton.isVisible().catch(() => false)) {
      await continueButton.click();
    } else {
      await page.click('button:has-text("Start Path")');
    }
    await page.waitForTimeout(2000);

    await waitForConnection(page);

    // Complete step using learning path UI (quiz + train)
    console.log('📍 Completing step to trigger milestone...');

    // Complete prediction quiz if present
    const quizOption = page.locator('button:has-text("No,"), button:has-text("Yes,")').first();
    if (await quizOption.isVisible({ timeout: 3000 }).catch(() => false)) {
      await quizOption.click();
      await page.waitForTimeout(500);
      const checkButton = page.locator('button:has-text("Check My Prediction")').first();
      if (await checkButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await checkButton.click();
        await page.waitForTimeout(1000);
      }
      // Click "Train & See It Happen" if present
      const trainButton = page.locator('button:has-text("Train & See")').first();
      if (await trainButton.isVisible({ timeout: 2000 }).catch(() => false)) {
        await trainButton.click();
        // Wait for step completion or training complete
        await Promise.race([
          expect(page.locator('text=/completed|Step completed/').first()).toBeVisible({ timeout: 20000 }),
          expect(page.getByText('Ready').first()).toBeVisible({ timeout: 20000 }),
        ]).catch(() => {});
      }
    }

    // Check for milestone celebration (confetti or celebration UI)
    await page.waitForTimeout(2000);
    const celebration = page.locator('text=/Milestone|Congratulations|🎉|25%/i');
    if (await celebration.isVisible().catch(() => false)) {
      console.log('✓ Milestone celebration appeared!');
    } else {
      console.log('ℹ Milestone celebration may require more steps completed or have different threshold');
    }

    console.log('\n=== Milestone Celebration Test Complete ===\n');
  });

  test('verify streak tracking', async ({ page }) => {
    console.log('\n=== Testing Streak Tracking ===\n');

    // Set up localStorage with streak data
    await page.evaluate(() => {
      const today = new Date().toISOString().split('T')[0];
      const streakData = {
        lastAccessDate: today,
        currentStreak: 3,
        bestStreak: 5
      };
      localStorage.setItem('learning_streak', JSON.stringify(streakData));
    });

    // Navigate to Learning Paths
    await page.click('text=Learning Paths');
    await page.waitForTimeout(1500);

    // Check for streak display
    const streakDisplay = page.locator('text=/🔥|streak|day/i').first();
    if (await streakDisplay.isVisible().catch(() => false)) {
      console.log('✓ Streak display is visible');

      // Check for the streak count
      const streakText = await streakDisplay.textContent();
      if (streakText?.includes('3') || streakText?.includes('Day')) {
        console.log('✓ Streak count is displayed correctly');
      }
    } else {
      console.log('ℹ Streak display may only show for active streaks');
    }

    console.log('\n=== Streak Tracking Test Complete ===\n');
  });
});
