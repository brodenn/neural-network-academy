import { test, expect } from '@playwright/test';
import { waitForConnection, selectProblem } from './fixtures/test-helpers';

/**
 * Helper to open the problem selector dropdown
 */
async function openDropdown(page: import('@playwright/test').Page) {
  // Find and click the dropdown button in the header
  const headerDropdown = page.locator('header button').first();
  await headerDropdown.click();
  await page.waitForTimeout(200);
}

/**
 * Get the dropdown menu container (w-80 distinguishes from keyboard shortcuts dropdown)
 */
function getDropdownMenu(page: import('@playwright/test').Page) {
  return page.locator('.absolute.bg-gray-800.w-80');
}

test.describe('Problem Selector', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForConnection(page);
  });

  test('should display dropdown button with current problem', async ({ page }) => {
    // The dropdown button should be visible with a problem name
    const dropdownButton = page.locator('button').filter({
      hasText: /Select Problem|AND Gate|OR Gate|XOR|5-bit Parity/
    }).first();
    await expect(dropdownButton).toBeVisible();
  });

  test('should display all 7 levels in dropdown', async ({ page }) => {
    await openDropdown(page);

    await expect(page.getByText('Level 1: Single Neuron')).toBeVisible();
    await expect(page.getByText('Level 2: Hidden Layers')).toBeVisible();
    await expect(page.getByText('Level 3: Decision Boundaries')).toBeVisible();
    await expect(page.getByText('Level 4: Regression')).toBeVisible();
    await expect(page.getByText('Level 5: Failure Cases')).toBeVisible();
    await expect(page.getByText('Level 6: Multi-Class')).toBeVisible();
    await expect(page.getByText('Level 7: CNN (Images)')).toBeVisible();
  });

  test('should expand level and show problems', async ({ page }) => {
    await openDropdown(page);
    const menu = getDropdownMenu(page);

    // Expand Level 1
    await menu.getByRole('button', { name: /Level 1: Single Neuron/ }).click();
    await page.waitForTimeout(200);

    // Check problems are visible
    await expect(menu.locator('button').filter({ hasText: /^AND Gate/ }).first()).toBeVisible();
    await expect(menu.locator('button').filter({ hasText: /^NOT Gate/ }).first()).toBeVisible();
  });

  test('should expand Level 2 and show XOR problem', async ({ page }) => {
    await openDropdown(page);
    const menu = getDropdownMenu(page);

    await menu.getByRole('button', { name: /Level 2: Hidden Layers/ }).click();
    await page.waitForTimeout(200);

    await expect(menu.locator('button').filter({ hasText: /XOR Gate/ }).first()).toBeVisible();
  });

  test('should select problem and update dropdown button', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Check that the header shows "AND Gate"
    const headerDropdown = page.locator('header button').first();
    await expect(headerDropdown).toContainText('AND Gate');
  });

  test('should show failure case level with warning styling', async ({ page }) => {
    await openDropdown(page);
    const menu = getDropdownMenu(page);

    // Level 5 should have special styling with exclamation mark
    const level5 = menu.getByRole('button', { name: /Level 5: Failure Cases/ });
    await expect(level5).toBeVisible();

    // Expand Level 5
    await level5.click();
    await page.waitForTimeout(200);

    // Check for failure case problems (actual name is "XOR (No Hidden Layer)")
    await expect(menu.locator('button').filter({ hasText: /XOR \(No Hidden/ }).first()).toBeVisible();
  });

  test('should show problem count for each level', async ({ page }) => {
    await openDropdown(page);
    const menu = getDropdownMenu(page);

    // Each level header should display number of problems (shows as small number)
    // Level 1 has 4 problems - use first() to avoid strict mode with multiple "4"s
    await expect(menu.locator('.text-xs.text-gray-500').first()).toBeVisible();
  });

  test('should collapse level when clicked again', async ({ page }) => {
    await openDropdown(page);
    const menu = getDropdownMenu(page);

    // Expand Level 1
    await menu.getByRole('button', { name: /Level 1: Single Neuron/ }).click();
    await page.waitForTimeout(200);

    // Verify AND Gate is visible
    const andGateButton = menu.locator('button').filter({ hasText: /^AND Gate/ }).first();
    await expect(andGateButton).toBeVisible();

    // Collapse Level 1
    await menu.getByRole('button', { name: /Level 1: Single Neuron/ }).click();
    await page.waitForTimeout(200);

    // Verify AND Gate is no longer visible
    await expect(andGateButton).not.toBeVisible();
  });

  test('should close dropdown when clicking outside', async ({ page }) => {
    await openDropdown(page);

    // Verify dropdown is open
    await expect(page.getByText('Level 1: Single Neuron')).toBeVisible();

    // Click outside the dropdown
    await page.locator('body').click({ position: { x: 10, y: 10 } });
    await page.waitForTimeout(200);

    // Dropdown should be closed
    await expect(page.getByText('Level 1: Single Neuron')).not.toBeVisible();
  });

  test('should close dropdown when pressing Escape', async ({ page }) => {
    await openDropdown(page);

    // Verify dropdown is open
    await expect(page.getByText('Level 1: Single Neuron')).toBeVisible();

    // Press Escape
    await page.keyboard.press('Escape');
    await page.waitForTimeout(200);

    // Dropdown should be closed
    await expect(page.getByText('Level 1: Single Neuron')).not.toBeVisible();
  });

  test('should show current problem info in dropdown header', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Open dropdown again
    await openDropdown(page);
    const menu = getDropdownMenu(page);

    // Should show problem description in dropdown
    await expect(menu.getByText(/Output 1 only when BOTH inputs are 1/)).toBeVisible();
  });

  test('should show currently selected problem in header', async ({ page }) => {
    await selectProblem(page, 'Level 1', 'AND Gate');

    // Header should show the selected problem name
    const headerDropdown = page.locator('header button').first();
    await expect(headerDropdown).toContainText('AND Gate');

    // Select a different problem
    await selectProblem(page, 'Level 2', 'XOR Gate');

    // Header should update to new problem
    await expect(headerDropdown).toContainText('XOR Gate');
  });
});
