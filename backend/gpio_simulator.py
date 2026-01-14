"""
GPIO Simulator for Raspberry Pi

Simulates button inputs and LED output for development without hardware.
Can be swapped with real GPIO implementation on Raspberry Pi 5.
"""

from typing import Callable
from dataclasses import dataclass, field
import threading
import time


@dataclass
class GPIOSimulator:
    """
    Simulates Raspberry Pi GPIO for buttons and LED.

    In simulation mode, button states are controlled via API.
    On real hardware, this would read actual GPIO pins.

    Attributes:
        button_pins: List of GPIO pins for buttons (e.g., [17, 27, 22, 23, 24])
        led_pin: GPIO pin for LED output (e.g., 18)
    """

    button_pins: list[int] = field(default_factory=lambda: [17, 27, 22, 23, 24])
    led_pin: int = 18

    # Internal state
    _button_states: list[int] = field(default_factory=list, init=False)
    _led_state: bool = field(default=False, init=False)
    _on_change_callback: Callable[[list[int]], None] | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        """Initialize button states to all zeros."""
        self._button_states = [0] * len(self.button_pins)
        self._led_state = False

    def read_buttons(self) -> list[int]:
        """
        Read current state of all buttons.

        Returns:
            List of button states (0 or 1), e.g., [0, 1, 0, 1, 0]
        """
        with self._lock:
            return self._button_states.copy()

    def read_button(self, index: int) -> int:
        """
        Read state of a single button.

        Args:
            index: Button index (0 to num_buttons-1)

        Returns:
            Button state (0 or 1)
        """
        with self._lock:
            if 0 <= index < len(self._button_states):
                return self._button_states[index]
            return 0

    def set_button(self, index: int, state: bool) -> None:
        """
        Set state of a single button (for simulation).

        Args:
            index: Button index (0 to num_buttons-1)
            state: True for pressed, False for released
        """
        with self._lock:
            if 0 <= index < len(self._button_states):
                old_states = self._button_states.copy()
                self._button_states[index] = 1 if state else 0
                new_states = self._button_states.copy()

        # Trigger callback if state changed
        if old_states != new_states and self._on_change_callback:
            self._on_change_callback(new_states)

    def set_buttons(self, states: list[int]) -> None:
        """
        Set all button states at once (for simulation).

        Args:
            states: List of button states (0 or 1)
        """
        with self._lock:
            old_states = self._button_states.copy()
            for i, state in enumerate(states):
                if i < len(self._button_states):
                    self._button_states[i] = 1 if state else 0
            new_states = self._button_states.copy()

        # Trigger callback if state changed
        if old_states != new_states and self._on_change_callback:
            self._on_change_callback(new_states)

    def toggle_button(self, index: int) -> int:
        """
        Toggle a button state.

        Args:
            index: Button index

        Returns:
            New button state (0 or 1)
        """
        with self._lock:
            if 0 <= index < len(self._button_states):
                old_states = self._button_states.copy()
                self._button_states[index] = 1 - self._button_states[index]
                new_state = self._button_states[index]
                new_states = self._button_states.copy()
            else:
                return 0

        # Trigger callback if state changed
        if old_states != new_states and self._on_change_callback:
            self._on_change_callback(new_states)

        return new_state

    def set_led(self, state: bool) -> None:
        """
        Set LED state.

        Args:
            state: True for ON, False for OFF
        """
        with self._lock:
            self._led_state = state

    def get_led_state(self) -> bool:
        """
        Get current LED state.

        Returns:
            True if LED is ON, False if OFF
        """
        with self._lock:
            return self._led_state

    def on_button_change(self, callback: Callable[[list[int]], None]) -> None:
        """
        Register callback for button state changes.

        Args:
            callback: Function to call with new button states
        """
        self._on_change_callback = callback

    def get_state(self) -> dict:
        """
        Get complete GPIO state.

        Returns:
            Dictionary with buttons and LED state
        """
        with self._lock:
            return {
                "buttons": self._button_states.copy(),
                "led": self._led_state,
                "button_pins": self.button_pins,
                "led_pin": self.led_pin
            }

    def reset(self) -> None:
        """Reset all buttons to 0 and LED to off."""
        with self._lock:
            old_states = self._button_states.copy()
            self._button_states = [0] * len(self.button_pins)
            self._led_state = False
            new_states = self._button_states.copy()

        if old_states != new_states and self._on_change_callback:
            self._on_change_callback(new_states)


class GPIOHardware:
    """
    Real Raspberry Pi GPIO implementation.

    This class would be used on actual Raspberry Pi 5 hardware.
    Uses lgpio library for GPIO access.

    To use on Raspberry Pi:
        1. Install: pip install lgpio
        2. Replace GPIOSimulator with GPIOHardware in app.py
    """

    def __init__(
        self,
        button_pins: list[int] = [17, 27, 22, 23, 24],
        led_pin: int = 18
    ):
        self.button_pins = button_pins
        self.led_pin = led_pin
        self._led_state = False
        self._on_change_callback = None
        self._previous_states = [0] * len(button_pins)

        # Try to import lgpio (only available on Raspberry Pi)
        try:
            import lgpio
            self._lgpio = lgpio
            self._chip = lgpio.gpiochip_open(0)

            # Setup button pins as inputs with pull-down
            for pin in button_pins:
                lgpio.gpio_claim_input(self._chip, pin, lgpio.SET_PULL_DOWN)

            # Setup LED pin as output
            lgpio.gpio_claim_output(self._chip, led_pin)
            lgpio.gpio_write(self._chip, led_pin, 0)

            self._hardware_available = True
            print(f"GPIO Hardware initialized: buttons={button_pins}, led={led_pin}")

        except ImportError:
            print("Warning: lgpio not available, GPIO hardware disabled")
            self._hardware_available = False
        except Exception as e:
            print(f"Warning: GPIO initialization failed: {e}")
            self._hardware_available = False

    def read_buttons(self) -> list[int]:
        """Read current state of all buttons from hardware."""
        if not self._hardware_available:
            return [0] * len(self.button_pins)

        states = []
        for pin in self.button_pins:
            state = self._lgpio.gpio_read(self._chip, pin)
            states.append(state)
        return states

    def read_button(self, index: int) -> int:
        """Read state of a single button."""
        if not self._hardware_available:
            return 0
        if 0 <= index < len(self.button_pins):
            return self._lgpio.gpio_read(self._chip, self.button_pins[index])
        return 0

    def set_led(self, state: bool) -> None:
        """Set LED state on hardware."""
        self._led_state = state
        if self._hardware_available:
            self._lgpio.gpio_write(self._chip, self.led_pin, 1 if state else 0)

    def get_led_state(self) -> bool:
        """Get current LED state."""
        return self._led_state

    def on_button_change(self, callback: Callable[[list[int]], None]) -> None:
        """Register callback for button state changes."""
        self._on_change_callback = callback

    def poll_buttons(self) -> bool:
        """
        Poll buttons and trigger callback if changed.

        Returns:
            True if state changed, False otherwise
        """
        current_states = self.read_buttons()
        if current_states != self._previous_states:
            self._previous_states = current_states.copy()
            if self._on_change_callback:
                self._on_change_callback(current_states)
            return True
        return False

    def get_state(self) -> dict:
        """Get complete GPIO state."""
        return {
            "buttons": self.read_buttons(),
            "led": self._led_state,
            "button_pins": self.button_pins,
            "led_pin": self.led_pin,
            "hardware_available": self._hardware_available
        }

    def reset(self) -> None:
        """Reset LED to off."""
        self.set_led(False)

    def cleanup(self) -> None:
        """Cleanup GPIO resources."""
        if self._hardware_available:
            self._lgpio.gpiochip_close(self._chip)


# -----------------------------------------------------------------------------
# Demo / Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("GPIO Simulator Demo")
    print("-" * 40)

    # Create simulator
    gpio = GPIOSimulator()

    # Register callback
    def on_change(states):
        print(f"Button change: {states}")

    gpio.on_button_change(on_change)

    # Test button operations
    print(f"Initial state: {gpio.get_state()}")

    gpio.set_button(0, True)
    print(f"After button 0 pressed: {gpio.read_buttons()}")

    gpio.toggle_button(2)
    print(f"After button 2 toggled: {gpio.read_buttons()}")

    gpio.set_led(True)
    print(f"LED state: {gpio.get_led_state()}")

    gpio.reset()
    print(f"After reset: {gpio.get_state()}")
