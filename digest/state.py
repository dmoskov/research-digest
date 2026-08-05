"""
State management for digest generation.

Tracks the last_8am_cutoff to ensure we only fetch new content since the last run.
"""

import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional


class DigestState:
    """Manages persistent state for digest generation."""

    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize state manager.

        Args:
            state_file: Path to state file (defaults to .digest_state.json in script dir)
        """
        if state_file is None:
            # Default to script directory
            script_dir = Path(__file__).parent
            self.state_file = script_dir / ".digest_state.json"
        else:
            self.state_file = Path(state_file)

        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load state file: {e}")
                return {}
        return {}

    def _save_state(self):
        """Save state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save state file: {e}")

    def get_last_8am_cutoff(self) -> Optional[datetime]:
        """
        Get the last 8am cutoff timestamp.

        Returns:
            datetime of last 8am cutoff, or None if not set
        """
        cutoff_str = self.state.get("last_8am_cutoff")
        if cutoff_str:
            try:
                return datetime.fromisoformat(cutoff_str)
            except ValueError:
                return None
        return None

    def update_last_8am_cutoff(self, cutoff: Optional[datetime] = None):
        """
        Update the last 8am cutoff timestamp.

        Args:
            cutoff: datetime to set (defaults to current 8am or previous day's 8am)
        """
        if cutoff is None:
            cutoff = self._calculate_current_8am_cutoff()

        self.state["last_8am_cutoff"] = cutoff.isoformat()
        self._save_state()

    def _calculate_current_8am_cutoff(self) -> datetime:
        """
        Calculate the current 8am cutoff.

        If it's currently after 8am today, use today at 8am.
        If it's currently before 8am today, use yesterday at 8am.

        Returns:
            datetime representing the most recent 8am
        """
        now = datetime.now()
        eight_am_today = datetime.combine(now.date(), time(8, 0, 0))

        if now >= eight_am_today:
            return eight_am_today
        else:
            return eight_am_today - timedelta(days=1)

    def get_cutoff_for_run(self, days_back: Optional[int] = None) -> Optional[datetime]:
        """
        Get the appropriate cutoff date for this digest run.

        If we have a saved last_8am_cutoff, use that.
        Otherwise, use days_back to calculate from current time.

        Args:
            days_back: Fallback number of days to look back if no saved state

        Returns:
            datetime to use as cutoff, or None for no cutoff
        """
        # Try to use saved state first
        last_cutoff = self.get_last_8am_cutoff()
        if last_cutoff:
            return last_cutoff

        # Fallback to days_back
        if days_back is not None:
            return datetime.now() - timedelta(days=days_back)

        return None

    def mark_run_complete(self):
        """
        Mark the current digest run as complete.

        Updates last_8am_cutoff to the current 8am boundary.
        """
        self.update_last_8am_cutoff()
        self.state["last_run"] = datetime.now().isoformat()
        self._save_state()

    def get_last_run(self) -> Optional[datetime]:
        """
        Get the timestamp of the last digest run.

        Returns:
            datetime of last run, or None if never run
        """
        last_run_str = self.state.get("last_run")
        if last_run_str:
            try:
                return datetime.fromisoformat(last_run_str)
            except ValueError:
                return None
        return None


def main():
    """Test state management."""
    state = DigestState()

    print("Current state:")
    print(f"  Last 8am cutoff: {state.get_last_8am_cutoff()}")
    print(f"  Last run: {state.get_last_run()}")
    print(f"  Current 8am boundary: {state._calculate_current_8am_cutoff()}")

    # Test getting cutoff for run
    cutoff = state.get_cutoff_for_run(days_back=7)
    print(f"\nCutoff for this run (with 7 days fallback): {cutoff}")

    # Simulate completing a run
    print("\nSimulating run completion...")
    state.mark_run_complete()
    print(f"  New last 8am cutoff: {state.get_last_8am_cutoff()}")
    print(f"  New last run: {state.get_last_run()}")


if __name__ == "__main__":
    main()
