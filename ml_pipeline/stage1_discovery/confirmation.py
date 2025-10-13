"""
Stage 1.5: Interactive Confirmation

Displays bucket selection summary and prompts user for approval.

Source: VideoDiscoveryCHILDTI.md Section 4.5
"""

import logging
from typing import Dict

from .constants import EXIT_CODE_USER_ABORT


logger = logging.getLogger(__name__)


class InteractiveConfirmation:
    """
    Handles interactive confirmation prompt after bucket selection.

    Implements:
    - Bucket summary display
    - User approval prompt
    - Auto-confirm mode support
    - User abort handling
    """

    def confirm_bucket_selection(
        self,
        selected_per_bucket: Dict[str, Dict],
        auto_confirm: bool = False
    ) -> bool:
        """
        Display bucket selection summary and prompt user for approval.

        Args:
            selected_per_bucket: Selected videos per bucket
            auto_confirm: Skip prompt if True (CI/CD mode)

        Returns:
            True if user confirms, False if user aborts

        Raises:
            KeyboardInterrupt: If user presses Ctrl+C
        """
        # Display summary
        self._display_summary(selected_per_bucket)

        # Skip prompt if auto-confirm enabled
        if auto_confirm:
            logger.info("Auto-confirm enabled. Proceeding without user prompt.")
            return True

        # Prompt user
        return self._prompt_user()

    def _display_summary(self, selected_per_bucket: Dict[str, Dict]):
        """
        Display bucket selection summary in human-readable format.
        """
        print("\n" + "="*80)
        print("STAGE 1: VIDEO DISCOVERY COMPLETE")
        print("="*80)
        print(f"\nSelected {len(selected_per_bucket)} winning bucket(s) for processing:\n")

        # Calculate totals
        total_videos = 0
        total_top = 0
        total_bottom = 0

        for bucket_name in sorted(selected_per_bucket.keys()):
            selection = selected_per_bucket[bucket_name]
            total_videos += selection['selected_count']
            total_top += selection['top_count']
            total_bottom += selection['bottom_count']

            print(f"  Bucket: {selection['bucket']}")
            print(f"    Strategy: {selection['strategy']}")
            print(f"    Selected: {selection['selected_count']} videos")
            print(f"      - Top performers: {selection['top_count']}")
            if selection['bottom_count'] > 0:
                print(f"      - Bottom performers: {selection['bottom_count']}")
            print()

        print("-"*80)
        print(f"TOTAL: {total_videos} videos selected across {len(selected_per_bucket)} bucket(s)")
        print(f"  - Top performers: {total_top}")
        if total_bottom > 0:
            print(f"  - Bottom performers: {total_bottom}")
        print("="*80)

    def _prompt_user(self) -> bool:
        """
        Prompt user for confirmation.

        Returns:
            True if user confirms (y/yes), False if user aborts (n/no)
        """
        print("\nNext Steps:")
        print("  - Stage 2 will download and process these videos through RumiAI")
        print("  - Processing time: ~60-80 seconds per video")
        print(f"  - Estimated total time: See above summary")
        print()

        try:
            response = input("Proceed with video processing? (y/n): ").strip().lower()

            if response in ['y', 'yes']:
                logger.info("User confirmed. Proceeding to Stage 2.")
                print("\n✓ Confirmed. Proceeding to Stage 2...\n")
                return True
            elif response in ['n', 'no']:
                logger.info("User aborted at confirmation prompt.")
                print("\n✗ Aborted by user.\n")
                return False
            else:
                print(f"\nInvalid response: '{response}'. Please enter 'y' or 'n'.")
                return self._prompt_user()  # Retry

        except KeyboardInterrupt:
            logger.info("User aborted with Ctrl+C.")
            print("\n\n✗ Aborted by user (Ctrl+C).\n")
            raise
        except EOFError:
            logger.info("User aborted with EOF (Ctrl+D).")
            print("\n\n✗ Aborted by user (EOF).\n")
            return False
