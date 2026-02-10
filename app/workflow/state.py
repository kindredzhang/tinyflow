"""Workflow state management."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

StateType = TypeVar("StateType")


class WorkflowState(Generic[StateType]):
    """Lightweight state manager for workflows with validation and snapshots."""

    def __init__(self, initial_state: Optional[StateType] = None):
        self._state: StateType = initial_state if initial_state is not None else {}  # type: ignore
        self._id: str = str(uuid.uuid4())
        self._snapshot_history: List[Dict[str, Any]] = []
        self._logger = logging.getLogger("tinyflow.workflow.state")

    @property
    def state(self) -> StateType:
        """Get the current state."""
        return self._state

    @property
    def id(self) -> str:
        """Get the workflow instance ID."""
        return self._id

    def __getitem__(self, key: str) -> Any:
        """Dict-style access to state."""
        if hasattr(self._state, "__getitem__"):
            return self._state[key]  # type: ignore
        return getattr(self._state, key)

    def __setitem__(self, key: str, value: Any):
        """Dict-style setting of state."""
        if hasattr(self._state, "__setitem__"):
            self._state[key] = value  # type: ignore
        else:
            setattr(self._state, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value with default.

        Logs a warning if the key doesn't exist and default is used.
        """
        if self.exists(key):
            return self[key]

        self._logger.warning(f"Key '{key}' not found in state, using default: {default}")
        return default

    def update(self, **kwargs) -> None:
        """Update state with multiple values."""
        if isinstance(self._state, dict):
            self._state.update(kwargs)
        else:
            for k, v in kwargs.items():
                setattr(self._state, k, v)

    def exists(self, key: str) -> bool:
        """Check if a key exists in the state.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        if isinstance(self._state, dict):
            return key in self._state
        return hasattr(self._state, key)

    def snapshot(self) -> str:
        """Create a snapshot of the current state.

        Returns:
            Unique snapshot ID
        """
        snapshot_id = str(uuid.uuid4())
        if hasattr(self._state, "model_dump"):
            state_snapshot = getattr(self._state, "model_dump")()
        elif isinstance(self._state, dict):
            state_snapshot = dict(self._state)
        else:
            # Try to convert object to dict if possible, otherwise store as is
            try:
                state_snapshot = vars(self._state)
            except TypeError:
                state_snapshot = self._state

        snapshot = {
            "id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "state": state_snapshot,
        }
        self._snapshot_history.append(snapshot)
        self._logger.debug(f"Created snapshot {snapshot_id}")
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot_id: The ID of the snapshot to restore

        Raises:
            ValueError: If snapshot_id is not found
        """
        for snapshot in self._snapshot_history:
            if snapshot["id"] == snapshot_id:
                self._state = snapshot["state"]
                self._logger.info(f"Restored state from snapshot {snapshot_id}")
                return
        raise ValueError(f"Snapshot with id '{snapshot_id}' not found")

    def validate(self) -> List[str]:
        """Validate the current state and return any errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(self._state, dict):
            return errors

        for key, value in self._state.items():
            # Check for None values
            if value is None:
                errors.append(f"Key '{key}' has None value")
            # Check for empty strings
            elif isinstance(value, str) and not value.strip():
                errors.append(f"Key '{key}' has empty string value")
            # Check for empty collections
            elif isinstance(value, (list, dict)) and len(value) == 0:
                errors.append(f"Key '{key}' has empty collection")

        return errors

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the snapshot history.

        Returns:
            List of all snapshots taken
        """
        return self._snapshot_history.copy()
