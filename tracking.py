"""Per-object identity for the detection stream.

The detector is stateless: every frame it reports whatever it sees, with no
link back to the previous frame.  Downstream that forces the consumer to guess
which detection is which object, and the guess fails exactly when it matters --
a long object whose centroid jumps as it enters the frame spawns a duplicate
track, and the arm reaches for empty space.

This module assigns each detection a ``track_id`` that stays the same while the
object crosses the field of view, so the consumer never has to guess.

Conveyor geometry does the heavy lifting: every object shares one belt
velocity, so a single estimate (learned from the detections themselves, no
configuration) predicts where each track should appear next.  Matching is a
global assignment over the whole frame, not a per-track nearest neighbour, so
two objects passing close together cannot swap identities.

Pure numpy/scipy -- no camera, no model, no torch -- so it runs in unit tests.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# A detection farther than its gate from every prediction starts a new track.
INFEASIBLE = 1e6

# Class disagreement should only break ties between otherwise-similar matches:
# a metal/transparent flip on one frame must not tear a track in half.
CLASS_MISMATCH_COST = 0.25


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    last_t: float
    class_votes: dict = field(default_factory=dict)
    hits: int = 1
    missed: int = 0
    # Set on the frame of closest approach to the image centre -- the one
    # measurement of this object worth keeping (see ObjectTracker.update).
    # Latched, so the confirmation is emitted exactly once per object.
    center_confirmed: bool = False

    @property
    def class_name(self):
        return max(self.class_votes, key=self.class_votes.get) if self.class_votes else None

    def vote(self, class_name, weight):
        self.class_votes[class_name] = self.class_votes.get(class_name, 0.0) + float(weight)


class ObjectTracker:
    """Stateful nearest-prediction tracker for objects on a moving belt.

    ``update()`` takes one frame's detections and returns their track ids, in
    the same order.  Positions are belt-plane metres (whatever ``project()``
    produced), so the gates below are in metres and independent of resolution.
    """

    def __init__(
        self,
        base_gate_m: float = 0.06,
        drift_frac: float = 0.35,
        bootstrap_speed_mps: float = 0.60,
        max_missed: int = 4,
        max_age_s: float = 3.0,
        velocity_alpha: float = 0.35,
    ):
        # Gate = base + drift * |v| * dt.  The base covers centroid jitter on a
        # stationary object; the drift term covers error in the belt-speed
        # estimate, which grows with how long the track went unseen.
        self.base_gate_m = base_gate_m
        self.drift_frac = drift_frac
        # Until the belt speed is known, gate on a generous upper bound instead
        # of a velocity of zero -- otherwise every second frame spawns twins.
        self.bootstrap_speed_mps = bootstrap_speed_mps
        self.max_missed = max_missed
        self.max_age_s = max_age_s
        self.velocity_alpha = velocity_alpha

        self._tracks: list[Track] = []
        self._next_id = 0
        self._last_t = None
        self.velocity = None  # (vx, vy) m/s once learned, else None
        #: Per-detection flags from the last update(): True on the single frame
        #: where that object was closest to the image centre.
        self.last_confirmed: list[bool] = []

    @property
    def tracks(self):
        return list(self._tracks)

    def _gate(self, dt, velocity_known):
        speed = (
            float(np.hypot(*self.velocity)) if velocity_known else self.bootstrap_speed_mps
        )
        return self.base_gate_m + self.drift_frac * speed * max(dt, 0.0)

    def _predict(self, track, dt):
        if self.velocity is None:
            return track.x, track.y
        return track.x + self.velocity[0] * dt, track.y + self.velocity[1] * dt

    def update(
        self,
        centroids: Sequence[Sequence[float]],
        class_names: Sequence[str],
        confidences: Sequence[float],
        timestamp: float,
    ) -> list[int]:
        """Assign a track id to each detection; returns ids in input order."""
        n = len(centroids)
        if len(class_names) != n or len(confidences) != n:
            raise ValueError("centroids / class_names / confidences length mismatch")

        dt = 0.0 if self._last_t is None else max(float(timestamp) - self._last_t, 0.0)
        self._last_t = float(timestamp)

        # Retire tracks that have gone unseen too long before matching, so a
        # stale one cannot capture a fresh detection of a different object.
        self._tracks = [
            t
            for t in self._tracks
            if t.missed <= self.max_missed and (timestamp - t.last_t) <= self.max_age_s
        ]

        pts = np.asarray(centroids, dtype=float).reshape((n, 2))
        ids = [-1] * n
        confirmed = [False] * n
        pairs, unmatched_det = self._associate(pts, class_names, dt, timestamp)

        matched_tracks = set()
        deltas = []
        for track, j in pairs:
            deltas.append((pts[j, 0] - track.x, pts[j, 1] - track.y))
            track.x, track.y = float(pts[j, 0]), float(pts[j, 1])
            track.last_t = float(timestamp)
            track.hits += 1
            track.missed = 0
            track.vote(class_names[j], confidences[j])
            ids[j] = track.track_id
            confirmed[j] = self._confirm_closest_approach(track, dt)
            matched_tracks.add(id(track))

        for t in self._tracks:
            if id(t) not in matched_tracks:
                t.missed += 1

        for j in unmatched_det:
            track = Track(self._next_id, float(pts[j, 0]), float(pts[j, 1]), float(timestamp))
            track.vote(class_names[j], confidences[j])
            self._tracks.append(track)
            ids[j] = self._next_id
            self._next_id += 1

        self._update_velocity(deltas, dt)
        self.last_confirmed = confirmed
        return ids

    def _confirm_closest_approach(self, track, dt) -> bool:
        """True on the one frame where ``track`` is nearest the image centre.

        That frame is where this object's measurement is worth keeping: height
        parallax and lens distortion both vanish on the optical axis and grow
        with radius, and an object near the frame edge may still be partly
        outside it with its centroid pulled inward.

        The belt velocity is already known, so the moment is recognised as it
        happens rather than a frame late: if the object would be FARTHER from
        the centre next frame, it is closest now.  No look-ahead, no stale
        pose -- the confirmation carries this frame's own measurement and
        timestamp.
        """
        if track.center_confirmed or self.velocity is None or dt <= 0.0:
            return False
        r_now = float(np.hypot(track.x, track.y))
        r_next = float(np.hypot(track.x + self.velocity[0] * dt,
                                track.y + self.velocity[1] * dt))
        if r_next < r_now:
            return False
        track.center_confirmed = True
        return True

    def _associate(self, pts, class_names, dt, timestamp):
        """Global assignment of detections to predicted track positions."""
        if not self._tracks or len(pts) == 0:
            return [], list(range(len(pts)))

        known = self.velocity is not None
        cost = np.full((len(self._tracks), len(pts)), INFEASIBLE)
        for i, track in enumerate(self._tracks):
            age = max(float(timestamp) - track.last_t, 0.0)
            px, py = self._predict(track, age if age > 0 else dt)
            gate = self._gate(age if age > 0 else dt, known)
            for j in range(len(pts)):
                d = float(np.hypot(pts[j, 0] - px, pts[j, 1] - py)) / gate
                if d < 1.0:
                    cost[i, j] = d + (
                        CLASS_MISMATCH_COST if class_names[j] != track.class_name else 0.0
                    )

        rows, cols = linear_sum_assignment(cost)
        pairs = [
            (self._tracks[i], int(j))
            for i, j in zip(rows, cols)
            if cost[i, j] < INFEASIBLE
        ]
        taken = {j for _, j in pairs}
        return pairs, [j for j in range(len(pts)) if j not in taken]

    def _update_velocity(self, deltas, dt):
        """Learn one belt velocity shared by every object.

        The median over this frame's matches rejects the odd centroid jump (a
        half-visible object at the frame edge) that would otherwise drag the
        estimate; the EMA then smooths across frames.
        """
        if not deltas or dt <= 0.0:
            return
        obs = np.median(np.asarray(deltas, dtype=float), axis=0) / dt
        if self.velocity is None:
            self.velocity = (float(obs[0]), float(obs[1]))
            return
        a = self.velocity_alpha
        self.velocity = (
            float((1 - a) * self.velocity[0] + a * obs[0]),
            float((1 - a) * self.velocity[1] + a * obs[1]),
        )
