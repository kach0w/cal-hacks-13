"""Change gate: only run the (expensive) detector when the scene changed.

The reference is the last frame detection actually ran on, so a permanent change triggers once
instead of forever. After any change keep detecting for SETTLE_S — otherwise an object that
vanished would get one miss and never reach MISSING_AFTER_K. Detect anyway every HEARTBEAT_S.
"""
import cv2


class ChangeGate:
    def __init__(self, cfg):
        self.width, self.delta, self.fraction = cfg.change_width, cfg.change_pixel_delta, cfg.change_fraction
        self.settle_s, self.heartbeat_s = cfg.settle_s, cfg.heartbeat_s
        self.ref = None
        self.hot_until = 0.0
        self.last_detect = 0.0
        self.last_fraction = 0.0   # for tuning CHANGE_FRACTION: logged with each pass

    def _small(self, image):
        h, w = image.shape[:2]
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        g = cv2.resize(g, (self.width, max(1, int(h * self.width / w))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (5, 5), 0)

    def check(self, image, now) -> tuple[bool, str]:
        """-> (run detection?, why): first | change | settling | heartbeat | static"""
        small = self._small(image)
        if self.ref is None or self.ref.shape != small.shape:
            return True, "first"
        self.last_fraction = float((cv2.absdiff(small, self.ref) > self.delta).mean())
        if self.last_fraction >= self.fraction:
            self.hot_until = now + self.settle_s
            return True, "change"
        if now < self.hot_until:
            return True, "settling"
        if now - self.last_detect >= self.heartbeat_s:
            return True, "heartbeat"
        return False, "static"

    def mark_detected(self, image, now):
        self.ref = self._small(image)
        self.last_detect = now
