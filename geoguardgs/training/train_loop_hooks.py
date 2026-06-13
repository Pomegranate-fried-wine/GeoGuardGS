"""Hook names used by GeoGuardGS training integration.

The current release keeps the primary training implementation in the
StreetGS-compatible train entrypoint. These hook names document where the
periodic feedback controller and Gaussian control manager are expected to run.
"""


BEFORE_ITERATION = "before_iteration"
AFTER_RENDER = "after_render"
AFTER_FEEDBACK_TRIGGER = "after_feedback_trigger"
BEFORE_LOSS_BACKWARD = "before_loss_backward"
AFTER_ITERATION = "after_iteration"
