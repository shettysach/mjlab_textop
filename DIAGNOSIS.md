Fix the confirmed live reference discontinuity with the smallest change.

The current rolling buffer evicts frames based on `latest_index`. This allows a fast producer to delete the frame MJLab is still consuming.

Make these changes:

1. Remove automatic max-frame eviction from `RollingMotionBuffer.append_block()`.

2. Add a buffer method that deletes frames before a supplied index.

3. After `OnlineMotionCommand.current_frame` successfully advances, delete only frames older than `current_frame`.

4. Keep the strict live advancement behavior:

   * advance by exactly one frame when the next complete future window exists;
   * otherwise hold the current frame;
   * never jump forward to the latest buffered frame;
   * never re-anchor the reference during normal live playback.

5. `max_buffer_frames` should no longer mean “retain the newest N producer frames.” It may be removed for now or later redefined around consumer-relative retention.

Do not add a new transport protocol or backpressure system in this change. Do not require additional block validation unless it is a simple assertion. The goal is only to ensure that incoming future frames can never evict the active consumer frame.
