# Controller session recovery

Stopping collection used to destroy the virtual controller while Cuphead was
still running. The memory runner also terminated its launched game during
cleanup. Either behavior could interrupt play; controller removal could leave
the game waiting for its assigned controller. These are confirmed code paths,
not a confirmed diagnosis of any particular past lockout.

## Starting a Windows session

1. Close Cuphead normally before starting an AI run.
2. Run one script with its own `--launch` argument pointing directly to
   `Cuphead.exe`. For the standalone launcher, put that executable after `--`.
   Steam URLs, shell wrappers and launcher executables are not supported for
   this Windows path because their exit does not establish that Cuphead exited.
3. Keep the script's terminal open. Do not start the standalone launcher and
   then start an agent: each would create a different virtual controller.
4. Confirm the character and HUD show the intended player before collecting
   data. Physical controllers and game bindings can still affect assignment.

The native controller factory now holds an OS file lock for the device lifetime.
A second Cuphead AI process for the same user fails before creating another
controller. The OS releases the lock when its owner exits; do not delete the
lock file to bypass a running owner.

## Stopping and recovering

On completion, a time limit, focus loss in the capture pilot, a normal exception,
or Ctrl+C, the live scripts release input. A directly launched gamepad session
then waits with its controller connected until Cuphead closes. The capture pilot
and memory runner write their summaries before this wait. Repeated Ctrl+C while
waiting leaves the controller connected. No runner forcibly kills the game.

Close Cuphead through its menu or normal window close action (Alt+F4 on Windows)
to let the script finish. To start another run, let the old session finish first.
If an older script already disconnected its controller and the game will not
respond, close Cuphead normally and start a fresh session with the updated script.
These changes do not alter saves, controller bindings, drivers or game files.

The wait intentionally has no automatic disconnect timeout: a timeout would
reintroduce removal of the controller from a running game. Killing the Python
process, closing its terminal, a driver crash or a machine shutdown can still
disconnect the device. Hardware-free tests cannot guarantee player assignment,
keyboard takeover or recovery in the installed game. Live verification remains
necessary before claiming lockouts are eliminated.

## Regression checks

Run `python -m unittest discover -s tests -v`. The session tests cover direct
launch validation, existing-game rejection, duplicate ownership across processes,
launch and capture failures, interruptions, neutral retries and cleanup ordering.
They use simulated devices and processes and never start Cuphead.
