from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Enable keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

print("Test Controller active. Click in the 3D window and press W, A, S, or D.")

while robot.step(timestep) != -1:
    key = keyboard.getKey()
    left_speed = 0.0
    right_speed = 0.0

    # LOWERCASE checks (or check for both upper and lower)
    if key == ord('W') or key == ord('w'):      # Forward
        left_speed, right_speed = 4.0, 4.0
    elif key == ord('S') or key == ord('s'):    # Backward
        left_speed, right_speed = -4.0, -4.0
    elif key == ord('A') or key == ord('a'):    # Left
        left_speed, right_speed = -2.0, 2.0
    elif key == ord('D') or key == ord('d'):    # Right
        left_speed, right_speed = 2.0, -2.0

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)