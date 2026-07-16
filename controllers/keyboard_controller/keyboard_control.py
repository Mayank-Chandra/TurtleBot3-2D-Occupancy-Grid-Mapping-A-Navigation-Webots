from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Grab TurtleBot3 motor devices
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# Configure for velocity control
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Set up keyboard listening
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

print("--- KEYBOARD CONTROLLER RUNNING ---")
print("Click in the 3D window and use W, A, S, D to drive.")

while robot.step(timestep) != -1:
    key = keyboard.getKey()
    left_speed = 0.0
    right_speed = 0.0
    
    if key == ord('W') or key == ord('w'):
        left_speed, right_speed = 4.0, 4.0
    elif key == ord('S') or key == ord('s'):
        left_speed, right_speed = -4.0, -4.0
    elif key == ord('A') or key == ord('a'):
        left_speed, right_speed = -2.0, 2.0
    elif key == ord('D') or key == ord('d'):
        left_speed, right_speed = 2.0, -2.0
        
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)