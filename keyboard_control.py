from dorna import Dorna
import sys, select, termios, tty
import traceback

msg = """
Reading from the keyboard to control Dorna Arm!
---------------------------
CTRL-C to quit
"""

moveBindings = {
        # x control
        'q': (1,0,0),
        'a': (-1,0,0),
        # y control
        'w': (0,1,0),
        's': (0,-1,0),
        # z control
        'e': (0,0,1),
        'd': (0,0,-1),
    }

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def generate_command(move_cmd, movement=1, coord_sys='xyz'):
    """ 
    Using the cartesian system, X, Y, and Z represent the position of the head of the robot. A represents the angle of the 
    tooltip (j1+j2+j3) with the xy plane and B is j4
    """
    x, y, z = move_cmd
    if coord_sys == 'xyz':
        cmd =  {"command": "move", "prm": {'path': 'joint', 'movement': movement, 'xyz': [x, y, z, 0, 0]}}
    elif coord_sys == 'joint':
        cmd = {"command": "move", "prm": {'path': 'joint', 'movement': movement, 'joint': [0, 0, 0, 0, 0]}}
    else:
        raise ValueError("Invalid coord sys!")
    return cmd

if __name__=="__main__":
    # Dorna arm initialisation
    robot = Dorna("/media/feru/HDD/arm_control/myconfig.json")
    robot.set_unit({"length": "mm"})
    robot.set_toolhead({"x": 80})  # Set gripper tip length for IK
    mv_scale = 3
    print(robot.connect())

    # Keyboard control loop
    settings = termios.tcgetattr(sys.stdin)
    try:
        print(msg)
        while(1):
            key = getKey()
            if key in moveBindings.keys():
                dxyz = tuple([mv_scale * param for param in moveBindings[key]])
                command = generate_command(dxyz)
                print(command)
                robot.play(command)
                # if (status == 14):
                #     print(msg)
                # status = (status + 1) % 15
            else:
                dxyz = (0, 0, 0)

                if (key == '\x03'):
                    break
                elif key == 'p': # stop robot
                    robot.halt()
                elif key == 'o':  # check position status
                    print(robot.position("xyz"))
                elif key == 'r': # try to re-connect in case of connection missing
                    robot.connect()
                elif key == 'h':
                    for joint in ["j3", "j2", "j1", "j0"]:
                        robot.home(joint)
                        print("{} homed".format(joint))
                elif key == 'g':
                    dxyz = [0, 0, 0]
                    command = generate_command(dxyz, movement=0, coord_sys='joint')
                    print(command)
                    robot.play(command)
                    print("Robot at 0")
                elif key == 'c':
                    robot.calibrate([0, 0, 0, 0 ,0])
                    print("Robot calibrated")
                elif key == 't':
                    robot.terminate()
                    print("Robot terminated")

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        robot.terminate()
        print("Robot terminated")